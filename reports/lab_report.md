# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / Student

- **Name**: Ho Duc Minh
- **Student ID**: 2A202600888
- **Date**: 2026-06-29

---

## 2. Architecture

The workflow is a `StateGraph` with **11 nodes** and **4 conditional routing functions**. All paths terminate at `finalize → END`.

```
START → intake → classify → [route_after_classify]
  ├─ simple       → answer → finalize → END
  ├─ tool         → tool → evaluate → [route_after_evaluate]
  │                   ├─ needs_retry → retry → [route_after_retry]
  │                   │                  ├─ attempt < max → tool  (bounded loop)
  │                   │                  └─ attempt >= max → dead_letter → finalize → END
  │                   └─ success → answer → finalize → END
  ├─ missing_info → clarify → finalize → END
  ├─ risky        → risky_action → approval → [route_after_approval]
  │                   ├─ approved → tool → evaluate → ...
  │                   └─ rejected → clarify → finalize → END
  └─ error        → retry → [route_after_retry] → ...
```

### Node Responsibilities

| Node | Purpose | LLM |
|---|---|---|
| `intake_node` | Normalize/trim raw query | No |
| `classify_node` | Classify intent into route | **Yes — structured output** |
| `tool_node` | Mock tool with error simulation for retry testing | No |
| `evaluate_node` | Retry-loop gate — check tool result quality | No |
| `answer_node` | Generate grounded response | **Yes** |
| `ask_clarification_node` | Request missing info from user | No |
| `risky_action_node` | Describe proposed action, stage for approval | No |
| `approval_node` | Mock HITL approval (auto-approved offline) | No |
| `retry_or_fallback_node` | Increment attempt counter, log error | No |
| `dead_letter_node` | Handle max-retry exhaustion, escalate | No |
| `finalize_node` | Emit final audit event — all routes pass here | No |

---

## 3. State Schema

`AgentState` uses `TypedDict` with two reducer strategies:

| Field | Reducer | Why |
|---|---|---|
| `query` | overwrite | Normalized once at intake |
| `route` | overwrite | Only current classification matters |
| `risk_level` | overwrite | Set by classify, read by risky_action |
| `attempt` | overwrite | Counter incremented by retry node |
| `max_attempts` | overwrite | Fixed per scenario from scenarios.jsonl |
| `final_answer` | overwrite | Last generated answer wins |
| `evaluation_result` | overwrite | Latest evaluate decision gates retry loop |
| `pending_question` | overwrite | Latest clarification question to user |
| `proposed_action` | overwrite | Risky action description staged for approval |
| `approval` | overwrite | HITL decision dict `{approved, reviewer, comment}` |
| `messages` | **append** (`add`) | Full audit trail of all node visits |
| `tool_results` | **append** (`add`) | All tool outputs including retry attempts |
| `errors` | **append** (`add`) | All error messages across retries |
| `events` | **append** (`add`) | Structured `LabEvent` dicts for metrics extraction |

Append-only fields use `Annotated[list, add]` — LangGraph merges them automatically without explicit mutation.

---

## 4. Scenario Results

All 7 scenarios pass with **100% success rate**.

| Scenario | Expected Route | Actual Route | Success | Nodes | Retries | Approval | Latency |
|---|---|---|:---:|:---:|:---:|:---:|---:|
| S01_simple | simple | simple | ✓ | 4 | 0 | — | ~5400ms |
| S02_tool | tool | tool | ✓ | 6 | 0 | — | ~2500ms |
| S03_missing | missing_info | missing_info | ✓ | 4 | 0 | — | ~1200ms |
| S04_risky | risky | risky | ✓ | 8 | 0 | ✓ | ~2600ms |
| S05_error | error | error | ✓ | 10 | 2 | — | ~2600ms |
| S06_delete | risky | risky | ✓ | 8 | 0 | ✓ | ~2500ms |
| S07_dead_letter | error | error | ✓ | 5 | 1 | — | ~1100ms |

**Summary** (`outputs/metrics.json`):

| Metric | Value |
|---|---|
| total_scenarios | 7 |
| success_rate | 100% |
| avg_nodes_visited | 6.4 |
| total_retries | 3 |
| total_interrupts | 2 |
| resume_success | true |

**Latency notes**: S01 is slowest (~5400ms) as it's the first call initializing the LLM client connection. S03 and S07 are fastest (~1100ms) because they don't call `answer_node` (no second LLM call).

---

## 5. Failure Analysis

### Failure Mode 1 — Dead Letter Queue (S07)

S07 sets `max_attempts=1` to simulate a completely unrecoverable failure:

1. LLM classifies "System failure cannot recover..." → `error`
2. `route_after_classify`: `error` → `retry` (attempt becomes 1)
3. `route_after_retry`: `attempt (1) < max_attempts (1)` → **False** → `dead_letter`
4. `dead_letter_node` sets `final_answer` with escalation message — no tool ever executes

**Key design**: routing to `retry` before `tool` means the counter increments first. This correctly exhausts `max_attempts=1` immediately without wasting a tool call.

### Failure Mode 2 — Transient Tool Error with Bounded Retry (S05)

S05 simulates a connection timeout that clears after 2 attempts:

1. `error` → `retry` (attempt=1) → `tool` → ERROR string → `evaluate` → `needs_retry`
2. → `retry` (attempt=2) → `tool` → success (attempt ≥ 2) → `evaluate` → `success`
3. → `answer` → `finalize`

**Bounded guarantee**: `route_after_retry` always checks `attempt < max_attempts`. Without this bound, the loop never terminates.

### Failure Mode 3 — Missing Information (S03)

"Can you fix it?" has no actionable context. LLM classifies it `missing_info` → `ask_clarification_node` sets `pending_question` instead of `final_answer`. Graph terminates at `clarify → finalize` without calling any tool or fabricating an answer.

---

## 6. Persistence / Recovery Evidence

**Checkpointer**: SQLite via `langgraph.checkpoint.sqlite.SqliteSaver`

**Config** (`configs/lab.yaml`):
```yaml
checkpointer: sqlite
database_url: outputs/langgraph.db
```

**Implementation** (`persistence.py`):
```python
conn = sqlite3.connect(db_path, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
return SqliteSaver(conn=conn)
```

**Per-run thread isolation**: Each scenario gets a unique `thread_id` with UUID suffix (e.g., `thread-S01_simple-a3f7c2b1`) to prevent checkpoint accumulation across multiple `make run-scenarios` invocations.

**Crash-resume demonstration**:
```
Run 1: graph.invoke(state) → SQLite saves checkpoint per node → connection closed
Run 2: new SqliteSaver(conn) → graph.get_state_history(thread_cfg)
       → 12 checkpoints recovered
       → latest state: route="simple", final_answer present ✓
```

`resume_success=true` is set in `metrics.json` when `get_state_history()` returns non-empty history after a fresh connection, proving state survives process termination.

---

## 7. Extension Work

### SQLite Persistence (completed)

- Implemented `build_checkpointer(kind="sqlite")` in `persistence.py`
- WAL mode (`PRAGMA journal_mode=WAL`) for safe concurrent writes
- Graph compiled with `checkpointer=checkpointer` — every node execution auto-checkpointed
- Unique `thread_id` per run prevents state bleed between runs
- Crash-resume verified: close connection → reopen → `get_state_history()` returns full history
- `resume_success=True` automatically detected and recorded in `metrics.json`
- Real `latency_ms` measured with `time.perf_counter()` per scenario

---

## 8. Improvement Plan

1. **LLM-as-judge in `evaluate_node`**: Replace `"ERROR" in string` heuristic with a structured LLM call evaluating tool result quality semantically — handles partial results that are technically not errors but insufficient.

2. **Real HITL with `interrupt()`**: Use `langgraph.types.interrupt()` in `approval_node` when `LANGGRAPH_INTERRUPT=true`. Build a Streamlit UI for reviewers to approve/reject with comments. Currently all risky actions auto-approve — unacceptable in production.

3. **Parallel fan-out with `Send()`**: For queries requiring multiple lookups, dispatch concurrent tool calls using the `Send()` API and merge results — reduces latency from O(n) sequential to O(1) parallel.

4. **Time travel replay**: Use `graph.invoke(None, config={"configurable": {"checkpoint_id": ...}})` to re-run from any historical checkpoint — essential for debugging production failures without full re-execution.

5. **Postgres persistence**: Switch from SQLite to Postgres for production multi-process deployments where WAL-mode SQLite is insufficient.
