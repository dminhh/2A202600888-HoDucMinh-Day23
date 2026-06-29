"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    lines = [
        "# Lab Report — Day 08 LangGraph Agentic Orchestration",
        "",
        "## 1. Metrics Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total scenarios | {metrics.total_scenarios} |",
        f"| Success rate | {metrics.success_rate:.1%} |",
        f"| Avg nodes visited | {metrics.avg_nodes_visited:.1f} |",
        f"| Total retries | {metrics.total_retries} |",
        f"| Total interrupts (approvals) | {metrics.total_interrupts} |",
        f"| Resume success | {metrics.resume_success} |",
        "",
        "## 2. Per-Scenario Results",
        "",
        "| Scenario ID | Expected Route | Actual Route | Success | Retries | Approval | Errors |",
        "|---|---|---|---|---|---|---|",
    ]

    for sm in metrics.scenario_metrics:
        success_mark = "✓" if sm.success else "✗"
        approval_mark = "✓" if sm.approval_observed else "-"
        error_count = len(sm.errors)
        lines.append(
            f"| {sm.scenario_id} | {sm.expected_route} | {sm.actual_route or 'N/A'} "
            f"| {success_mark} | {sm.retry_count} | {approval_mark} | {error_count} |"
        )

    lines += [
        "",
        "## 3. Architecture",
        "",
        "### Graph Design",
        "The workflow uses a `StateGraph` with 11 nodes and conditional routing:",
        "",
        "```",
        "START → intake → classify → [route_after_classify]",
        "  simple       → answer → finalize → END",
        "  tool         → tool → evaluate → [route_after_evaluate]",
        "                   needs_retry → retry → [route_after_retry]",
        "                                  tool (retry loop, bounded)",
        "                                  dead_letter → finalize → END",
        "                   success → answer → finalize → END",
        "  missing_info → clarify → finalize → END",
        "  risky        → risky_action → approval → [route_after_approval]",
        "                   approved → tool → evaluate → ...",
        "                   rejected → clarify → finalize → END",
        "  error        → retry → [route_after_retry] → ...",
        "```",
        "",
        "### State Schema",
        "- **Overwrite fields**: `route`, `risk_level`, `attempt`, `final_answer`,",
        "  `evaluation_result`, `pending_question`, `proposed_action`, `approval`",
        "- **Append-only fields** (using `Annotated[list, add]` reducer):",
        "  `messages`, `tool_results`, `errors`, `events`",
        "",
        "### LLM Integration",
        "- `classify_node`: uses `llm.with_structured_output(Classification)` for reliable enum routing",
        "- `answer_node`: uses `llm.invoke()` to generate grounded responses from tool results and context",
        "",
        "## 4. Failure Analysis",
        "",
        "**Failure Mode 1 — Dead Letter (S07)**",
        "Scenario S07 sets `max_attempts=1`. After the first tool error, `retry_or_fallback_node`",
        "increments `attempt` to 1. `route_after_retry` checks `attempt < max_attempts` (1 < 1 = False),",
        "so routes to `dead_letter_node`, which sets a final escalation message.",
        "",
        "**Failure Mode 2 — Missing Info (S03)**",
        "Query 'Can you fix it?' is too vague for any action. The LLM classifies it as `missing_info`.",
        "`route_after_classify` sends it to `ask_clarification_node`, which sets `pending_question`",
        "instead of `final_answer`. The route terminates at `clarify → finalize → END`.",
        "",
        "**Failure Mode 3 — Transient Tool Error with Retry (S05)**",
        "Error-route scenarios simulate connection timeouts for `attempt < 2`. After two retries,",
        "`tool_node` returns a success result, `evaluate_node` sets `evaluation_result=success`,",
        "and `route_after_evaluate` proceeds to `answer_node`.",
        "",
        "## 5. Improvement Ideas",
        "",
        "1. **LLM-as-judge in `evaluate_node`**: Replace heuristic ERROR string check with an",
        "   LLM call that evaluates result quality semantically.",
        "2. **Parallel fan-out**: Use `Send()` API for concurrent tool calls when multiple",
        "   lookups are needed in a single query.",
        "3. **Real HITL**: Set `LANGGRAPH_INTERRUPT=true` and use `interrupt()` in `approval_node`",
        "   for production human-in-the-loop workflows.",
        "4. **Time travel**: Use `get_state_history()` to replay and debug failed scenarios.",
        "5. **SQLite persistence**: Wire SQLite checkpointer to survive process crashes",
        "   and resume from last checkpoint.",
    ]

    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
