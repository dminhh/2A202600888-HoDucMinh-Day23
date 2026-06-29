"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Pydantic model for structured LLM output ────────────────────────
class Classification(BaseModel):
    route: str      # one of: simple, tool, missing_info, risky, error
    risk_level: str  # "high" or "low"
    reasoning: str


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    llm = get_llm()
    query = state.get("query", "")

    prompt = f"""You are a customer support routing system. Classify the following support query into exactly one category.

Priority order (apply the HIGHEST matching priority):
1. risky — actions with irreversible side effects: refunds, deletions, sending emails, cancellations, account modifications
2. tool — information lookups requiring data retrieval: order status, tracking, account info, search
3. missing_info — vague or incomplete queries that lack enough context to process
4. error — system-level failures: timeouts, crashes, service unavailable, cannot recover
5. simple — general FAQ questions answerable without tools or risky actions

Rules:
- If the query mentions refund, delete, cancel, send email, or modify account → risky
- If the query asks to look up, check, find, or retrieve information → tool
- If the query is too vague to act on → missing_info
- If the query describes a system error or failure → error
- Otherwise → simple

Set risk_level to "high" if route is "risky", otherwise "low".

Query: {query}"""

    structured_llm = llm.with_structured_output(Classification)
    result = structured_llm.invoke(prompt)

    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "events": [make_event("classify", "completed", f"route={result.route} risk={result.risk_level}")],
        "messages": [f"classify:{result.route}"],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient error simulation for retry testing."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    # Simulate transient failure for error-route scenarios until attempt >= 2
    if route == "error" and attempt < 2:
        result = f"ERROR: Tool execution failed on attempt {attempt} — connection timeout"
    else:
        result = (
            f"Tool result for '{query[:50]}': "
            "Record found. Order #12345 — Status: shipped, ETA: 2 business days."
        )

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", result[:80])],
        "messages": [f"tool:attempt={attempt}"],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate. Uses heuristic check."""
    results = state.get("tool_results", [])

    if not results:
        evaluation = "needs_retry"
    else:
        latest = results[-1]
        evaluation = "needs_retry" if "ERROR" in latest else "success"

    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", evaluation)],
        "messages": [f"evaluate:{evaluation}"],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM grounded in available context."""
    llm = get_llm()
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context_parts = [f"Customer query: {query}"]
    if tool_results:
        context_parts.append(f"Retrieved information: {tool_results[-1]}")
    if approval:
        reviewer = approval.get("reviewer", "system")
        context_parts.append(f"Action approved by: {reviewer}. Comment: {approval.get('comment', '')}")

    prompt = (
        "You are a helpful customer support agent. "
        "Generate a concise, professional response to the customer.\n\n"
        + "\n".join(context_parts)
    )

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content,
        "events": [make_event("answer", "completed", "llm response generated")],
        "messages": ["answer:generated"],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        f"I'd be happy to help, but I need a bit more information. "
        f"Your message '{query[:60]}' doesn't have enough detail. "
        "Could you please clarify: What specific issue are you experiencing, "
        "and what account or order number is involved?"
    )

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification question sent")],
        "messages": ["clarify:question_sent"],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    proposed = (
        f"PROPOSED ACTION: {query}\n"
        f"Risk level: {risk_level}. "
        "This action has irreversible side effects and requires explicit human approval before execution."
    )

    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "completed", "action staged for approval")],
        "messages": ["risky_action:staged"],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step. Defaults to mock approval for offline testing."""
    decision = {
        "approved": True,
        "reviewer": "mock-auto-approver",
        "comment": "Auto-approved in offline/test mode",
    }

    if os.getenv("LANGGRAPH_INTERRUPT") == "true":
        from langgraph.types import interrupt
        interrupt("Waiting for human approval of: " + state.get("proposed_action", ""))

    return {
        "approval": decision,
        "events": [make_event("approval", "completed", f"approved={decision['approved']}")],
        "messages": [f"approval:{decision['approved']}"],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt — increment counter and log the failure."""
    attempt = state.get("attempt", 0) + 1
    tool_results = state.get("tool_results", [])
    last_error = tool_results[-1] if tool_results else "no result available"

    return {
        "attempt": attempt,
        "errors": [f"attempt {attempt}: {last_error[:100]}"],
        "events": [make_event("retry", "retry", f"attempt={attempt}")],
        "messages": [f"retry:attempt={attempt}"],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    max_attempts = state.get("max_attempts", 3)
    answer = (
        f"We were unable to complete your request after {max_attempts} attempts due to a system error. "
        "Your case has been escalated to our senior support team who will contact you within 24 hours."
    )

    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "error", "max retries exceeded — escalated")],
        "messages": ["dead_letter:escalated"],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
        "messages": ["finalize:done"],
    }
