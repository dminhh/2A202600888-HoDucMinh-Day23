"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None):
    """Build and compile the complete LangGraph workflow."""
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("intake",        intake_node)
    g.add_node("classify",      classify_node)
    g.add_node("tool",          tool_node)
    g.add_node("evaluate",      evaluate_node)
    g.add_node("answer",        answer_node)
    g.add_node("clarify",       ask_clarification_node)
    g.add_node("risky_action",  risky_action_node)
    g.add_node("approval",      approval_node)
    g.add_node("retry",         retry_or_fallback_node)
    g.add_node("dead_letter",   dead_letter_node)
    g.add_node("finalize",      finalize_node)

    # Fixed edges
    g.add_edge(START,          "intake")
    g.add_edge("intake",       "classify")
    g.add_edge("tool",         "evaluate")
    g.add_edge("risky_action", "approval")
    g.add_edge("answer",       "finalize")
    g.add_edge("clarify",      "finalize")
    g.add_edge("dead_letter",  "finalize")
    g.add_edge("finalize",     END)

    # Conditional edges
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "answer":       "answer",
            "tool":         "tool",
            "clarify":      "clarify",
            "risky_action": "risky_action",
            "retry":        "retry",
        },
    )
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "retry", "answer": "answer"},
    )
    g.add_conditional_edges(
        "retry",
        route_after_retry,
        {"tool": "tool", "dead_letter": "dead_letter"},
    )
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"tool": "tool", "clarify": "clarify"},
    )

    return g.compile(checkpointer=checkpointer)
