"""Deterministic routing for the local customer-support assistant."""

from __future__ import annotations

import re
from typing import Any

from prompts import (
    HUMAN_ESCALATION_MESSAGE,
    LOW_CONFIDENCE_MESSAGE,
    ORDER_ID_CLARIFICATION,
    RETURN_REASON_CLARIFICATION,
)
from rag import IndexNotBuiltError, RAGService
from tools import create_return_request, get_order_status, get_refund_status


MIN_RAG_SCORE = 0.05
_ORDER_ID_PATTERN = re.compile(r"\bORD-[0-9]{4,12}\b", re.IGNORECASE)
_RETURN_REASON_PATTERNS = (
    re.compile(r"\bbecause\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bdue\s+to\s+(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\breason\s*(?:is|:)\s*(.+?)(?:[.!?]|$)", re.IGNORECASE),
)

_GENERAL_RETURN_PATTERNS = (
    re.compile(r"\breturn(?:s)?\s+policy\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:do|can)\s+i\s+return\s+(?:a|an)\s+(?:product|item)\b", re.IGNORECASE),
    re.compile(
        r"^how\s+(?:do|can)\s+i\s+initiate\s+(?:a\s+)?return[?.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\breturn\s+window\b", re.IGNORECASE),
)
_GENERAL_REFUND_PATTERNS = (
    re.compile(r"\brefund(?:s)?\s+policy\b", re.IGNORECASE),
    re.compile(r"\brefund\s+timeline\b", re.IGNORECASE),
    re.compile(
        r"^when\s+will\s+i\s+receive\s+my\s+refund[?.!]?\s*$",
        re.IGNORECASE,
    ),
)
_GENERAL_ORDER_PATTERNS = (
    re.compile(r"^how\s+do\s+i\s+place\s+an?\s+order[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"^how\s+can\s+i\s+track\s+my\s+order[?.!]?\s*$", re.IGNORECASE),
    re.compile(
        r"^what\s+are\s+the\s+different\s+order\s+statuses[?.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^do\s+you\s+deliver\s+to\s+all\s+locations[?.!]?\s*$", re.IGNORECASE),
    re.compile(r"^what\s+if\s+my\s+order\s+is\s+delayed[?.!]?\s*$", re.IGNORECASE),
)
_GENERAL_PAYMENT_PATTERNS = (
    re.compile(r"^is\s+it\s+safe\s+to\s+pay\s+online[?.!]?\s*$", re.IGNORECASE),
)
_RETURN_ACTION_PATTERNS = (
    re.compile(r"\b(?:return|replace)\s+(?:my\s+)?(?:order|item|product)\b", re.IGNORECASE),
    re.compile(r"\b(?:start|create|initiate|request|open)\s+(?:a\s+)?return\b", re.IGNORECASE),
)
_ORDER_STATUS_PATTERNS = (
    re.compile(r"\bwhere\s+is\s+my\s+order\b", re.IGNORECASE),
    re.compile(r"\btrack\s+(?:my\s+)?order\b", re.IGNORECASE),
    re.compile(r"\border\s+status\b", re.IGNORECASE),
    re.compile(r"\bstatus\s+of\s+(?:my\s+)?order\b", re.IGNORECASE),
)
_RAG_KEYWORDS = {
    "cancel",
    "cancellation",
    "contact",
    "delivery",
    "faq",
    "payment",
    "policy",
    "privacy",
    "return",
    "refund",
    "shipping",
    "support",
    "warranty",
}
_PRODUCT_KEYWORDS = {
    "audio",
    "charger",
    "earbuds",
    "headphones",
    "hdmi",
    "monitor",
    "power bank",
    "price",
    "product",
    "specification",
    "specifications",
    "specs",
    "stock",
}


def _result(
    route: str,
    answer: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    tool_result: dict[str, Any] | None = None,
    escalation: bool = False,
) -> dict[str, Any]:
    return {
        "route": route,
        "answer": answer,
        "sources": sources or [],
        "tool_result": tool_result,
        "escalation": escalation,
    }


def _matches_any(query: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(query) for pattern in patterns)


def _extract_order_id(query: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(query)
    return match.group(0).upper() if match else None


def _extract_return_reason(query: str) -> str | None:
    for pattern in _RETURN_REASON_PATTERNS:
        match = pattern.search(query)
        if match:
            reason = match.group(1).strip()
            if reason:
                return reason
    return None


def _is_general_knowledge(query: str) -> bool:
    return any(
        _matches_any(query, patterns)
        for patterns in (
            _GENERAL_RETURN_PATTERNS,
            _GENERAL_REFUND_PATTERNS,
            _GENERAL_ORDER_PATTERNS,
            _GENERAL_PAYMENT_PATTERNS,
        )
    )


def _is_refund_request(query: str) -> bool:
    return "refund" in query.lower() and not _matches_any(
        query, _GENERAL_REFUND_PATTERNS
    )


def _is_return_request(query: str) -> bool:
    return _matches_any(query, _RETURN_ACTION_PATTERNS) and not _matches_any(
        query, _GENERAL_RETURN_PATTERNS
    )


def _is_order_status_request(query: str) -> bool:
    return _matches_any(query, _ORDER_STATUS_PATTERNS) and not _matches_any(
        query, _GENERAL_ORDER_PATTERNS
    )


def _is_rag_question(query: str) -> bool:
    lowered = query.lower()
    return _is_general_knowledge(query) or any(
        keyword in lowered for keyword in _RAG_KEYWORDS | _PRODUCT_KEYWORDS
    )


def _order_answer(tool_result: dict[str, Any]) -> str:
    if not tool_result.get("ok"):
        return str(tool_result["error"]["message"])

    order = tool_result["order"]
    status = str(order["status"]).replace("_", " ").title()
    answer = f"Order {order['order_id']} is currently {status}."
    tracking = order.get("tracking", {})
    if tracking.get("tracking_id"):
        answer += (
            f" Tracking ID: {tracking['tracking_id']} via {tracking['carrier']}."
        )
    if tracking.get("last_update"):
        answer += f" Latest update: {tracking['last_update']}."
    return answer


def _return_answer(tool_result: dict[str, Any]) -> str:
    if not tool_result.get("ok"):
        return str(tool_result["error"]["message"])

    eligibility = tool_result["eligibility"]
    decision = "eligible" if eligibility["eligible"] else "not eligible"
    return (
        f"Simulation only—nothing was submitted or saved. The recorded return "
        f"for {tool_result['order_id']} is {decision}. "
        f"Status: {eligibility['status']}. Reason: {eligibility['reason']}"
    )


def _refund_answer(tool_result: dict[str, Any]) -> str:
    if not tool_result.get("ok"):
        return str(tool_result["error"]["message"])
    if not tool_result["refund_found"]:
        return str(tool_result["message"])

    summaries = []
    for refund in tool_result["refunds"]:
        status = str(refund["status"]).replace("_", " ")
        date = refund["refund_date"] or "not completed yet"
        summaries.append(
            f"{refund['refund_id']}: INR {refund['amount_inr']}, {status}, "
            f"refund date {date}"
        )
    return f"Refund status for {tool_result['order_id']}: " + "; ".join(summaries) + "."


class CustomerSupportAgent:
    """Route support queries to local retrieval, mock tools, or escalation."""

    def __init__(
        self,
        rag_service: RAGService | None = None,
        minimum_rag_score: float = MIN_RAG_SCORE,
    ) -> None:
        self.rag_service = rag_service or RAGService()
        self.minimum_rag_score = minimum_rag_score

    def respond(self, query: str) -> dict[str, Any]:
        """Return a structured response without external calls or data mutation."""
        if not isinstance(query, str) or not query.strip():
            return _result(
                "clarification",
                "Please enter a support question so I can help.",
            )

        query = query.strip()
        order_id = _extract_order_id(query)

        if _is_refund_request(query):
            if not order_id:
                return _result("clarification", ORDER_ID_CLARIFICATION)
            tool_result = get_refund_status(order_id)
            return _result(
                "refund_status",
                _refund_answer(tool_result),
                tool_result=tool_result,
                escalation=tool_result.get("error", {}).get("code") == "data_unavailable",
            )

        if _is_return_request(query):
            if not order_id:
                return _result("clarification", ORDER_ID_CLARIFICATION)
            reason = _extract_return_reason(query)
            if not reason:
                return _result("clarification", RETURN_REASON_CLARIFICATION)
            tool_result = create_return_request(order_id, reason)
            escalation_codes = {"data_unavailable", "return_eligibility_unavailable"}
            return _result(
                "return_request",
                _return_answer(tool_result),
                tool_result=tool_result,
                escalation=tool_result.get("error", {}).get("code") in escalation_codes,
            )

        if _is_order_status_request(query):
            if not order_id:
                return _result("clarification", ORDER_ID_CLARIFICATION)
            tool_result = get_order_status(order_id)
            return _result(
                "order_status",
                _order_answer(tool_result),
                tool_result=tool_result,
                escalation=tool_result.get("error", {}).get("code") == "data_unavailable",
            )

        if _is_rag_question(query):
            try:
                matches = self.rag_service.search_knowledge(query, top_k=3)
            except (IndexNotBuiltError, OSError, ValueError):
                return _result(
                    "human_escalation", HUMAN_ESCALATION_MESSAGE, escalation=True
                )

            if not matches or matches[0]["score"] < self.minimum_rag_score:
                return _result(
                    "human_escalation", LOW_CONFIDENCE_MESSAGE, escalation=True
                )

            sources = [
                {
                    "source": match["source"],
                    "citation": match["citation"],
                    "score": match["score"],
                    "metadata": match["metadata"],
                }
                for match in matches
            ]
            return _result(
                "rag",
                "Here is the most relevant information from the local knowledge "
                f"base:\n\n{matches[0]['text']}",
                sources=sources,
            )

        return _result(
            "human_escalation", HUMAN_ESCALATION_MESSAGE, escalation=True
        )


def handle_support_query(
    query: str, rag_service: RAGService | None = None
) -> dict[str, Any]:
    """Convenience entry point for deterministic support routing."""
    return CustomerSupportAgent(rag_service=rag_service).respond(query)
