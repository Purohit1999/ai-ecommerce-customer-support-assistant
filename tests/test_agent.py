"""Tests for deterministic agent routing and safe fallbacks."""

from __future__ import annotations

from pathlib import Path
import socket
from typing import Any

import pytest

from agent import CustomerSupportAgent
from config import WORKING_DATA_DIR
from rag import RAGService


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if routing or a dependency attempts socket access."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("External network access is forbidden in agent tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def support_agent(tmp_path: Path) -> CustomerSupportAgent:
    rag_service = RAGService(
        persist_directory=tmp_path / "vector_store",
        data_directory=WORKING_DATA_DIR,
        include_products=True,
    )
    rag_service.build_index()
    return CustomerSupportAgent(rag_service=rag_service)


def test_faq_routes_to_rag(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond("What are the standard shipping charges?")

    assert result["route"] == "rag"
    assert result["escalation"] is False
    assert result["sources"]
    assert result["sources"][0]["source"] in {"faq.md", "policies.md"}
    assert result["tool_result"] is None


def test_product_question_routes_to_rag(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond("Does the 24-inch monitor support HDMI?")

    assert result["route"] == "rag"
    assert result["sources"][0]["source"] == "products.json"
    assert result["sources"][0]["metadata"]["product_id"] == "P1006"


def test_order_status_routes_to_tool(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond("Where is my order ORD-1006?")

    assert result["route"] == "order_status"
    assert result["tool_result"]["status"] == "out_for_delivery"
    assert "BD-IN-81006" in result["answer"]
    assert result["sources"] == []


def test_return_request_routes_to_tool(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond(
        "Please return my order ORD-1001 because I changed my mind."
    )

    assert result["route"] == "return_request"
    assert result["tool_result"]["simulated"] is True
    assert result["tool_result"]["persisted"] is False
    assert result["tool_result"]["eligibility"]["eligible"] is True
    assert "nothing was submitted or saved" in result["answer"]


def test_refund_request_routes_to_tool(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond("Where is my refund for order ORD-1007?")

    assert result["route"] == "refund_status"
    assert result["tool_result"]["refund_found"] is True
    assert result["tool_result"]["refunds"][0]["refund_id"] == "REF-3003"


@pytest.mark.parametrize(
    "query",
    [
        "Where is my order?",
        "I want to return my item because it is damaged.",
        "Where is my refund?",
    ],
)
def test_missing_order_id_requests_clarification(
    support_agent: CustomerSupportAgent, query: str
) -> None:
    result = support_agent.respond(query)

    assert result["route"] == "clarification"
    assert "order ID" in result["answer"]
    assert result["escalation"] is False
    assert result["tool_result"] is None


def test_return_without_reason_requests_clarification(
    support_agent: CustomerSupportAgent,
) -> None:
    result = support_agent.respond("Return my order ORD-1001")

    assert result["route"] == "clarification"
    assert "reason" in result["answer"].lower()


def test_unsupported_request_escalates(support_agent: CustomerSupportAgent) -> None:
    result = support_agent.respond("Please write a poem about the moon.")

    assert result["route"] == "human_escalation"
    assert result["escalation"] is True
    assert "human support" in result["answer"]


def test_low_confidence_rag_result_escalates() -> None:
    class LowConfidenceRAG:
        def search_knowledge(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
            return [
                {
                    "text": "Unrelated text",
                    "source": "faq.md",
                    "citation": "faq.md#unrelated",
                    "score": 0.001,
                    "metadata": {"source_type": "faq"},
                }
            ]

    agent = CustomerSupportAgent(rag_service=LowConfidenceRAG())  # type: ignore[arg-type]
    result = agent.respond("What is your warranty policy?")

    assert result["route"] == "human_escalation"
    assert result["escalation"] is True
    assert result["sources"] == []
