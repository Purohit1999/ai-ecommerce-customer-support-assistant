"""Small non-browser tests for Streamlit integration helpers."""

from __future__ import annotations

import socket

import pytest

from app import initialize_agent, response_notice, safe_agent_response
from config import WORKING_DATA_DIR


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if UI initialization or agent handling opens a socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("External network access is forbidden in UI tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def test_initialize_agent_builds_temporary_index(tmp_path) -> None:
    store = tmp_path / "vector_store"
    agent = initialize_agent(store, WORKING_DATA_DIR)

    assert agent.rag_service.index_count() == 18
    result = safe_agent_response(agent, "What are the shipping charges?")
    assert result["route"] == "rag"
    assert result["sources"]


def test_initialize_agent_reuses_existing_index(tmp_path) -> None:
    store = tmp_path / "vector_store"
    first_agent = initialize_agent(store, WORKING_DATA_DIR)
    index_path = first_agent.rag_service.index_path
    initial_mtime = index_path.stat().st_mtime_ns

    second_agent = initialize_agent(store, WORKING_DATA_DIR)

    assert second_agent.rag_service.index_count() == 18
    assert index_path.stat().st_mtime_ns == initial_mtime


def test_safe_agent_response_hides_internal_error() -> None:
    class BrokenAgent:
        def respond(self, query: str):
            raise RuntimeError("sensitive internal detail")

    result = safe_agent_response(BrokenAgent(), "hello")  # type: ignore[arg-type]

    assert result["route"] == "human_escalation"
    assert result["escalation"] is True
    assert "sensitive internal detail" not in result["answer"]


def test_clarification_notice() -> None:
    notice = response_notice(
        {"route": "clarification", "escalation": False, "tool_result": None}
    )

    assert notice == ("warning", "Clarification needed")


def test_simulated_return_notice() -> None:
    notice = response_notice(
        {
            "route": "return_request",
            "escalation": False,
            "tool_result": {"simulated": True, "persisted": False},
        }
    )

    assert notice is not None
    assert notice[0] == "info"
    assert "not persisted" in notice[1]


def test_escalation_notice() -> None:
    notice = response_notice(
        {"route": "human_escalation", "escalation": True, "tool_result": None}
    )

    assert notice == ("error", "Human support escalation recommended")
