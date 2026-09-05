"""Tests for optional grounded LLM synthesis without network access."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from llm import (
    LLMEnhancedSupportAgent,
    LLMSettings,
    build_grounded_input,
    configure_optional_llm,
)


class StubBaseAgent:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.rag_service = object()

    def respond(self, query: str) -> dict[str, Any]:
        return self.result


class CapturingAdapter:
    def __init__(self, answer: str, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[dict[str, str]] = []

    def generate(self, *, instructions: str, input_text: str) -> str:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if any optional-LLM test opens a socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is forbidden in optional LLM tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _order_result() -> dict[str, Any]:
    return {
        "route": "order_status",
        "answer": (
            "Order ORD-1006 is currently Out For Delivery. "
            "Tracking ID: BD-IN-81006 via BlueDart."
        ),
        "sources": [],
        "tool_result": {
            "ok": True,
            "order_id": "ORD-1006",
            "status": "out_for_delivery",
        },
        "escalation": False,
    }


def test_unconfigured_environment_returns_original_offline_agent() -> None:
    base = StubBaseAgent(_order_result())

    configured = configure_optional_llm(base, environ={})  # type: ignore[arg-type]

    assert configured is base
    assert configured.respond("Where is my order ORD-1006?") == base.result


@pytest.mark.parametrize(
    "environ",
    [
        {"SUPPORT_LLM_ENABLED": "false"},
        {
            "SUPPORT_LLM_ENABLED": "true",
            "SUPPORT_LLM_PROVIDER": "openai",
            "SUPPORT_LLM_MODEL": "",
            "OPENAI_API_KEY": "test-key",
        },
        {
            "SUPPORT_LLM_ENABLED": "true",
            "SUPPORT_LLM_PROVIDER": "openai",
            "SUPPORT_LLM_MODEL": "test-model",
            "OPENAI_API_KEY": "",
        },
    ],
)
def test_incomplete_or_disabled_configuration_stays_offline(
    environ: dict[str, str],
) -> None:
    base = StubBaseAgent(_order_result())
    called = False

    def factory(settings: LLMSettings) -> CapturingAdapter:
        nonlocal called
        called = True
        return CapturingAdapter(base.result["answer"])

    configured = configure_optional_llm(  # type: ignore[arg-type]
        base, environ=environ, adapter_factory=factory
    )

    assert configured is base
    assert called is False


def test_complete_environment_configuration_enables_injected_adapter() -> None:
    base = StubBaseAgent(_order_result())
    adapter = CapturingAdapter(base.result["answer"])
    captured_settings: list[LLMSettings] = []

    def factory(settings: LLMSettings) -> CapturingAdapter:
        captured_settings.append(settings)
        return adapter

    configured = configure_optional_llm(  # type: ignore[arg-type]
        base,
        environ={
            "SUPPORT_LLM_ENABLED": "true",
            "SUPPORT_LLM_PROVIDER": "openai",
            "SUPPORT_LLM_MODEL": "test-model",
            "OPENAI_API_KEY": "test-key",
        },
        adapter_factory=factory,
    )

    assert isinstance(configured, LLMEnhancedSupportAgent)
    assert captured_settings[0].model == "test-model"
    assert "test-key" not in repr(captured_settings[0])


def test_grounded_tool_context_is_passed_to_adapter() -> None:
    base_result = _order_result()
    base = StubBaseAgent(base_result)
    adapter = CapturingAdapter(
        "Your order ORD-1006 is currently Out For Delivery. "
        "Tracking ID BD-IN-81006 via BlueDart."
    )
    enhanced = LLMEnhancedSupportAgent(  # type: ignore[arg-type]
        base, adapter, provider="test", model="test-model"
    )

    result = enhanced.respond("Where is my order ORD-1006?")

    assert len(adapter.calls) == 1
    prompt = adapter.calls[0]["input_text"]
    assert "ORD-1006" in prompt
    assert "out_for_delivery" in prompt
    assert "BD-IN-81006" in prompt
    assert base_result["answer"] in prompt
    assert result["generation"]["grounded"] is True


def test_rag_sources_are_in_grounded_input() -> None:
    result = {
        "route": "rag",
        "answer": "Standard delivery takes 3–6 business days.",
        "sources": [
            {
                "source": "faq.md",
                "citation": "faq.md#2-shipping-delivery",
                "score": 0.4,
                "metadata": {"section": "Shipping & Delivery"},
            }
        ],
        "tool_result": None,
        "escalation": False,
    }

    prompt = build_grounded_input("How long is delivery?", result)

    assert "Standard delivery takes 3–6 business days." in prompt
    assert "faq.md#2-shipping-delivery" in prompt
    assert "How long is delivery?" in prompt


def test_ungrounded_candidate_falls_back_to_offline_answer() -> None:
    base_result = _order_result()
    base = StubBaseAgent(base_result)
    adapter = CapturingAdapter(
        "Order ORD-1006 will definitely arrive tomorrow with a free voucher."
    )
    enhanced = LLMEnhancedSupportAgent(  # type: ignore[arg-type]
        base, adapter, provider="test", model="test-model"
    )

    result = enhanced.respond("Where is my order ORD-1006?")

    assert result == base_result


def test_adapter_error_falls_back_to_offline_answer() -> None:
    base_result = _order_result()
    base = StubBaseAgent(base_result)
    adapter = CapturingAdapter("", error=RuntimeError("provider unavailable"))
    enhanced = LLMEnhancedSupportAgent(  # type: ignore[arg-type]
        base, adapter, provider="test", model="test-model"
    )

    result = enhanced.respond("Where is my order ORD-1006?")

    assert result == base_result


def test_escalation_is_never_sent_to_llm() -> None:
    base_result = {
        "route": "human_escalation",
        "answer": "Please contact human support.",
        "sources": [],
        "tool_result": None,
        "escalation": True,
    }
    adapter = CapturingAdapter("Anything")
    enhanced = LLMEnhancedSupportAgent(  # type: ignore[arg-type]
        StubBaseAgent(base_result), adapter, provider="test", model="test-model"
    )

    result = enhanced.respond("Unsupported request")

    assert result == base_result
    assert adapter.calls == []
