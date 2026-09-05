"""Optional grounded answer synthesis layered after deterministic routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Callable, Mapping, Protocol

from agent import CustomerSupportAgent
from prompts import GROUNDED_SYNTHESIS_PROMPT, SYSTEM_PROMPT


_TRUE_VALUES = {"1", "true", "yes", "on"}
_SYNTHESIS_ROUTES = {"rag", "order_status", "return_request", "refund_status"}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SAFE_CONNECTIVE_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "here",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "so",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


class LLMAdapter(Protocol):
    """Minimal provider boundary used by grounded synthesis."""

    def generate(self, *, instructions: str, input_text: str) -> str:
        """Generate text from an authoritative prompt."""


@dataclass(frozen=True)
class LLMSettings:
    """Environment-derived optional LLM settings."""

    enabled: bool = False
    provider: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)

    @property
    def configured(self) -> bool:
        return bool(
            self.enabled
            and self.provider == "openai"
            and self.model
            and self.api_key
        )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "LLMSettings":
        source = os.environ if environ is None else environ
        return cls(
            enabled=source.get("SUPPORT_LLM_ENABLED", "").strip().lower()
            in _TRUE_VALUES,
            provider=source.get("SUPPORT_LLM_PROVIDER", "").strip().lower(),
            model=source.get("SUPPORT_LLM_MODEL", "").strip(),
            api_key=source.get("OPENAI_API_KEY", "").strip(),
        )


class OpenAIResponsesAdapter:
    """Lazy OpenAI Responses API adapter.

    Constructing the adapter does not make a network request. A request occurs
    only if the environment is fully configured and a supported query reaches
    synthesis.
    """

    def __init__(self, *, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, *, instructions: str, input_text: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
        )
        return response.output_text


def build_grounded_input(query: str, result: Mapping[str, Any]) -> str:
    """Serialize only the existing query and authoritative deterministic result."""
    payload = {
        "customer_query": query,
        "route": result.get("route"),
        "authoritative_answer": result.get("answer"),
        "sources": result.get("sources", []),
        "tool_result": result.get("tool_result"),
    }
    return GROUNDED_SYNTHESIS_PROMPT.format(
        grounded_context=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _candidate_is_grounded(
    candidate: object,
    baseline_answer: str,
    route: str,
) -> bool:
    if not isinstance(candidate, str) or not candidate.strip():
        return False
    candidate = candidate.strip()
    if len(candidate) > max(600, len(baseline_answer) * 2):
        return False

    allowed_tokens = set(_TOKEN_PATTERN.findall(baseline_answer.casefold()))
    allowed_tokens.update(_SAFE_CONNECTIVE_WORDS)
    candidate_tokens = set(_TOKEN_PATTERN.findall(candidate.casefold()))
    if not candidate_tokens.issubset(allowed_tokens):
        return False

    if route == "return_request":
        lowered = candidate.casefold()
        preserves_simulation_warning = "simulation" in lowered and (
            "not persisted" in lowered
            or "nothing was submitted" in lowered
            or "nothing was saved" in lowered
        )
        if not preserves_simulation_warning:
            return False
    return True


class LLMEnhancedSupportAgent:
    """Optionally rewrite grounded answers after the base agent has finished."""

    def __init__(
        self,
        base_agent: CustomerSupportAgent,
        adapter: LLMAdapter,
        *,
        provider: str,
        model: str,
    ) -> None:
        self.base_agent = base_agent
        self.adapter = adapter
        self.provider = provider
        self.model = model
        self.rag_service = base_agent.rag_service

    def respond(self, query: str) -> dict[str, Any]:
        result = self.base_agent.respond(query)
        route = str(result.get("route", ""))
        tool_result = result.get("tool_result")
        if (
            route not in _SYNTHESIS_ROUTES
            or result.get("escalation")
            or (
                isinstance(tool_result, dict)
                and tool_result.get("ok") is not True
            )
        ):
            return result

        baseline_answer = str(result.get("answer", ""))
        try:
            candidate = self.adapter.generate(
                instructions=SYSTEM_PROMPT,
                input_text=build_grounded_input(query, result),
            )
        except Exception:
            return result

        if not _candidate_is_grounded(candidate, baseline_answer, route):
            return result

        enhanced = dict(result)
        enhanced["answer"] = candidate.strip()
        enhanced["generation"] = {
            "mode": "llm",
            "provider": self.provider,
            "model": self.model,
            "grounded": True,
        }
        return enhanced


AdapterFactory = Callable[[LLMSettings], LLMAdapter]


def _default_adapter_factory(settings: LLMSettings) -> LLMAdapter:
    return OpenAIResponsesAdapter(api_key=settings.api_key, model=settings.model)


def configure_optional_llm(
    base_agent: CustomerSupportAgent,
    *,
    environ: Mapping[str, str] | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> CustomerSupportAgent | LLMEnhancedSupportAgent:
    """Return the base agent unless environment configuration fully opts in."""
    settings = LLMSettings.from_environment(environ)
    if not settings.configured:
        return base_agent

    factory = adapter_factory or _default_adapter_factory
    try:
        adapter = factory(settings)
    except Exception:
        return base_agent
    return LLMEnhancedSupportAgent(
        base_agent,
        adapter,
        provider=settings.provider,
        model=settings.model,
    )
