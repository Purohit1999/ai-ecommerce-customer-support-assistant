"""Deterministic offline evaluation for the customer-support assistant."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent import CustomerSupportAgent
from config import PROJECT_ROOT, WORKING_DATA_DIR
from rag import RAGService


DEFAULT_CASES_PATH = WORKING_DATA_DIR / "evaluation_queries.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "report" / "evaluation_report.md"
TOOL_ROUTES = {"order_status", "return_request", "refund_status"}


@dataclass(frozen=True)
class EvaluationCase:
    """One query and its deterministic expected outcome."""

    case_id: str
    query: str
    expected_route: str
    expected_escalation: bool
    expected_sources: tuple[str, ...]
    expected_tool_ok: bool | None
    expected_tool_field: str | None
    expected_tool_value: Any
    expected_answer_contains: str | None


@dataclass(frozen=True)
class CaseResult:
    """Component scores and diagnostics for one evaluated query."""

    case_id: str
    query: str
    actual_route: str
    routing_correct: bool
    tool_correct: bool | None
    rag_correct: bool | None
    clarification_correct: bool | None
    escalation_correct: bool
    passed: bool
    failures: tuple[str, ...]


def _parse_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field_name} must be true or false, got {value!r}")


def _parse_expected_value(value: str) -> Any:
    normalized = value.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(normalized)
    except ValueError:
        return normalized


def load_evaluation_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    """Load and validate deterministic expectations from CSV."""
    required_columns = {
        "id",
        "query",
        "expected_route",
        "expected_escalation",
        "expected_source",
        "expected_tool_ok",
        "expected_tool_field",
        "expected_tool_value",
        "expected_answer_contains",
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")
        rows = list(reader)

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for row in rows:
        case_id = row["id"].strip()
        query = row["query"].strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Evaluation case ID is missing or duplicated: {case_id!r}")
        if not query:
            raise ValueError(f"Evaluation query is empty for {case_id}")
        seen_ids.add(case_id)

        tool_ok_text = row["expected_tool_ok"].strip()
        expected_tool_ok = (
            _parse_bool(tool_ok_text, "expected_tool_ok") if tool_ok_text else None
        )
        source_text = row["expected_source"].strip()
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=query,
                expected_route=row["expected_route"].strip(),
                expected_escalation=_parse_bool(
                    row["expected_escalation"], "expected_escalation"
                ),
                expected_sources=tuple(
                    source.strip()
                    for source in source_text.split("|")
                    if source.strip()
                ),
                expected_tool_ok=expected_tool_ok,
                expected_tool_field=row["expected_tool_field"].strip() or None,
                expected_tool_value=_parse_expected_value(
                    row["expected_tool_value"]
                ),
                expected_answer_contains=(
                    row["expected_answer_contains"].strip() or None
                ),
            )
        )
    return cases


def _nested_value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def evaluate_case(
    case: EvaluationCase,
    agent: CustomerSupportAgent,
    rag_service: RAGService,
) -> CaseResult:
    """Evaluate one case without a model-based judge."""
    response = agent.respond(case.query)
    failures: list[str] = []

    actual_route = str(response.get("route", ""))
    routing_correct = actual_route == case.expected_route
    if not routing_correct:
        failures.append(
            f"route expected {case.expected_route}, received {actual_route or 'missing'}"
        )

    escalation_correct = (
        response.get("escalation") is case.expected_escalation
    )
    if not escalation_correct:
        failures.append(
            f"escalation expected {case.expected_escalation}, "
            f"received {response.get('escalation')!r}"
        )

    answer = str(response.get("answer", ""))
    if (
        case.expected_answer_contains
        and case.expected_answer_contains.lower() not in answer.lower()
    ):
        failures.append(
            f"answer missing {case.expected_answer_contains!r}"
        )

    tool_correct: bool | None = None
    if case.expected_tool_ok is not None:
        tool_result = response.get("tool_result")
        tool_correct = (
            actual_route in TOOL_ROUTES
            and isinstance(tool_result, dict)
            and tool_result.get("ok") is case.expected_tool_ok
        )
        if tool_correct and case.expected_tool_field:
            actual_value = _nested_value(tool_result, case.expected_tool_field)
            tool_correct = actual_value == case.expected_tool_value
        if not tool_correct:
            failures.append("tool result did not match the expected deterministic value")

    rag_correct: bool | None = None
    if case.expected_route == "rag":
        sources = response.get("sources")
        source_correct = (
            isinstance(sources, list)
            and bool(sources)
            and sources[0].get("source") in case.expected_sources
        )
        retrieved = rag_service.search_knowledge(case.query, top_k=3)
        grounded = bool(retrieved) and retrieved[0]["text"] in answer
        expected_fact_present = (
            not case.expected_answer_contains
            or case.expected_answer_contains.lower() in answer.lower()
        )
        rag_correct = bool(source_correct and grounded and expected_fact_present)
        if not rag_correct:
            failures.append("RAG source or grounding did not match expectations")

    clarification_correct: bool | None = None
    if case.expected_route == "clarification":
        clarification_correct = (
            actual_route == "clarification"
            and not bool(response.get("escalation"))
            and bool(case.expected_answer_contains)
            and case.expected_answer_contains.lower() in answer.lower()
        )
        if not clarification_correct:
            failures.append("clarification response did not request the expected detail")

    return CaseResult(
        case_id=case.case_id,
        query=case.query,
        actual_route=actual_route,
        routing_correct=routing_correct,
        tool_correct=tool_correct,
        rag_correct=rag_correct,
        clarification_correct=clarification_correct,
        escalation_correct=escalation_correct,
        passed=not failures,
        failures=tuple(failures),
    )


def evaluate_cases(
    cases: list[EvaluationCase], index_directory: Path
) -> list[CaseResult]:
    """Build a temporary index and evaluate all cases locally."""
    rag_service = RAGService(
        persist_directory=index_directory,
        data_directory=WORKING_DATA_DIR,
        include_products=True,
    )
    rag_service.build_index()
    agent = CustomerSupportAgent(rag_service=rag_service)
    return [evaluate_case(case, agent, rag_service) for case in cases]


def _metric(
    results: list[CaseResult], attribute: str
) -> dict[str, int | float]:
    applicable = [
        getattr(result, attribute)
        for result in results
        if getattr(result, attribute) is not None
    ]
    passed = sum(value is True for value in applicable)
    total = len(applicable)
    return {
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total else 0.0,
    }


def calculate_metrics(
    results: list[CaseResult],
) -> dict[str, dict[str, int | float]]:
    """Calculate requested component and overall metrics."""
    metrics = {
        "Routing accuracy": _metric(results, "routing_correct"),
        "Tool correctness": _metric(results, "tool_correct"),
        "RAG source/grounding correctness": _metric(results, "rag_correct"),
        "Clarification correctness": _metric(results, "clarification_correct"),
        "Escalation correctness": _metric(results, "escalation_correct"),
    }
    overall_passed = sum(result.passed for result in results)
    metrics["Overall pass rate"] = {
        "passed": overall_passed,
        "total": len(results),
        "accuracy": overall_passed / len(results) if results else 0.0,
    }
    return metrics


def render_report(
    cases: list[EvaluationCase],
    results: list[CaseResult],
    metrics: dict[str, dict[str, int | float]],
) -> str:
    """Render a concise deterministic Markdown report."""
    lines = [
        "# Offline Evaluation Report",
        "",
        f"Cases evaluated: **{len(cases)}**",
        "",
        "| Metric | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['passed']} | {values['total']} | "
            f"{float(values['accuracy']):.1%} |"
        )

    failures = [result for result in results if not result.passed]
    lines.extend(["", "## Failed cases", ""])
    if not failures:
        lines.append("None.")
    else:
        for result in failures:
            lines.append(
                f"- **{result.case_id}** ({result.actual_route}): "
                + "; ".join(result.failures)
            )

    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "- Deterministic expectations only; no LLM judge or network access.",
            "- RAG correctness requires the expected source and fact, with the answer containing the top retrieved chunk.",
            "- Tool checks compare explicit nested fields against local JSON data.",
            "- The low-confidence case is synthetic and tied to the current hashing embeddings.",
            "- This small curated set measures regression behavior, not production quality.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    cases_path: Path = DEFAULT_CASES_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> tuple[list[CaseResult], dict[str, dict[str, int | float]]]:
    """Run the offline benchmark with temporary storage and save its report."""
    cases = load_evaluation_cases(cases_path)
    with TemporaryDirectory(prefix="support-eval-") as temporary_directory:
        results = evaluate_cases(cases, Path(temporary_directory) / "vector_store")
    metrics = calculate_metrics(results)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(cases, results, metrics),
        encoding="utf-8",
    )
    return results, metrics


def main() -> None:
    """Run the benchmark and print the same concise summary saved to disk."""
    cases = load_evaluation_cases()
    results, metrics = run_evaluation()
    print(render_report(cases, results, metrics))
    print(f"Report saved to: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
