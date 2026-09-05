"""Tests for the deterministic offline evaluation harness."""

from __future__ import annotations

from pathlib import Path
import socket

import pytest

from evaluate import (
    calculate_metrics,
    evaluate_cases,
    load_evaluation_cases,
    render_report,
    run_evaluation,
)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if the evaluation attempts to access a network socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("External network access is forbidden in evaluation tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def test_evaluation_dataset_is_complete_and_unique() -> None:
    cases = load_evaluation_cases()

    assert 20 <= len(cases) <= 25
    assert len({case.case_id for case in cases}) == len(cases)
    assert {
        "rag",
        "order_status",
        "return_request",
        "refund_status",
        "clarification",
        "human_escalation",
    }.issubset({case.expected_route for case in cases})


def test_evaluation_metrics_are_deterministic(tmp_path: Path) -> None:
    cases = load_evaluation_cases()
    results = evaluate_cases(cases, tmp_path / "vector_store")
    metrics = calculate_metrics(results)

    assert [result.case_id for result in results if not result.passed] == ["Q010"]
    assert metrics["Routing accuracy"]["accuracy"] == 1.0
    assert metrics["Tool correctness"]["accuracy"] == 1.0
    assert metrics["RAG source/grounding correctness"] == {
        "passed": 8,
        "total": 9,
        "accuracy": 8 / 9,
    }
    assert metrics["Overall pass rate"]["accuracy"] == 24 / 25


def test_report_contains_requested_metrics(tmp_path: Path) -> None:
    cases = load_evaluation_cases()
    results = evaluate_cases(cases, tmp_path / "vector_store")
    metrics = calculate_metrics(results)
    report = render_report(cases, results, metrics)

    assert "Routing accuracy" in report
    assert "Tool correctness" in report
    assert "RAG source/grounding correctness" in report
    assert "Clarification correctness" in report
    assert "Escalation correctness" in report
    assert "Overall pass rate" in report


def test_run_evaluation_writes_only_requested_report(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "evaluation.md"
    results, metrics = run_evaluation(report_path=report_path)

    assert report_path.is_file()
    assert len(results) == 25
    assert metrics["Overall pass rate"]["accuracy"] == 24 / 25
    assert "Q010" in report_path.read_text(encoding="utf-8")
