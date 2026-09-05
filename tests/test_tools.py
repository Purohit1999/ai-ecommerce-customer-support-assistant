"""Tests for the local, non-persistent mock support tools."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import pytest

from config import WORKING_DATA_DIR
from tools import create_return_request, get_order_status, get_refund_status


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_order_lookup() -> None:
    result = get_order_status(" ord-1006 ")

    assert result["ok"] is True
    assert result["order_id"] == "ORD-1006"
    assert result["status"] == "out_for_delivery"
    assert result["order"]["tracking"]["tracking_id"] == "BD-IN-81006"
    assert result["order"]["totals"]["total_inr"] == 4998


def test_unknown_order_returns_safe_error() -> None:
    result = get_order_status("ORD-9999")

    assert result["ok"] is False
    assert result["error"]["code"] == "order_not_found"
    assert result["order_id"] == "ORD-9999"


def test_valid_return_scenario_is_simulated_without_writing() -> None:
    returns_path = WORKING_DATA_DIR / "returns.json"
    before = _sha256(returns_path)

    result = create_return_request("ORD-1001", "Changed my mind")

    assert result["ok"] is True
    assert result["simulated"] is True
    assert result["persisted"] is False
    assert result["eligibility"]["eligible"] is True
    assert result["eligibility"]["status"] == "requested"
    assert result["existing_return"]["return_id"] == "RET-2001"
    assert _sha256(returns_path) == before


def test_ineligible_return_scenario() -> None:
    result = create_return_request("ORD-1008", "Changed my mind")

    assert result["ok"] is True
    assert result["simulated"] is True
    assert result["persisted"] is False
    assert result["eligibility"]["eligible"] is False
    assert result["eligibility"]["status"] == "rejected"
    assert "7-day" in result["eligibility"]["reason"]


def test_refund_found() -> None:
    result = get_refund_status("ORD-1007")

    assert result["ok"] is True
    assert result["refund_found"] is True
    assert result["count"] == 1
    assert result["refunds"][0]["refund_id"] == "REF-3003"
    assert result["refunds"][0]["status"] == "completed"


def test_no_refund_found_for_known_order() -> None:
    result = get_refund_status("ORD-1002")

    assert result["ok"] is True
    assert result["refund_found"] is False
    assert result["count"] == 0
    assert result["refunds"] == []


@pytest.mark.parametrize("invalid_id", ["", "   ", None, "../../orders.json"])
@pytest.mark.parametrize(
    "tool",
    [
        get_order_status,
        get_refund_status,
        lambda order_id: create_return_request(order_id, "Changed my mind"),
    ],
)
def test_invalid_or_empty_order_ids_return_safe_errors(
    tool: Callable[[object], dict[str, object]], invalid_id: object
) -> None:
    result = tool(invalid_id)

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_order_id"  # type: ignore[index]


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_invalid_return_reason_is_rejected(reason: object) -> None:
    result = create_return_request("ORD-1001", reason)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_reason"
    assert result["persisted"] is False
