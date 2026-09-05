"""Deterministic, read-only mock commerce tools backed by local project data."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from config import WORKING_DATA_DIR


DATA_DIRECTORY = WORKING_DATA_DIR
_ORDER_ID_PATTERN = re.compile(r"^ORD-[0-9]{4,12}$")
_MAX_REASON_LENGTH = 500


class _LocalDataError(RuntimeError):
    """Internal error raised when a local mock-data file cannot be used."""


def _error(code: str, message: str, order_id: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "order_id": order_id,
        "error": {"code": code, "message": message},
    }


def _normalize_order_id(order_id: object) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(order_id, str) or not order_id.strip():
        return None, _error(
            "invalid_order_id", "order_id must be a non-empty string."
        )

    normalized = order_id.strip().upper()
    if not _ORDER_ID_PATTERN.fullmatch(normalized):
        return None, _error(
            "invalid_order_id",
            "order_id must use the format ORD- followed by 4 to 12 digits.",
        )
    return normalized, None


def _normalize_reason(reason: object) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(reason, str) or not reason.strip():
        return None, _error(
            "invalid_reason", "reason must be a non-empty string."
        )
    normalized = reason.strip()
    if len(normalized) > _MAX_REASON_LENGTH:
        return None, _error(
            "invalid_reason",
            f"reason must not exceed {_MAX_REASON_LENGTH} characters.",
        )
    return normalized, None


def _load_records(filename: str) -> list[dict[str, Any]]:
    path = Path(DATA_DIRECTORY) / filename
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _LocalDataError(f"Local data file {filename} is unavailable or invalid.") from exc

    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise _LocalDataError(f"Local data file {filename} has an invalid structure.")
    return records


def _data_error(exc: _LocalDataError, order_id: str | None) -> dict[str, Any]:
    return _error("data_unavailable", str(exc), order_id)


def _find_order(
    orders: list[dict[str, Any]], order_id: str
) -> dict[str, Any] | None:
    return next((order for order in orders if order.get("order_id") == order_id), None)


def get_order_status(order_id: str) -> dict[str, Any]:
    """Return a local order record, or a safe structured error."""
    normalized_id, validation_error = _normalize_order_id(order_id)
    if validation_error:
        return validation_error

    try:
        order = _find_order(_load_records("orders.json"), normalized_id)
    except _LocalDataError as exc:
        return _data_error(exc, normalized_id)

    if order is None:
        return _error(
            "order_not_found",
            f"No order was found for order ID {normalized_id}.",
            normalized_id,
        )
    return {
        "ok": True,
        "order_id": normalized_id,
        "status": order["status"],
        "order": order,
    }


def create_return_request(order_id: str, reason: str) -> dict[str, Any]:
    """Simulate a return request using only recorded eligibility information.

    This function never writes to ``returns.json`` or any other file.
    """
    normalized_id, validation_error = _normalize_order_id(order_id)
    if validation_error:
        return {**validation_error, "simulated": True, "persisted": False}
    normalized_reason, reason_error = _normalize_reason(reason)
    if reason_error:
        return {
            **reason_error,
            "order_id": normalized_id,
            "simulated": True,
            "persisted": False,
        }

    try:
        orders = _load_records("orders.json")
        returns = _load_records("returns.json")
    except _LocalDataError as exc:
        return {
            **_data_error(exc, normalized_id),
            "simulated": True,
            "persisted": False,
        }

    order = _find_order(orders, normalized_id)
    if order is None:
        return {
            **_error(
                "order_not_found",
                f"No order was found for order ID {normalized_id}.",
                normalized_id,
            ),
            "simulated": True,
            "persisted": False,
        }

    matching_returns = [
        record for record in returns if record.get("order_id") == normalized_id
    ]
    if not matching_returns:
        return {
            **_error(
                "return_eligibility_unavailable",
                "No existing return eligibility record is available for this order.",
                normalized_id,
            ),
            "simulated": True,
            "persisted": False,
            "requested_reason": normalized_reason,
            "eligibility": {
                "eligible": None,
                "status": "not_evaluated",
                "reason": "No recorded eligibility decision is available.",
            },
        }

    recorded_return = max(
        matching_returns, key=lambda record: str(record.get("requested_date", ""))
    )
    eligible = bool(recorded_return.get("eligible"))
    return {
        "ok": True,
        "order_id": normalized_id,
        "simulated": True,
        "persisted": False,
        "requested_reason": normalized_reason,
        "eligibility": {
            "eligible": eligible,
            "status": recorded_return.get("status"),
            "reason": recorded_return.get("eligibility_reason"),
        },
        "existing_return": recorded_return,
        "message": (
            "Simulation only: the recorded return is eligible."
            if eligible
            else "Simulation only: the recorded return is not eligible."
        ),
    }


def get_refund_status(order_id: str) -> dict[str, Any]:
    """Return all recorded refunds for a known order without changing data."""
    normalized_id, validation_error = _normalize_order_id(order_id)
    if validation_error:
        return validation_error

    try:
        orders = _load_records("orders.json")
        refunds = _load_records("refunds.json")
    except _LocalDataError as exc:
        return _data_error(exc, normalized_id)

    if _find_order(orders, normalized_id) is None:
        return _error(
            "order_not_found",
            f"No order was found for order ID {normalized_id}.",
            normalized_id,
        )

    matching_refunds = [
        refund for refund in refunds if refund.get("order_id") == normalized_id
    ]
    return {
        "ok": True,
        "order_id": normalized_id,
        "refund_found": bool(matching_refunds),
        "count": len(matching_refunds),
        "refunds": matching_refunds,
        "message": (
            "Refund record found."
            if matching_refunds
            else "No refund has been recorded for this order."
        ),
    }
