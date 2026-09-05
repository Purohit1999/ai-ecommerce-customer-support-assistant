"""Targeted tests for offline RAG ingestion and retrieval."""

from __future__ import annotations

from pathlib import Path
import socket

import pytest

from config import WORKING_DATA_DIR
from rag import RAGService


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every test if the implementation attempts to open a socket."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("External network access is forbidden in RAG tests")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def indexed_service(tmp_path: Path) -> RAGService:
    service = RAGService(
        persist_directory=tmp_path / "vector_store",
        data_directory=WORKING_DATA_DIR,
        include_products=True,
    )
    count = service.build_index()
    assert count > 0
    assert service.index_count() == count
    return service


def test_indexing_succeeds_and_is_idempotent(tmp_path: Path) -> None:
    service = RAGService(tmp_path / "vector_store", WORKING_DATA_DIR)
    first_count = service.build_index()
    second_count = service.build_index()

    assert first_count >= 18
    assert second_count == first_count
    assert service.index_count() == first_count


def test_faq_retrieval(indexed_service: RAGService) -> None:
    results = indexed_service.search_knowledge("How can I track my order?", top_k=3)

    assert results[0]["source"] == "faq.md"
    assert "track" in results[0]["text"].lower()
    assert results[0]["citation"].startswith("faq.md#")
    assert results[0]["metadata"]["source_type"] == "faq"


def test_policy_retrieval(indexed_service: RAGService) -> None:
    results = indexed_service.search_knowledge(
        "What privacy policy applies when sharing data with logistics partners?",
        top_k=3,
    )

    assert results[0]["source"] == "policies.md"
    assert "privacy" in results[0]["text"].lower()
    assert results[0]["metadata"]["source_type"] == "policy"


def test_product_retrieval(indexed_service: RAGService) -> None:
    results = indexed_service.search_knowledge(
        "Does the 24-inch monitor support HDMI?", top_k=3
    )

    assert results[0]["source"] == "products.json"
    assert results[0]["metadata"]["product_id"] == "P1006"
    assert "HDMI" in results[0]["text"]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_or_invalid_query_is_rejected(
    indexed_service: RAGService, query: object
) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        indexed_service.search_knowledge(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected(
    indexed_service: RAGService, top_k: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        indexed_service.search_knowledge("shipping", top_k=top_k)  # type: ignore[arg-type]
