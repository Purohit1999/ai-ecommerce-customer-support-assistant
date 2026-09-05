"""Offline document ingestion and vector retrieval for support knowledge.

The implementation uses deterministic feature-hashing embeddings and SQLite from
the Python standard library. It never downloads a model or contacts a service.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any

from config import CHROMA_STORE_DIR, WORKING_DATA_DIR


EMBEDDING_DIMENSIONS = 384
INDEX_FILENAME = "knowledge.sqlite3"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class IndexNotBuiltError(RuntimeError):
    """Raised when retrieval is attempted before an index is available."""


@dataclass(frozen=True)
class KnowledgeChunk:
    """A text chunk and the citation metadata stored alongside it."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _embedding_features(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def create_embedding(text: str) -> list[float]:
    """Create a deterministic, normalized hashing embedding without I/O."""
    counts = Counter(_embedding_features(text))
    vector = [0.0] * EMBEDDING_DIMENSIONS

    for feature, count in counts.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[position] += sign * (1.0 + math.log(count))

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude:
        vector = [value / magnitude for value in vector]
    return vector


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _split_long_text(
    text: str, max_chars: int = 1_200, overlap_chars: int = 120
) -> list[str]:
    """Split oversized sections near paragraph/word boundaries with overlap."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + max_chars, len(text))
        end = proposed_end
        if proposed_end < len(text):
            minimum_break = start + max_chars // 2
            candidates = (
                text.rfind("\n\n", minimum_break, proposed_end),
                text.rfind("\n", minimum_break, proposed_end),
                text.rfind(" ", minimum_break, proposed_end),
            )
            end = next((candidate for candidate in candidates if candidate > start), proposed_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - overlap_chars, start + 1)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def _markdown_chunks(path: Path, source_type: str) -> list[KnowledgeChunk]:
    content = path.read_text(encoding="utf-8")
    headings = list(_SECTION_PATTERN.finditer(content))
    chunks: list[KnowledgeChunk] = []

    for section_number, heading in enumerate(headings, start=1):
        section_end = (
            headings[section_number].start()
            if section_number < len(headings)
            else len(content)
        )
        title = heading.group(1).strip()
        section_text = content[heading.start() : section_end].strip()
        parts = _split_long_text(section_text)

        for part_number, part in enumerate(parts, start=1):
            citation = f"{path.name}#{_slug(title)}"
            if len(parts) > 1:
                citation = f"{citation}-part-{part_number}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{source_type}-{section_number:02d}-{part_number:02d}",
                    text=part,
                    metadata={
                        "source": path.name,
                        "source_type": source_type,
                        "section": title,
                        "section_index": section_number,
                        "part_index": part_number,
                        "citation": citation,
                    },
                )
            )
    return chunks


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _product_chunks(path: Path) -> list[KnowledgeChunk]:
    products = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(products, list):
        raise ValueError(f"Expected a JSON array in {path}")

    chunks: list[KnowledgeChunk] = []
    for index, product in enumerate(products, start=1):
        specs = product.get("specs", {})
        spec_lines = "\n".join(
            f"- {key.replace('_', ' ')}: {_format_value(value)}"
            for key, value in specs.items()
        )
        text = (
            f"Product: {product['name']}\n"
            f"Product ID: {product['id']}\n"
            f"Category: {product['category']}\n"
            f"Price: {product['currency']} {product['price_inr']}\n"
            f"Description: {product['description']}\n"
            f"Specifications:\n{spec_lines}\n"
            f"Stock status: {product['stock_status']}"
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"product-{product['id'].lower()}",
                text=text,
                metadata={
                    "source": path.name,
                    "source_type": "product",
                    "section": product["name"],
                    "section_index": index,
                    "part_index": 1,
                    "product_id": product["id"],
                    "citation": f"{path.name}#{product['id']}",
                },
            )
        )
    return chunks


def load_knowledge_chunks(
    data_directory: Path = WORKING_DATA_DIR, include_products: bool = True
) -> list[KnowledgeChunk]:
    """Load and chunk approved project copies from ``data_directory``."""
    data_directory = Path(data_directory)
    required = (data_directory / "faq.md", data_directory / "policies.md")
    missing = [str(path) for path in required if not path.is_file()]
    product_path = data_directory / "products.json"
    if include_products and not product_path.is_file():
        missing.append(str(product_path))
    if missing:
        raise FileNotFoundError(f"Missing knowledge source(s): {', '.join(missing)}")

    chunks = _markdown_chunks(required[0], "faq")
    chunks.extend(_markdown_chunks(required[1], "policy"))
    if include_products:
        chunks.extend(_product_chunks(product_path))
    return chunks


class RAGService:
    """Persist and search offline knowledge embeddings in a local vector store."""

    def __init__(
        self,
        persist_directory: Path = CHROMA_STORE_DIR,
        data_directory: Path = WORKING_DATA_DIR,
        include_products: bool = True,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.data_directory = Path(data_directory)
        self.include_products = include_products
        self.index_path = self.persist_directory / INDEX_FILENAME

    def build_index(self) -> int:
        """Create or replace this service's document index and return its size."""
        chunks = load_knowledge_chunks(self.data_directory, self.include_products)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.index_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        chunk_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute("DELETE FROM documents")
                connection.executemany(
                    """
                    INSERT INTO documents
                        (chunk_id, text, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            chunk.chunk_id,
                            chunk.text,
                            json.dumps(
                                chunk.metadata, ensure_ascii=False, sort_keys=True
                            ),
                            json.dumps(create_embedding(chunk.text)),
                        )
                        for chunk in chunks
                    ),
                )
        return len(chunks)

    def index_count(self) -> int:
        """Return the persisted chunk count."""
        if not self.index_path.is_file():
            return 0
        with closing(sqlite3.connect(self.index_path)) as connection:
            row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        return int(row[0]) if row else 0

    def search_knowledge(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return matching text with scores and source/citation metadata."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not self.index_path.is_file():
            raise IndexNotBuiltError(
                f"Knowledge index not found at {self.index_path}; call build_index() first."
            )

        with closing(sqlite3.connect(self.index_path)) as connection:
            rows = connection.execute(
                "SELECT chunk_id, text, metadata_json, embedding_json FROM documents"
            ).fetchall()
        if not rows:
            raise IndexNotBuiltError("Knowledge index is empty; call build_index() first.")

        query_embedding = create_embedding(query)
        ranked: list[dict[str, Any]] = []
        for chunk_id, text, metadata_json, embedding_json in rows:
            embedding = json.loads(embedding_json)
            score = sum(
                query_value * document_value
                for query_value, document_value in zip(query_embedding, embedding)
            )
            metadata = json.loads(metadata_json)
            ranked.append(
                {
                    "text": text,
                    "source": metadata["source"],
                    "citation": metadata["citation"],
                    "metadata": metadata,
                    "score": round(float(score), 6),
                    "chunk_id": chunk_id,
                }
            )

        ranked.sort(key=lambda result: (-result["score"], result["chunk_id"]))
        return ranked[: min(top_k, len(ranked))]

    def retrieve(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Backward-compatible alias for :meth:`search_knowledge`."""
        return self.search_knowledge(query, top_k=limit)


def build_knowledge_index(
    persist_directory: Path = CHROMA_STORE_DIR,
    data_directory: Path = WORKING_DATA_DIR,
    include_products: bool = True,
) -> int:
    """Build the configured local knowledge index."""
    return RAGService(
        persist_directory=persist_directory,
        data_directory=data_directory,
        include_products=include_products,
    ).build_index()


def search_knowledge(
    query: str,
    top_k: int = 3,
    persist_directory: Path = CHROMA_STORE_DIR,
) -> list[dict[str, Any]]:
    """Search an existing configured local knowledge index."""
    return RAGService(persist_directory=persist_directory).search_knowledge(query, top_k)
