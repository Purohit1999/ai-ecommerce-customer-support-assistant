# Technical Report

## Scope and assumptions

This project is a local-first proof of concept for e-commerce customer support.
It covers FAQ and policy retrieval, product information, order tracking,
simulated return-eligibility checks, refund lookup, clarification, and safe
escalation. Runtime knowledge and transaction data come from project-ready files
under `data/`; the retained extracted dataset and ZIP are source inputs.

The transaction records are small demonstration datasets: 9 orders, 5 returns,
and 4 refunds. The knowledge corpus also contains 5 products. The system is not
a production commerce integration: it has no customer authentication, payment
processing, carrier connection, mutation workflow, return persistence, or live
human-ticket handoff.

## Architecture and data pipeline

`app.py` initializes a `RAGService`, building the local index only if it is
missing or empty, then creates a `CustomerSupportAgent`. A fully enabled optional
provider configuration wraps that base agent in `LLMEnhancedSupportAgent`.

The base agent selects one of six outcomes:

1. RAG retrieval for FAQ, policy, and product questions.
2. Local order-status lookup.
3. Simulated return-eligibility lookup.
4. Local refund-status lookup.
5. Clarification for missing input.
6. Human-escalation guidance for unsupported, unavailable, or weakly grounded
   requests.

RAG reads `data/faq.md`, `data/policies.md`, and `data/products.json`. The tools
read `data/orders.json`, `data/returns.json`, and `data/refunds.json`.
`evaluate.py` reads `data/evaluation_queries.csv`, builds an isolated temporary
index, evaluates the same deterministic agent, and writes
`report/evaluation_report.md`.

## Prompt and agent design

`prompts.py` defines support tone, minimum-information clarification messages,
escalation messages, safety rules, a RAG context template, and grounded rewrite
instructions. The central constraints are to use only retrieved knowledge or
verified local tool results, avoid inferred transactional facts, preserve return
simulation warnings, and escalate when evidence is inadequate.

`agent.py` is an ordered deterministic router, not an LLM classifier. It checks
refund actions before return actions, then order-status actions and RAG topics.
Account-specific actions require an ID matching `ORD-` plus 4-12 digits, and a
return action also requires a reason. Each response has a stable schema:
`route`, `answer`, `sources`, `tool_result`, and `escalation`.

This approach makes routing and tool selection reproducible. It does not perform
autonomous LLM tool calling.

## RAG implementation

Level-two sections in the FAQ and policy Markdown files become chunks. Sections
over approximately 1,200 characters are divided near paragraph, line, or word
boundaries with 120 characters of overlap. Each product JSON object becomes one
structured chunk.

The embedding function tokenizes lowercase alphanumeric unigrams and adjacent
bigrams. BLAKE2b maps each feature to a signed position in a 384-dimensional
vector; feature counts are log-scaled and the result is L2-normalized. The
current corpus creates 18 chunks: 6 FAQ, 7 policy, and 5 product chunks.

Text, metadata, and JSON-encoded vectors are stored in
`chroma_store/knowledge.sqlite3`. Despite the directory name, this is a
standard-library SQLite store, not ChromaDB. Query vectors are produced by the
same function, and retrieval performs a full scan ranked by cosine-equivalent dot
product with deterministic chunk-ID tie-breaking. Results expose text, source,
citation, metadata, score, and chunk ID. The agent escalates when the top score
is below `0.05`.

The design is reproducible, local, and requires no model download, but hash
embeddings are lexical rather than transformer-based semantic embeddings and the
full-scan design is intended only for a small corpus.

## Mock tools

- `get_order_status` validates and normalizes an order ID, then returns the
  matching local order or a structured error.
- `create_return_request` reads existing order and return records, reports the
  recorded eligibility decision, and always returns `simulated: true` and
  `persisted: false`. It never writes a return.
- `get_refund_status` validates that the order exists and returns all matching
  refunds. A known order without a refund is a successful empty result, distinct
  from an unknown order.

All three tools are deterministic and return JSON-friendly structures. Malformed
input and invalid or unavailable local data are handled without fabricating a
business outcome.

## Optional LLM layer

`llm.py` implements an environment-gated adapter for the OpenAI Responses API.
The integration is disabled unless `SUPPORT_LLM_ENABLED` is true and the
provider, model, and API key are all present. `.env.example` documents these
variables, but the application reads the process environment and does not load a
`.env` file automatically.

The deterministic agent always runs first. Only non-escalated successful RAG,
order, return, and refund routes may be rewritten. The adapter receives the
original query plus the authoritative answer, source metadata, and tool result;
it cannot route or invoke tools. Candidate text must pass a conservative
vocabulary-based grounding check, and return answers must retain a simulation and
non-persistence warning. Invalid output, construction errors, and provider errors
fall back to the unchanged local answer.

Optional OpenAI LLM integration is implemented and adapter-tested with test
doubles and blocked sockets. Live-provider execution was not tested; this report
does not claim live API validation, operational model behavior, latency, cost, or
deployment readiness.

## Safety and privacy

- Offline/default mode makes no external model request and requires no API key.
- Transaction data is read-only at runtime, and returns are explicitly simulated.
- Missing inputs cause clarification; unavailable data and low-confidence or
  unsupported requests cause advisory escalation.
- The UI catches unexpected response/startup failures and returns a safe result
  without exposing stack traces.
- Credentials are read only from process environment variables and the API key is
  excluded from the settings representation.
- Enabling the provider transmits the customer query and grounded context
  externally. A real deployment needs consent, minimization/redaction, retention,
  access-control, and governance review.
- There is no operational human handoff, so users must contact the stated support
  channel themselves.

## Evaluation methodology and results

The deterministic evaluator runs 25 curated cases with temporary index storage
and no network or LLM judge. It compares the actual route with the expected
route, checks explicit nested tool fields, verifies expected RAG source and fact,
requires the answer to contain the top retrieved chunk, checks clarification
content, and compares escalation flags. A case passes only when all applicable
checks pass.

| Metric | Passed | Total | Accuracy |
|---|---:|---:|---:|
| Routing accuracy | 25 | 25 | 100% |
| Tool correctness | 9 | 9 | 100% |
| RAG source/grounding correctness | 8 | 9 | 88.9% |
| Clarification correctness | 4 | 4 | 100% |
| Escalation correctness | 25 | 25 | 100% |
| Overall pass rate | 24 | 25 | 96% |

Q010 is the only failed case. “What warranty comes with electronics?” routes
correctly to RAG, but the lexical hash retriever places a privacy-policy chunk
above the warranty chunk. The authoritative answer therefore lacks the expected
`6–24 months` fact. This known result is retained rather than changing the
benchmark or claiming 100% evaluation accuracy.

Run the implementation checks from the project root with:

```powershell
python -m pytest
# Console-script alternative in a correctly activated environment:
pytest
python evaluate.py
```

If a global `pytest` console entry point cannot import root-level project
modules, the module invocation avoids that environment-specific path issue.

The automated suite covers UI helpers, routing and fallbacks, RAG, the three
tools, evaluation logic, and the optional LLM wrapper. Relevant tests block
network sockets, use temporary indexes, and verify that return-tool execution
does not modify the source JSON.

## Limitations and future work

Current limitations are lexical retrieval errors such as Q010, rule-based intent
coverage, a small curated evaluation set, full-scan SQLite ranking, synthetic
tool data, no authentication, no persistent commerce actions, no operational
escalation, and a grounding validator that may reject safe paraphrases. The
OpenAI adapter also lacks live-provider validation.

Future work could add a locally packaged semantic embedding model, hybrid and
section-aware retrieval, broader adversarial evaluation, authenticated commerce
connectors, explicit confirmation around mutations, live ticket integration,
observability, and approved provider testing in a secured environment.

`requirements.txt` still lists `chromadb` and `sentence-transformers` from the
initial scaffold, although neither is used by the implemented retrieval path.
Separating or removing unused dependencies is a future maintenance task.
