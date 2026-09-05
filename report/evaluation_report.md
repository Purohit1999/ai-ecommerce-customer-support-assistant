# Offline Evaluation Report

Cases evaluated: **25**

| Metric | Passed | Total | Accuracy |
|---|---:|---:|---:|
| Routing accuracy | 25 | 25 | 100.0% |
| Tool correctness | 9 | 9 | 100.0% |
| RAG source/grounding correctness | 8 | 9 | 88.9% |
| Clarification correctness | 4 | 4 | 100.0% |
| Escalation correctness | 25 | 25 | 100.0% |
| Overall pass rate | 24 | 25 | 96.0% |

## Failed cases

- **Q010** (rag): answer missing '6–24 months'; RAG source or grounding did not match expectations

## Method and limitations

- Deterministic expectations only; no LLM judge or network access.
- RAG correctness requires the expected source and fact, with the answer containing the top retrieved chunk.
- Tool checks compare explicit nested fields against local JSON data.
- The low-confidence case is synthetic and tied to the current hashing embeddings.
- This small curated set measures regression behavior, not production quality.
