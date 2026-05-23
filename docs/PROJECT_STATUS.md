# stratoclave-distill: Implementation Status

**Last updated**: 2026-05-23
**Project started**: 2026-05-22
**Current stage**: Stage B (ingest pipeline) shipped via PR #1; planning
Stage B+ (branching, claim_type, conflict / gap relations, retrieval
lanes) before Stage C (query / pack / export / gc).

## Overall progress

### Component status

| Component                      | Tests | Status     | Notes |
|--------------------------------|-------|------------|-------|
| `core.types` (frozen dataclasses) | unit  | done    | Round-tripped through pytest, frozen + slots locked in |
| `core.errors`                  | unit  | done       | Hierarchy verified |
| `config.DistillerConfig`       | unit  | done       | Env loader + invariants enforced |
| Provider abstractions (Stub / Anthropic / OpenAI / Voyage) | unit | done | Lazy SDK imports, dispatch tested |
| CLI (`version`, `check-config`, `ingest`) | unit | done | Stage B+ / C subcommands not yet wired |
| Postgres + pgvector schema     | unit (static), integration | done | alembic migration `0001_initial_schema` |
| `JsonlSessionReader`           | unit  | done       | Strict / lenient modes, malformed-line reporting |
| `Distiller` (Stage 1 LLM extract) | unit | done    | One-shot prompt → purpose + digest + learnings |
| `Curator` (dedup / merge / supersede) | unit | done | tau_merge / tau_conflict thresholds, RRF inputs |
| `IngestRunner` (orchestrator)  | unit  | done       | Watermark-driven incremental ingest, error isolation |
| Watermark / Purpose / Digest stores | unit + integration | done | In-memory + asyncpg, monotonic watermark, single-row digest |
| `LearningStore` hybrid search  | unit + integration | done | Cosine + ts_rank_cd RRF in a single SQL CTE |
| Branching columns (`parent_session_id`, `branched_at_seq`, `branch_kind`, `branch_state`) | - | designed | Stage B+ migration 0002 |
| `learning_conflicts` / `session_gaps` side tables | - | designed | Stage B+ migration 0002 |
| `Learning.claim_type` (observation / interpretation / signal / norm) | - | designed | Stage B+ Distiller prompt update |
| Curator action: `CONFLICT_NOTED` (4-way decision) | - | designed | Stage B+ |
| Retriever lanes (canonical / emerging) | - | designed | Stage B+ |
| Branch CLI (`ingest --branch-from`, `branch close`, `branch list --tree`) | - | designed | Stage B+ |
| `Aggregator` (group rollup)    | -     | planned    | Stage C |
| `Retriever` (RRF hybrid surface) | -   | planned    | Stage C (CLI `query`) |
| `ContextPacker` (token budget) | -     | planned    | Stage C |

### Integration status

| Integration             | Status      | Notes |
|-------------------------|-------------|-------|
| docker-compose Postgres | done        | `pgvector/pgvector:pg16` |
| alembic migrations      | done        | `DISTILL_EMBEDDING_DIM` env-driven |
| asyncpg + pgvector codec | done       | `init=_register_vector` on every connection |
| asyncpg timestamptz binds | done      | datetime coercion at the boundary, ISO-Z output (PR #1 hotfix) |
| Anthropic LLM           | scaffolded  | Real call exercised in Stage B e2e |
| Voyage embedding        | scaffolded  | Real call exercised in Stage B e2e |
| OpenAI LLM / embedding  | scaffolded  | Optional fallback |
| GitHub Actions CI       | done        | matrix on Python 3.11 / 3.12, ruff + mypy + pytest, integration job green on every PR |

## Completed work

### Stage A — Bootstrap

- Repository skeleton: `LICENSE`, `README.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.gitignore`.
- `pyproject.toml` with hatchling, ruff, mypy strict, pytest markers
  (`integration`, `e2e`, `slow`).
- Public dataclasses in `core.types`, error hierarchy in `core.errors`,
  env-driven `DistillerConfig` in `config.py`.
- LLM and Embedding provider abstractions with `Stub*` implementations.
- CLI scaffolding with `version` and `check-config`.
- alembic migration `0001_initial_schema`: 5 tables + HNSW + tsvector
  GIN indexes, embedding dimension env-driven.
- Documentation: `GETTING_STARTED.md`, `PROJECT_RULES.md`,
  `PROJECT_STATUS.md`, `DESIGN.md`.

### Stage B — Ingest pipeline (shipped via PR #1)

- `pipeline.reader.JsonlSessionReader` parses JSONL transcripts into
  `NormalizedTurn` instances. Lenient mode reports malformed lines via
  `SkippedLine` records; strict mode aborts on the first parse error.
- `pipeline.distiller.Distiller` implements the Stage 1 extraction:
  builds a single LLM prompt (with optional prior `SessionPurpose`
  context), validates the JSON envelope, and produces a tuple of
  `SessionPurpose`, `SessionDigest`, and a list of `Learning` rows
  alongside their embedding vectors.
- `pipeline.curator.Curator` decides whether each candidate Learning is
  a `merge` (above `tau_merge` cosine), a `supersede` (above
  `tau_conflict` and explicitly contradictory), or an `insert`. The
  cosine threshold is the gate; RRF is used only to rank the candidate
  pool the threshold sees.
- `pipeline.ingest.IngestRunner` orchestrates a full ingest: read JSONL,
  group by `session_id`, skip turns at-or-below the watermark, run the
  Distiller, run the Curator, persist via the four stores, and advance
  the watermark. An error in one session does not block the others;
  strict mode re-raises the first failure.
- `db.stores` Protocols (`WatermarkStore`, `PurposeStore`, `DigestStore`,
  `LearningStore`) define the persistence contract. `LearningSearchHit`
  carries the cosine, the per-modality ranks, and the RRF score so
  callers can apply thresholds without re-running the search.
- `db.memory` provides in-memory implementations for unit tests and
  offline demos. The hybrid search is done in pure Python: cosine on
  every active row plus a deterministic BM25 surrogate, then RRF.
- `db.asyncpg` provides the production implementation against Postgres
  + pgvector. The pool registers the pgvector codec on every
  connection. `search_hybrid` runs a single SQL statement that fuses
  cosine and `ts_rank_cd` ranks via RRF in a CTE. Timestamps are
  coerced to timezone-aware `datetime` at the binary-protocol boundary
  and rendered back as `...Z` ISO-8601 on read so the in-memory store
  and the asyncpg store are byte-equivalent.
- CLI `ingest <path> [--dry-run] [--strict] [--version-id ...]`
  subcommand. Dry-run wires in-memory stores + stub providers (no DB,
  no API keys). Production mode reads `DistillerConfig.from_env()`,
  builds real LLM + Embedding providers, and opens an asyncpg pool.

### Test surface (Stage A + B)

- 225 passing unit tests; 11 integration tests gated on
  `DISTILL_TEST_DATABASE_URL` (alembic migration round-trip + asyncpg
  store contract — watermark monotonic advance, purpose idempotency,
  digest delete-then-insert, learning insert / update / supersede /
  list_active, and `search_hybrid` cosine ordering, archived exclusion,
  scope filtering). All eleven integration tests now run in CI on
  every PR.
- Line + branch coverage on `src/stratoclave_distill`: **92%**. The
  remaining gap is mostly in the asyncpg SQL paths exercised by the
  integration suite.

## Stage B+ — Branching, claim_type, conflict & gap (planned, design done)

`docs/STAGE_B_PLUS_DESIGN.md` is the single source for this milestone.
The rollout is split into small PRs so each step can be reviewed and
deployed independently.

| Step | Scope | Tests | Migration |
|------|-------|-------|-----------|
| 1    | Land design doc + status refresh                        | -                            | -    |
| 2    | Migration `0002_branching_and_relations`                | alembic round-trip           | 0002 |
| 3    | Public types: `BranchKind`, `BranchState`, `ClaimType`  | unit                         | -    |
| 4    | Distiller prompt: extract `claim_type` (default: signal)| unit (stub LLM)              | -    |
| 5    | Curator 4-way action set (`CONFLICT_NOTED`)             | unit + integration           | -    |
| 6    | Retriever lanes (`canonical` / `emerging`)              | unit + integration           | -    |
| 7    | CLI: `ingest --branch-from`, `branch close`, `branch list --tree` | unit (CLI smoke + dry-run) | -    |

The design notes the **stable / evolving / tunable** layering:
branching identifiers and audit-trail invariants live in the stable
layer; the specific claim_type vocabulary, the canonical lane policy,
and the conflict-resolution heuristic live in the evolving layer;
thresholds and display knobs live in the tunable layer.

## Technical highlights

- **No hard-coded paths, URLs, or credentials.** Every provider knob is
  routed through `DistillerConfig`. See `PROJECT_RULES.md`.
- **In-memory parity with asyncpg.** The pipeline depends on Protocols,
  so unit tests exercise the real orchestrator with the in-memory
  stores; the integration suite exercises the same Protocols against a
  live Postgres.
- **Watermark-driven incremental ingest.** Re-running `ingest` on an
  appended JSONL only distills the new turns, and a session-level
  failure does not corrupt other sessions' watermarks.
- **Audit-trail preservation.** `Curator` never deletes a Learning;
  `supersede` sets `archived_at` + `superseded_by` on the old row.
- **Schema is dimension-agnostic.** A single migration adapts to Voyage
  (1024) or OpenAI text-embedding-3-small (1536) via
  `DISTILL_EMBEDDING_DIM`.

## Future work

| Priority | Item | Stage |
|----------|------|-------|
| P0 | Stage B+ migration 0002 + types + Distiller / Curator wiring | B+ |
| P0 | Stage B+ retrieval lanes + branch CLI                        | B+ |
| P0 | Retriever RRF hybrid search (CLI `query`)                    | C  |
| P0 | ContextPacker token-budget rendering                         | C  |
| P0 | CLI `query` / `export` / `gc`                                | C  |
| P1 | Aggregator group rollup (`group_learnings`)                  | C  |
| P2 | Optional MCP server                                          | v0.x |
| P2 | Optional FastAPI HTTP server                                 | v0.x |

## Team / ownership

| Role             | Owner        | Status      | Current task |
|------------------|--------------|-------------|--------------|
| Maintainer       | littlemex    | active      | Stage B shipped (PR #1); landing Stage B+ design |

## Next steps

1. Land Stage B+ design doc (`STAGE_B_PLUS_DESIGN.md`) + this status
   refresh as a docs-only PR.
2. Implement migration `0002_branching_and_relations` with downgrade
   path and integration test.
3. Add `BranchKind` / `BranchState` / `ClaimType` to `core.types` and
   re-export from `stratoclave_distill.__init__`.
4. Extend the Distiller prompt to extract `claim_type`; default to
   `signal` when the LLM omits it.
5. Promote the Curator to a 4-way action set, persisting borderline
   cases as `learning_conflicts` rows instead of swallowing them.
6. Introduce retrieval lanes and the branch CLI surface.
7. Begin Stage C (Retriever / ContextPacker / `query` / `export`
   / `gc`) once Stage B+ lands.
