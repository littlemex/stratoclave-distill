# stratoclave-distill: Implementation Status

**Last updated**: 2026-05-23
**Project started**: 2026-05-22
**Current stage**: Stage B+ engine implemented (branching, claim_type,
conflict / gap relations, retrieval lanes). Schema migration `0002`
shipped via PR #3; engine + CLI + tests landing in this PR. Next: Stage C
(query / pack / export / gc).

## Overall progress

### Component status

| Component                      | Tests | Status     | Notes |
|--------------------------------|-------|------------|-------|
| `core.types` (frozen dataclasses) | unit  | done    | + `BranchKind`, `BranchState`, `ClaimType`, `LearningConflict`, `SessionGap` |
| `core.errors`                  | unit  | done       | Hierarchy verified |
| `config.DistillerConfig`       | unit  | done       | Env loader + invariants enforced |
| Provider abstractions (Stub / Anthropic / OpenAI / Voyage) | unit | done | Lazy SDK imports, dispatch tested |
| CLI (`version`, `check-config`, `ingest`, `branch close`, `branch list`) | unit | done | `--branch-from / --branch-session / --at-seq` wired |
| Postgres + pgvector schema     | unit (static), integration | done | alembic migrations `0001_initial_schema` + `0002_branching_and_relations` |
| `JsonlSessionReader`           | unit  | done       | Strict / lenient modes, malformed-line reporting |
| `Distiller` (Stage 1 LLM extract) | unit | done    | Prompt extracts `claim_type` (observation / interpretation / signal / norm) |
| `Curator` (4-action: INSERT / MERGE / SUPERSEDE / CONFLICT_NOTED) | unit | done | Borderline cases recorded as `learning_conflicts` |
| `IngestRunner` (orchestrator)  | unit  | done       | Watermark-driven incremental ingest, error isolation |
| Watermark / Purpose / Digest stores | unit + integration | done | In-memory + asyncpg, branching round-trip on PurposeStore |
| `LearningStore` hybrid search  | unit + integration | done | Cosine + ts_rank_cd RRF + canonical / emerging lane gating |
| `ConflictStore` / `GapStore`   | unit + integration | done | In-memory + asyncpg side-relations; partial indexes on unresolved subset |
| `Retriever` (canonical / emerging lanes) | unit | done | Lane gating: `canonical_min_evidence`, `canonical_min_age_days` |
| Branching columns (`parent_session_id`, `branched_at_seq`, `branch_kind`, `branch_state`, `closed_at`) | unit + integration | done | Migration 0002 |
| `learning_conflicts` / `session_gaps` side tables | unit + integration | done | Migration 0002, FK + CHECK constraints |
| `Learning.claim_type` (observation / interpretation / signal / norm) | unit + integration | done | Persisted via Distiller -> Curator -> LearningStore |
| Branch CLI (`ingest --branch-from`, `branch close`, `branch list --tree/--json`) | unit (CLI dry-run) | done | All-or-nothing rule on `--branch-from / --branch-session / --at-seq` |
| `Aggregator` (group rollup)    | -     | planned    | Stage C |
| `ContextPacker` (token budget) | -     | planned    | Stage C |

### Integration status

| Integration             | Status      | Notes |
|-------------------------|-------------|-------|
| docker-compose Postgres | done        | `pgvector/pgvector:pg16` |
| alembic migrations      | done        | `DISTILL_EMBEDDING_DIM` env-driven; both `0001` and `0002` round-trip clean |
| asyncpg + pgvector codec | done       | `init=_register_vector` on every connection |
| asyncpg timestamptz binds | done      | datetime coercion at the boundary, ISO-Z output (PR #1 hotfix) |
| Anthropic LLM           | scaffolded  | Real call exercised in Stage B e2e |
| Voyage embedding        | scaffolded  | Real call exercised in Stage B e2e |
| OpenAI LLM / embedding  | scaffolded  | Optional fallback |
| GitHub Actions CI       | done        | matrix on Python 3.11 / 3.12, ruff + mypy + pytest, integration job green on every PR |

## Completed work

### Stage A — Bootstrap (shipped)

- Repository skeleton, `pyproject.toml`, frozen `core.types`, error
  hierarchy, env-driven `DistillerConfig`, LLM and Embedding provider
  abstractions with `Stub*` implementations, alembic migration
  `0001_initial_schema`, CLI `version` + `check-config`,
  `GETTING_STARTED.md` / `PROJECT_RULES.md` / `PROJECT_STATUS.md` /
  `DESIGN.md`.

### Stage B — Ingest pipeline (shipped via PR #1)

- `JsonlSessionReader`, `Distiller`, `Curator`, `IngestRunner`, store
  Protocols (`WatermarkStore`, `PurposeStore`, `DigestStore`,
  `LearningStore`), in-memory + asyncpg implementations, hybrid search
  via cosine + `ts_rank_cd` RRF in a single SQL CTE, CLI `ingest <path>
  [--dry-run] [--strict] [--version-id ...]` subcommand.
- Audit-trail invariant: `Curator` never deletes a Learning; supersede
  sets `archived_at` + `superseded_by`.

### Stage B+ design — shipped via PR #2

- `docs/STAGE_B_PLUS_DESIGN.md`: data model + erDiagram, claim_type
  taxonomy, conflict / gap side-relations, retrieval lanes, CLI
  surface, visualization tree, migration plan, test plan, rollout
  schedule, Q&A.

### Stage B+ migration `0002_branching_and_relations` (shipped via PR #3)

- Adds branching columns to `session_purposes` (`parent_session_id`,
  `branched_at_seq`, `branch_kind`, `branch_state`, `closed_at`).
- Adds `claim_type` column to `learnings` and widens scope CHECK to
  include `experiment`.
- Adds `learning_conflicts` and `session_gaps` side tables, with
  CHECK-constrained `resolution` enum (`open`, `merged`, `superseded`,
  `coexist`) and a generated `bm25_tsv` column for gap lexical search.
- Partial indexes on the unresolved subset for both side tables.
- alembic round-trip integration test exercises both upgrade and
  downgrade paths.

### Stage B+ engine (this PR)

- Public types extended: `BranchKind = "main" | "experiment"`,
  `BranchState = "open" | "closed" | "promoted"`, `ClaimType =
  "observation" | "interpretation" | "signal" | "norm"`. New
  dataclasses `LearningConflict`, `SessionGap`. `SessionPurpose`
  extended with branching fields, `Learning` extended with
  `claim_type`.
- `Distiller` prompt extracts `claim_type` from the LLM JSON envelope.
  When the LLM omits the field, the value falls through to retrieval
  time where the `signal` lane absorbs it.
- `Curator` promoted to a 4-way action set: `INSERT`, `MERGE`,
  `SUPERSEDE`, `CONFLICT_NOTED`. Borderline candidates that contradict
  an existing Learning but do not pass the supersede gate are
  persisted as `learning_conflicts` rows so the user can see and
  resolve them later.
- `Retriever` exposes canonical / emerging lanes. The canonical lane
  gates on `canonical_min_evidence` and `canonical_min_age_days`; the
  emerging lane returns the rest with the same RRF ordering.
- `db.asyncpg` and `db.memory` both implement the new
  `ConflictStore` / `GapStore` Protocols and round-trip the new
  branching columns / `claim_type` field.
- CLI: `ingest --branch-from <session> --branch-session <new_id>
  --at-seq <int> [--branch-kind experiment|main]` opens a branch;
  `branch close <session_id>` closes it; `branch list [--tree | --json]`
  prints the branch topology. The branch flags are all-or-nothing —
  supplying any one of them without the others is an error.

### Test surface (Stage A + B + B+)

- 277 passing unit tests; 16 integration tests gated on
  `DISTILL_TEST_DATABASE_URL`.
- Integration coverage exercises both alembic round-trips, the asyncpg
  store contracts (Watermark / Purpose / Digest / Learning / Conflict
  / Gap), `search_hybrid` lane filtering, and the branching round-trip
  on `PurposeStore`.

## Technical highlights

- **No hard-coded paths, URLs, or credentials.** Every provider knob is
  routed through `DistillerConfig`. See `PROJECT_RULES.md`.
- **In-memory parity with asyncpg.** The pipeline depends on Protocols,
  so unit tests exercise the real orchestrator with the in-memory
  stores; the integration suite exercises the same Protocols against a
  live Postgres.
- **Audit-trail preservation.** `Curator` never deletes a Learning; the
  4-action set escalates contradictions into `learning_conflicts`
  rather than silently dropping them.
- **Stable / evolving / tunable layering.** Branching identifiers and
  audit-trail invariants are stable; the claim_type vocabulary, the
  canonical lane policy, and the conflict-resolution heuristic are
  evolving; thresholds and display knobs are tunable.
- **Schema is dimension-agnostic.** Both migrations adapt to Voyage
  (1024) or OpenAI text-embedding-3-small (1536) via
  `DISTILL_EMBEDDING_DIM`.

## Future work

| Priority | Item | Stage |
|----------|------|-------|
| P0 | Retriever surface in CLI (`query <text>` with `--lane canonical/emerging`) | C |
| P0 | ContextPacker token-budget rendering                                       | C |
| P0 | CLI `query` / `export` / `gc`                                              | C |
| P1 | Conflict / gap CLI (`conflict list/resolve`, `gap list/resolve`)           | C |
| P1 | Aggregator group rollup (`group_learnings`)                                | C |
| P2 | Optional MCP server                                                        | v0.x |
| P2 | Optional FastAPI HTTP server                                               | v0.x |

## Team / ownership

| Role             | Owner        | Status      | Current task |
|------------------|--------------|-------------|--------------|
| Maintainer       | littlemex    | active      | Stage B+ engine landing; planning Stage C retriever surface |

## Next steps

1. Land Stage B+ engine (this PR): types, stores, Distiller, Curator,
   Retriever, branch CLI, tests.
2. Begin Stage C: wire the existing `Retriever` into a CLI `query`
   subcommand with `--lane canonical/emerging` and JSON / table
   rendering.
3. Implement `ContextPacker` with token-budget eviction.
4. Add `conflict list/resolve` and `gap list/resolve` CLI surfaces so
   the new side-relations are reachable from the operator workflow.
