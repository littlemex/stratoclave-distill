# stratoclave-distill: Implementation Status

**Last updated**: 2026-05-25
**Project started**: 2026-05-22
**Current stage**: Stage D shipped (Aggregator + group rollup pipeline).
The `Aggregator` produces one `GroupLearning` per group_id from the LLM,
`AsyncpgGroupLearningStore` persists it (audit-trail preserving), the
`Retriever` exposes the latest rollup per group, and the `ContextPacker`
emits a `## Group rollups` section above the canonical / emerging lanes.
A new `aggregate run --group-id <id> [--dry-run]` and `aggregate list`
CLI complete the operator surface. Stage A/B/B+/C all remain green.
Next: Stage E (conflict / gap CLI, MCP / HTTP servers).

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
| `ContextPacker` (token budget, lane × claim_type grouping) | unit | done | Approximate token counter; pluggable `TokenCounter` callable |
| `query` CLI (`--lane / --limit / --pack / --token-budget / --dry-run`) | unit (dry-run) | done | Wraps `Retriever.retrieve()`; JSON or Markdown output |
| `export` CLI (`<session_id> [--include-archived] [--include-side-relations]`) | unit (smoke) | done | Dumps purpose + digest + learnings (+ conflicts / gaps) as JSON |
| `gc` CLI (`--older-than-days N [--apply]`) | unit (smoke) | done | Dry-run by default; DELETE only behind explicit `--apply` |
| `GroupLearning` dataclass + `GroupLearningStore` Protocol | unit | done | Frozen, slots; `upsert / get / list_by_group / list_latest_per_group / search_hybrid` |
| `InMemoryGroupLearningStore` / `AsyncpgGroupLearningStore` | unit + integration | done | RRF over latest-per-group; HNSW + GIN indexes ready in migration `0001` |
| `Aggregator` (group rollup)    | unit  | done       | One LLM call → one embedding → one `AggregationResult`; audit-preserving (re-aggregate appends a new row) |
| `Retriever.groups`             | unit  | done       | Surfaces the latest rollup per `group_id`; cosine + BM25 RRF, configurable `top_k_groups` |
| `ContextPacker` group section  | unit  | done       | `## Group rollups` H2 emitted before canonical / emerging; oversized rollups dropped atomically |
| `aggregate run / aggregate list` CLI | unit (dry-run, env-gating) | done | `--group-id` validated; `aggregate list [--group <id>]` falls through to `list_latest_per_group` |

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

### Stage B+ engine (shipped via PR #4)

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

### Stage C P0 surface (this PR)

- `retrieval.ContextPacker`: budgeted Markdown formatter that groups
  `RetrievalResult` hits by lane (canonical → emerging) and `claim_type`
  (`norm` → `observation` → `interpretation` → `signal`). Greedy
  admission with a pluggable `TokenCounter`; the default is a pure
  `ceil(len/chars_per_token)` approximator so the runtime stays free
  of `tiktoken` / `transformers`. Optional sidecar sections for open
  conflicts and gaps.
- `query <text>` CLI: wraps `Retriever.retrieve()` and prints either
  the full `RetrievalResult` as JSON (default) or a packed Markdown
  bundle (`--pack`). Lane filter (`--lane canonical|emerging|both`),
  result cap (`--limit N`), scope filter, gap session selector, and
  token budget override are all wired. `--dry-run` short-circuits
  without touching the database so downstream tooling can pipe
  arbitrary queries through the CLI shape without a Postgres backend.
- `export <session_id>` CLI: dumps purpose + digest + learnings as a
  single JSON payload, with optional `--include-archived` (superseded
  rows) and `--include-side-relations` (conflicts / gaps).
- `gc` CLI: dry-run-by-default cleanup of archived `learnings` rows
  older than `--older-than-days` (default 90). The destructive
  `DELETE ... RETURNING 1` is gated behind explicit `--apply`; without
  it, only counts are reported. Negative ages are rejected at the
  CLI parser level.

### Stage D — Aggregator + group rollup (this PR)

- New public type `GroupLearning` (frozen, slots): a per-group rollup
  with `group_learning_id`, `group_id`, `summary_md`,
  `contributing_learnings`, `bm25_text`, `created_at`. The `embedding`
  rides alongside the dataclass via a separate kwarg on the store
  contract so `GroupLearning` itself stays JSON-friendly.
- `GroupLearningStore` Protocol: `upsert / get / list_by_group /
  list_latest_per_group / search_hybrid`. Re-aggregation produces a
  *new* `group_learning_id` and inserts a fresh row so the audit
  trail survives; `latest_only=True` (default) on `list_by_group` and
  the `DISTINCT ON (group_id)` dedup inside `search_hybrid` keep the
  retrieval surface aligned with the most recent rollup per group.
- `InMemoryGroupLearningStore` and `AsyncpgGroupLearningStore` both
  implement the Protocol; the asyncpg path uses the `group_learnings`
  table reserved by migration `0001` (HNSW vector index + GIN tsvector
  index already present, no new migration needed).
- `Aggregator` pipeline: single LLM call returning one JSON object
  `{"summary_md": str, "bm25_text": str}`, single embedding call,
  deterministic dataclass output. Caller selects which `Learning`
  rows to feed (e.g. `LearningStore.list_active` filtered by
  `group_id`); the Aggregator raises `LLMError` on degenerate input
  (empty group_id, no learnings, or learning whose `group_id` does
  not match).
- `Retriever` extended: `RetrievalResult.groups` carries the latest
  rollup per `group_id`; `top_k_groups` (default 3) is configurable
  and shares the `rrf_k` constant with the per-row hybrid search.
- `ContextPacker` extended: a `## Group rollups` H2 is emitted
  immediately after the title / query echo and before the lane loop,
  with one `### Group rollup: <group_id> [<group_learning_id>]` H3
  per rollup. Oversized rollups are dropped atomically (no partial
  blocks); the lane loop continues admitting hits with the remaining
  budget.
- CLI: `aggregate run --group-id <id> [--dry-run]` runs the pipeline
  and either prints the result envelope (dry-run; uses an in-memory
  fixture and a deterministic stub LLM response so smoke tests
  produce stable JSON) or persists via `AsyncpgGroupLearningStore`
  (prod path; honors `DistillerConfig.from_env`). `aggregate list
  [--group <id>]` falls through to either `list_by_group(latest_only=
  False)` (history for one group) or `list_latest_per_group()` (one
  row per group across the whole table).

### Test surface (Stage A + B + B+ + C + D)

- 338 passing unit tests; 21 integration tests gated on
  `DISTILL_TEST_DATABASE_URL` (Stage D adds 6: round-trip, latest
  per group, history audit, hybrid search dedup-to-latest, two-group
  ranking, and `Aggregator → AsyncpgGroupLearningStore.upsert`
  end-to-end). Stage D unit tests added: 14 group-learning store
  tests, 12 Aggregator tests, 5 retriever group-tier tests, 4
  packer group-section tests, and 3 CLI aggregate tests.
- Integration coverage exercises both alembic round-trips, the asyncpg
  store contracts (Watermark / Purpose / Digest / Learning / Conflict
  / Gap / GroupLearning), `search_hybrid` lane filtering, the
  branching round-trip on `PurposeStore`, and the full Aggregator →
  GroupLearningStore round-trip with the StubLLM / StubEmbedding
  providers and a real pgvector index.

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
| P1 | Conflict / gap CLI (`conflict list/resolve`, `gap list/resolve`) | E |
| P1 | Token budget: tiktoken-backed counter as opt-in extra            | E |
| P2 | Optional MCP server                                              | v0.x |
| P2 | Optional FastAPI HTTP server                                     | v0.x |

## Team / ownership

| Role             | Owner        | Status      | Current task |
|------------------|--------------|-------------|--------------|
| Maintainer       | littlemex    | active      | Stage B+ engine landing; planning Stage C retriever surface |

## Next steps

1. Land Stage D (this PR): `GroupLearning` types, `GroupLearningStore`
   Protocol + InMemory + asyncpg, `Aggregator` pipeline, `Retriever`
   group tier, `ContextPacker` group section, `aggregate run / list`
   CLI, integration tests, docs refresh.
2. Stage E P1: `conflict list/resolve` and `gap list/resolve` CLI so
   the side-relations from Stage B+ are reachable from the operator
   workflow.
3. Stage E P1: opt-in `tiktoken`-backed `TokenCounter` extra so the
   pack budget can match the deployed model exactly.
4. v0.x: Optional MCP server and FastAPI HTTP server fronting the
   existing pipeline + retriever surface.
