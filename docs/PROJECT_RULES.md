# stratoclave-distill: Project Rules

**Last updated**: 2026-05-22

These rules apply to every change merged into `main`. They are stricter than
loom's because distill carries persistent state and exposes a query surface
that other systems will rely on.

## Coding standards

- Python 3.11+ only. `from __future__ import annotations` at the top of
  every module.
- `ruff check` (with the rules enabled in `pyproject.toml`) and
  `ruff format` must be clean.
- `mypy --strict` must be clean across `src/stratoclave_distill`.
- Every public dataclass is `@dataclass(frozen=True, slots=True)`.
- Every Protocol that adapters implement uses `@runtime_checkable` so we
  can sanity-check at fixture-construction time.
- Imports: standard library first, then third-party, then local
  (`stratoclave_distill.*`). ruff's isort handles this.

## Testing policy

distill is a complex pipeline; testing therefore is the single most
important quality gate. The bar:

- **Unit tests must accompany every public function or method.** A change
  that adds a new code path without a test is not ready to merge.
- **Tests document the contract.** Each test's docstring explains why the
  assertion exists, not what the code does.
- **Three layers of tests**:
  - `tests/unit` — fast, no external services, runs on every commit.
  - `tests/integration` — requires the docker-compose Postgres; gated by
    the `integration` marker and `DISTILL_TEST_DATABASE_URL`.
  - `tests/e2e` — requires real LLM / embedding providers and consumes
    credit; gated by the `e2e` marker.
- **Stubs are first-class fixtures**. Pipeline tests use `StubLLM` and
  `StubEmbedding` so they stay deterministic and offline.
- **Coverage target**: ≥ 90 % line coverage on `src/stratoclave_distill`,
  measured via `pytest --cov`. CI fails below 80 %.

## No-hardcode policy

This is non-negotiable. The following are explicitly forbidden in
production code:

- Database URLs, hostnames, ports.
- LLM / embedding model identifiers in `src/` (test fixtures may carry
  defaults explicitly).
- API keys, tokens, or any secret material.
- Absolute filesystem paths to user-specific directories.

Everything must flow through `DistillerConfig` or a documented environment
variable. CI enforces a smoke check: a grep for hostnames and `sk-` /
`pa-` token prefixes runs on every PR.

## Migrations

- Every schema change is a new alembic revision under
  `migrations/versions/`. We do not edit revisions that have shipped.
- Migrations must include a meaningful `downgrade()` so we can roll back
  in production.
- Embedding dimension is read from `DISTILL_EMBEDDING_DIM` so the same
  migration adapts to multiple providers.
- A migration that creates a table also creates its required indexes in
  the same revision; we never ship a migration that produces a slow
  query path.

## Git workflow

- Branch from `main`; one logical change per PR.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat:` new functionality
  - `fix:` bug fix
  - `chore:` build / tooling
  - `docs:` documentation only
  - `test:` test-only change
  - `refactor:` no-behaviour-change cleanup
- **No `Co-Authored-By` trailers.** This is a strict project rule.
- **No force-push to `main`.** Use `--force-with-lease` on feature
  branches if you must rewrite history.
- PRs require ruff, mypy, and pytest to pass before merge.

## Communication

- Issues, PRs, and commit messages are written in English.
- Documentation comments inside the codebase are also English so the
  surface looks consistent to OSS contributors.
- Internal design discussions in private channels may be Japanese; once a
  decision is made, mirror it into the relevant doc in English.

## Specific constraints

- **Async everywhere on the I/O boundary.** Every provider call,
  database call, and pipeline stage entrypoint is `async`. We do not
  silently call `asyncio.run` inside library code.
- **Determinism in tests.** Tests must not depend on wall-clock time,
  network access, or random numbers. Use the explicit fixtures.
- **No silent error swallowing.** Provider errors are wrapped in
  `LLMError` / `EmbeddingError` so the caller can react; we never
  catch-and-log without re-raising.
- **No emojis in code, docs, commit messages, or filenames.** Reserved
  for the rare situation where the user explicitly asks. (See
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).)

## Review checklist

Reviewers should refuse to approve a PR until:

- [ ] Tests cover every new code path.
- [ ] No hard-coded paths / URLs / secrets.
- [ ] mypy strict is clean.
- [ ] ruff lint and format are clean.
- [ ] If the PR touches the schema: a new alembic revision is included
      with a working downgrade path.
- [ ] Public API additions are exported from `stratoclave_distill.__init__`
      and reflected in `docs/PROJECT_STATUS.md`.
- [ ] No `Co-Authored-By` lines in any commit.
