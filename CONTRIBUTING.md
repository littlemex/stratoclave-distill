# Contributing to stratoclave-distill

Thanks for your interest in improving stratoclave-distill. This document describes
how to set up a development environment, the expectations we hold around
code quality, and the workflow for submitting changes.

By participating you agree to uphold our
[Code of Conduct](./CODE_OF_CONDUCT.md).

> **Note:** stratoclave-distill is in alpha. The public Python API, CLI
> surface, and database schema may change between commits. If you are
> planning non-trivial work, please open an issue to discuss the approach
> first.

## Table of Contents

- [Ways to contribute](#ways-to-contribute)
- [Reporting bugs](#reporting-bugs)
- [Proposing features](#proposing-features)
- [Development setup](#development-setup)
- [Testing](#testing)
- [Coding style](#coding-style)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [Security issues](#security-issues)

## Ways to contribute

- **Bug reports** with clear reproduction steps.
- **Feature proposals** that articulate the problem first, solution second.
- **Documentation improvements** — typo fixes, clearer examples, translations.
- **New LLM / embedding provider adapters** under
  `src/stratoclave_distill/providers/`.
- **Pipeline improvements** to the Distiller, Curator, Aggregator,
  Retriever, or Context Packer.
- **Reviews** of open pull requests from other contributors.

## Reporting bugs

Use the **Bug report** issue template. Include:

- What you expected to happen vs. what actually happened.
- Minimal reproduction steps.
- Environment: commit SHA or release tag, OS, Python version, Postgres /
  pgvector version, and any provider name and version.
- Redacted logs if they help. **Remove secrets** (API keys, OAuth tokens).

Do not report suspected vulnerabilities in public issues — see
[Security issues](#security-issues).

## Proposing features

Use the **Feature request** issue template. Keep the focus on:

1. The problem you're trying to solve, for whom.
2. Your proposed approach.
3. Alternatives you considered and why you discarded them.

We favour small, composable changes. Large features usually require a design
discussion in an issue before a PR is reviewed. New providers should
implement the Protocols in `src/stratoclave_distill/providers/`; pipeline
changes should be accompanied by tests under `tests/unit/pipeline/`.

## Development setup

### Prerequisites

- **Python 3.11+**
- `pip` (or `uv` if you prefer)
- Docker / finch / podman for the local Postgres + pgvector container
  (see `docker-compose.yml`)
- Optional provider extras (`pip install -e ".[anthropic,voyage,openai]"`)
  if you intend to run the e2e tests

### Fork and clone

```bash
git clone https://github.com/<your-username>/stratoclave-distill.git
cd stratoclave-distill
git remote add upstream https://github.com/littlemex/stratoclave-distill.git
```

Work on feature branches created from `main`:

```bash
git checkout -b feat/descriptive-name
```

### Install in editable mode

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the library plus the development extras (`pytest`, `ruff`,
`mypy`).

## Testing

We expect PRs to be accompanied by tests. The standard set:

- **Unit** tests for pure logic (`tests/unit/`); these must run without
  Docker, network, or provider keys.
- **Integration** tests against a live Postgres + pgvector
  (`tests/integration/`); gated by the `integration` marker and the
  `DISTILL_TEST_DATABASE_URL` env var.
- **End-to-end** tests against real LLM / embedding providers
  (`tests/e2e/`); gated by the `e2e` marker. Do not enable these on
  every push — they consume credit.

Run the suite:

```bash
pytest
```

If a test harness is missing for the area you're touching, add one or note
the gap in the PR description.

## Coding style

Formatters and linters are authoritative — run them before pushing.

| Tool   | Command                  |
|--------|--------------------------|
| Format | `ruff format .`          |
| Lint   | `ruff check .`           |
| Types  | `mypy src/ tests/`       |

Other expectations:

- Prefer small, focused modules over large omnibus files.
- Avoid introducing new runtime dependencies without justifying them.
- Public APIs (functions exported from `stratoclave_distill`) must have
  docstrings.
- **Do not hard-code paths, URLs, model names, or any environment-specific
  values.** Use environment variables (see `DistillerConfig`) or function
  arguments. This is a project-wide rule and CI enforces it via
  `scripts/check-no-hardcoded-secrets.sh`.

## Commit messages

We use **[Conventional Commits](https://www.conventionalcommits.org/)**:

```
<type>(<scope>): <short summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`,
`ci`, `perf`. Example:

```
feat(adapters): add OpenCode backend
fix(transport): drain stdout on cancel before SIGTERM
```

Keep commits atomic and write meaningful bodies for non-trivial changes.
Reference issues with `Refs #N` or `Closes #N`.

## Pull requests

1. Rebase on the latest `main` before opening the PR.
2. Fill in the pull-request template completely.
3. Keep PRs focused. If a change grows, split it.
4. Ensure CI is green (formatters, linters, type checks, unit tests).
5. Request review from a maintainer. We typically respond within a week.
6. Address review feedback with additional commits; we squash on merge.

We do not require Contributor License Agreements (CLAs); the Apache-2.0
license covers contributions.

## Security issues

Do **not** report suspected vulnerabilities in public issues or pull
requests. Follow the process in [`SECURITY.md`](./SECURITY.md).

---

If you have questions before filing an issue or PR, feel free to reach out
via GitHub Discussions (once enabled) or an issue tagged `question`.
