"""branching and relations: claim_type, branch state, learning_conflicts, session_gaps

Revision ID: 0002_branching_and_relations
Revises: 0001_initial_schema
Create Date: 2026-05-23 00:00:00

Stage B+ extends the schema along three independent axes:

1. ``session_purposes`` records branching topology so an experimental
   branch can sit alongside main without duplicating the parent's
   learnings. The columns are nullable on existing rows; the ``main`` /
   ``experiment`` / ``open`` / ``closed`` / ``promoted`` enums are the
   stable layer of the design.

2. ``learnings`` carries an optional ``claim_type`` so the retriever and
   the canonical lane can tell observation from interpretation from
   signal from norm. Existing rows stay NULL — the retriever falls back
   to ``signal`` semantics when claim_type is absent.

3. Two side-relation tables, ``learning_conflicts`` and
   ``session_gaps``, store first-class records of contradictions
   between learnings and unresolved questions a session noted but did
   not answer.

The migration is purely additive on existing data: ``main`` rows keep
``branch_kind = 'main'`` and ``branch_state = 'open'`` from the column
defaults. Downgrade rejects rows that hit the new states
(``branch_state = 'promoted'`` or ``scope = 'experiment'``) loudly so
operators do not silently lose Stage B+ semantics.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_branching_and_relations"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE session_purposes
          ADD COLUMN parent_session_id UUID NULL
            REFERENCES session_purposes(session_id),
          ADD COLUMN branched_at_seq BIGINT NULL,
          ADD COLUMN branch_kind TEXT NOT NULL DEFAULT 'main'
            CHECK (branch_kind IN ('main', 'experiment')),
          ADD COLUMN branch_state TEXT NOT NULL DEFAULT 'open'
            CHECK (branch_state IN ('open', 'closed', 'promoted')),
          ADD COLUMN closed_at TIMESTAMPTZ NULL
        """
    )
    op.execute(
        "CREATE INDEX idx_session_purposes_parent ON session_purposes(parent_session_id)"
    )
    op.execute(
        "CREATE INDEX idx_session_purposes_branch_state "
        "ON session_purposes(branch_state) WHERE branch_state <> 'closed'"
    )

    op.execute(
        """
        ALTER TABLE learnings
          ADD COLUMN claim_type TEXT NULL
            CHECK (claim_type IS NULL OR claim_type IN
              ('observation', 'interpretation', 'signal', 'norm'))
        """
    )
    op.execute("ALTER TABLE learnings DROP CONSTRAINT IF EXISTS learnings_scope_check")
    op.execute(
        "ALTER TABLE learnings ADD CONSTRAINT learnings_scope_check "
        "CHECK (scope IN ('session', 'project', 'group', 'shared', 'experiment'))"
    )

    op.execute(
        """
        CREATE TABLE learning_conflicts (
            conflict_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            from_id              UUID NOT NULL
              REFERENCES learnings(learning_id) ON DELETE CASCADE,
            to_id                UUID NOT NULL
              REFERENCES learnings(learning_id) ON DELETE CASCADE,
            reason               TEXT NOT NULL,
            cosine_at_detection  REAL NOT NULL,
            detected_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolution           TEXT NOT NULL DEFAULT 'open'
              CHECK (resolution IN ('open', 'merged', 'superseded', 'coexist'))
        )
        """
    )
    op.execute("CREATE INDEX learning_conflicts_from_idx ON learning_conflicts(from_id)")
    op.execute("CREATE INDEX learning_conflicts_to_idx ON learning_conflicts(to_id)")
    op.execute(
        "CREATE INDEX learning_conflicts_unresolved "
        "ON learning_conflicts(detected_at) WHERE resolution = 'open'"
    )

    op.execute(
        """
        CREATE TABLE session_gaps (
            gap_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id            UUID NOT NULL
              REFERENCES session_purposes(session_id) ON DELETE CASCADE,
            topic                 TEXT NOT NULL,
            why_unknown           TEXT NOT NULL,
            bm25_text             TEXT NOT NULL DEFAULT '',
            bm25_tsv              tsvector GENERATED ALWAYS AS
              (to_tsvector('simple', bm25_text)) STORED,
            detected_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            resolved_at           TIMESTAMPTZ NULL,
            resolved_by_learning  UUID NULL REFERENCES learnings(learning_id)
        )
        """
    )
    op.execute("CREATE INDEX session_gaps_session_idx ON session_gaps(session_id)")
    op.execute(
        "CREATE INDEX session_gaps_unresolved "
        "ON session_gaps(detected_at) WHERE resolved_at IS NULL"
    )
    op.execute("CREATE INDEX session_gaps_bm25_idx ON session_gaps USING GIN(bm25_tsv)")


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM session_purposes WHERE branch_state = 'promoted'
            ) THEN
                RAISE EXCEPTION
                  'cannot downgrade: session_purposes has branch_state=''promoted''. '
                  'Reset to ''open'' or ''closed'' before downgrading.';
            END IF;
            IF EXISTS (
                SELECT 1 FROM learnings WHERE scope = 'experiment'
            ) THEN
                RAISE EXCEPTION
                  'cannot downgrade: learnings has scope=''experiment''. '
                  'Reassign scope before downgrading.';
            END IF;
        END$$;
        """
    )

    op.execute("DROP TABLE IF EXISTS session_gaps")
    op.execute("DROP TABLE IF EXISTS learning_conflicts")

    op.execute("ALTER TABLE learnings DROP CONSTRAINT IF EXISTS learnings_scope_check")
    op.execute(
        "ALTER TABLE learnings ADD CONSTRAINT learnings_scope_check "
        "CHECK (scope IN ('session', 'project', 'group', 'shared'))"
    )
    op.execute("ALTER TABLE learnings DROP COLUMN IF EXISTS claim_type")

    op.execute("DROP INDEX IF EXISTS idx_session_purposes_branch_state")
    op.execute("DROP INDEX IF EXISTS idx_session_purposes_parent")
    op.execute(
        """
        ALTER TABLE session_purposes
          DROP COLUMN IF EXISTS closed_at,
          DROP COLUMN IF EXISTS branch_state,
          DROP COLUMN IF EXISTS branch_kind,
          DROP COLUMN IF EXISTS branched_at_seq,
          DROP COLUMN IF EXISTS parent_session_id
        """
    )
