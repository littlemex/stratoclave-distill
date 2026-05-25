"""Distiller / Curator / Aggregator pipeline.

Stage B exposes :class:`JsonlSessionReader` here so callers can iterate
over a captured session without learning the module layout. Distiller
and Curator land alongside it as their implementations are completed.
"""

from stratoclave_distill.pipeline.aggregator import (
    AggregationResult,
    Aggregator,
    build_aggregate_prompt,
)
from stratoclave_distill.pipeline.curator import (
    ConflictJudge,
    ConflictVerdict,
    CurationAction,
    CurationOutcome,
    Curator,
    CuratorDecision,
)
from stratoclave_distill.pipeline.distiller import (
    CandidateLearning,
    DistillationResult,
    Distiller,
    build_distill_prompt,
)
from stratoclave_distill.pipeline.ingest import (
    BranchPlan,
    IngestReport,
    IngestRunner,
    SessionIngestResult,
)
from stratoclave_distill.pipeline.reader import JsonlSessionReader, SkippedLine

__all__ = [
    "AggregationResult",
    "Aggregator",
    "BranchPlan",
    "CandidateLearning",
    "ConflictJudge",
    "ConflictVerdict",
    "CurationAction",
    "CurationOutcome",
    "Curator",
    "CuratorDecision",
    "DistillationResult",
    "Distiller",
    "IngestReport",
    "IngestRunner",
    "JsonlSessionReader",
    "SessionIngestResult",
    "SkippedLine",
    "build_aggregate_prompt",
    "build_distill_prompt",
]
