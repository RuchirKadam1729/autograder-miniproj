"""Exam Grader — AI-powered automated marking system."""

from .config import AppConfig, LLMConfig, VisionConfig, config, setup_logging
from .grading import GradingResult, MarkBreakdown
from .llm_client import LLMClient, LLMError
from .pipeline import GradingPipeline, PipelineResult
from .synoptic import SynopticEntry, SynopticMap, build_synoptic_map, load_synoptic
from .vision import VisionModel

__all__ = [
    "AppConfig",
    "LLMConfig",
    "VisionConfig",
    "config",
    "setup_logging",
    "GradingResult",
    "MarkBreakdown",
    "LLMClient",
    "LLMError",
    "GradingPipeline",
    "PipelineResult",
    "SynopticEntry",
    "SynopticMap",
    "build_synoptic_map",
    "load_synoptic",
    "VisionModel",
]
