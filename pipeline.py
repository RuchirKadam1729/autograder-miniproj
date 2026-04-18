"""
Top-level pipeline orchestrator.
Wires together extraction → grading → report for a single paper.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .extraction import extract_answers
from .grading import build_report, fill_missing_questions, grade_all_answers
from .llm_client import LLMClient
from .synoptic import SynopticMap, build_synoptic_map
from .vision import VisionModel

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    status: str           # "success" | "failed"
    paper_name: str
    total_score: float = 0.0
    total_max: float = 0.0
    percentage: float = 0.0
    report_text: str = ""
    graded_df: pd.DataFrame = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.graded_df is None:
            self.graded_df = pd.DataFrame()


class GradingPipeline:
    """
    Stateful pipeline object that holds loaded models and orchestrates
    the three-phase process (extract → grade → report) for each paper.
    """

    def __init__(
        self,
        config: AppConfig,
        vision_model: VisionModel,
        llm_client: LLMClient,
    ) -> None:
        self._config = config
        self._vision = vision_model
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        paper_path: str | Path,
        synoptic_map: SynopticMap,
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """
        Process one student paper end-to-end.

        Parameters
        ----------
        paper_path:   Path to the student's answer PDF.
        synoptic_map: Pre-built marking scheme map.
        output_dir:   Optional directory to save CSV artefacts.

        Returns
        -------
        PipelineResult with all grading details.
        """
        paper_path = Path(paper_path)
        paper_name = paper_path.stem
        out = Path(output_dir) if output_dir else Path(self._config.output_dir) / paper_name
        out.mkdir(parents=True, exist_ok=True)

        logger.info("═" * 60)
        logger.info("Processing: %s", paper_name)

        # ── Phase 1: Extraction ──────────────────────────────────────────
        logger.info("Phase 1: Extracting answers …")
        try:
            answers_df, _ = extract_answers(
                pdf_path=paper_path,
                synoptic_map=synoptic_map,
                vision_model=self._vision,
                llm_client=self._llm,
            )
        except Exception as exc:
            logger.exception("Extraction failed")
            return PipelineResult(status="failed", paper_name=paper_name, reason=str(exc))

        if answers_df.empty:
            return PipelineResult(
                status="failed", paper_name=paper_name, reason="no_answers_extracted"
            )

        self._save(answers_df, out / "1_extracted.csv")

        # ── Phase 2: Grading ─────────────────────────────────────────────
        logger.info("Phase 2: Grading …")
        try:
            graded_df = grade_all_answers(answers_df, synoptic_map, self._llm)
        except Exception as exc:
            logger.exception("Grading failed")
            return PipelineResult(status="failed", paper_name=paper_name, reason=str(exc))

        if graded_df.empty:
            return PipelineResult(
                status="failed", paper_name=paper_name, reason="grading_returned_empty"
            )

        graded_df = fill_missing_questions(graded_df, synoptic_map)
        self._save(graded_df, out / "2_graded.csv")

        # ── Phase 3: Report ──────────────────────────────────────────────
        logger.info("Phase 3: Building report …")
        report = build_report(graded_df, synoptic_map, paper_name)
        (out / "RESULTS.txt").write_text(report, encoding="utf-8")

        total_max = sum(e.marks for e in synoptic_map.values())
        total_score = graded_df["marks_awarded"].sum()
        percentage = (total_score / total_max * 100) if total_max > 0 else 0.0

        logger.info("Result: %.1f / %.1f  (%.1f%%)", total_score, total_max, percentage)

        return PipelineResult(
            status="success",
            paper_name=paper_name,
            total_score=float(total_score),
            total_max=float(total_max),
            percentage=float(percentage),
            report_text=report,
            graded_df=graded_df,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug("Saved %s (%d rows)", path.name, len(df))
