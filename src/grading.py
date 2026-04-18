"""
Grading pipeline.
Takes extracted answers + synoptic map → awarded marks + breakdown.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .llm_client import LLMClient
from .synoptic import SynopticEntry, SynopticMap, build_synoptic_map, find_entry, get_all_topics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MarkBreakdown:
    point: str
    allocated_marks: float
    marks_awarded: float
    explanation: str


@dataclass
class GradingResult:
    question: str
    subpart: str
    marks_awarded: float
    max_marks: float
    breakdown: list[MarkBreakdown] = field(default_factory=list)
    confidence: str = "medium"
    method: str = "unknown"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def round_to_half(value: float) -> float:
    """Round to nearest 0.5 — standard exam rounding."""
    return round(value * 2) / 2


# ---------------------------------------------------------------------------
# Single-answer grader
# ---------------------------------------------------------------------------

_GRADE_PROMPT_TEMPLATE = """You are a strict but fair exam marker.

QUESTION: {question_key}
MAXIMUM MARKS: {max_marks}

MARKING SCHEME:
{synoptic_content}

TOPICS NOT RELEVANT TO THIS QUESTION (do NOT award marks for these):
{excluded_topics}

STUDENT ANSWER:
{answer_text}

TASK:
Grade the student's answer strictly against the marking scheme points only.
Do NOT award marks for content that belongs to other questions (bleed-through).
If a point is partially met, award proportional marks.

OUTPUT (JSON array only, no other text):
[
  {{
    "point": "marking point description",
    "allocated_marks": <number>,
    "marks_awarded": <number>,
    "explanation": "brief reason"
  }}
]
JSON array:"""


def _grade_single(
    answer_text: str,
    entry: SynopticEntry,
    excluded_topics: list[str],
    llm_client: LLMClient,
) -> Optional[list[MarkBreakdown]]:
    """Ask the LLM to grade one answer segment."""
    excluded = ", ".join(t for t in excluded_topics if t != entry.key)
    prompt = _GRADE_PROMPT_TEMPLATE.format(
        question_key=entry.key,
        max_marks=entry.marks,
        synoptic_content=entry.content[:1500],
        excluded_topics=excluded or "none",
        answer_text=answer_text[:2000],
    )

    result = llm_client.evaluate(prompt)
    if not isinstance(result, list):
        return None

    breakdowns: list[MarkBreakdown] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        try:
            breakdowns.append(
                MarkBreakdown(
                    point=str(item.get("point", "")),
                    allocated_marks=float(item.get("allocated_marks", 0)),
                    marks_awarded=float(item.get("marks_awarded", 0)),
                    explanation=str(item.get("explanation", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return breakdowns


# ---------------------------------------------------------------------------
# Batch grader
# ---------------------------------------------------------------------------


def grade_all_answers(
    answers_df: pd.DataFrame,
    synoptic_map: SynopticMap,
    llm_client: LLMClient,
) -> pd.DataFrame:
    """
    Grade every row in *answers_df* against *synoptic_map*.

    Returns a DataFrame with columns:
      question, subpart, marks_awarded, max_marks, breakdown, confidence, method.
    """
    all_topics = get_all_topics(synoptic_map)
    results: list[dict] = []

    for _, row in answers_df.iterrows():
        question = row["question"]
        subpart = row["subpart"]
        answer_text = row["answer_text"]

        logger.info("Grading %s.%s …", question, subpart)

        entry = find_entry(question, subpart, synoptic_map)
        if entry is None:
            logger.warning("No synoptic entry for %s.%s — skipping.", question, subpart)
            continue

        breakdowns = _grade_single(answer_text, entry, all_topics, llm_client)
        if breakdowns is None:
            logger.error("LLM grading returned nothing for %s.%s", question, subpart)
            continue

        awarded = round_to_half(
            min(sum(b.marks_awarded for b in breakdowns), entry.marks)
        )

        results.append(
            {
                "question": question,
                "subpart": subpart,
                "marks_awarded": awarded,
                "max_marks": entry.marks,
                "breakdown": [b.__dict__ for b in breakdowns],
                "confidence": row.get("confidence", "medium"),
                "method": row.get("detection_method", "unknown"),
            }
        )
        logger.info("  → %s / %s marks", awarded, entry.marks)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Missing-question filler
# ---------------------------------------------------------------------------


def fill_missing_questions(
    graded_df: pd.DataFrame,
    synoptic_map: SynopticMap,
) -> pd.DataFrame:
    """
    Append zero-mark rows for any synoptic question not present in *graded_df*.
    """
    graded_questions = set(graded_df["question"].tolist()) if not graded_df.empty else set()
    expected_questions = {key.split(".")[0] for key in synoptic_map}

    extra_rows = []
    for q in expected_questions - graded_questions:
        q_keys = [k for k in synoptic_map if k.startswith(q + ".")]
        q_max = sum(synoptic_map[k].marks for k in q_keys)
        extra_rows.append(
            {
                "question": q,
                "subpart": "-",
                "marks_awarded": 0.0,
                "max_marks": q_max,
                "breakdown": [
                    {
                        "point": "Not attempted",
                        "allocated_marks": q_max,
                        "marks_awarded": 0.0,
                        "explanation": "No answer found in the paper.",
                    }
                ],
                "confidence": "certain",
                "method": "not_attempted",
            }
        )

    if extra_rows:
        graded_df = pd.concat([graded_df, pd.DataFrame(extra_rows)], ignore_index=True)
    return graded_df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_report(graded_df: pd.DataFrame, synoptic_map: SynopticMap, paper_name: str) -> str:
    """Generate a human-readable plain-text grading report."""
    total_awarded = graded_df["marks_awarded"].sum()
    total_max = sum(e.marks for e in synoptic_map.values())
    percentage = (total_awarded / total_max * 100) if total_max > 0 else 0.0

    lines = [
        "╔══════════════════════════════════════════╗",
        "║          EXAM GRADING REPORT             ║",
        "╚══════════════════════════════════════════╝",
        f"\nPaper: {paper_name}",
        f"Score: {total_awarded:.1f} / {total_max:.1f}",
        f"Grade: {percentage:.1f}%",
        "\n── QUESTION BREAKDOWN ─────────────────────",
    ]

    for key in sorted(synoptic_map.keys()):
        q, sub = key.split(".", 1)
        rows = graded_df[graded_df["question"] == q]
        if sub != "-":
            rows = rows[rows["subpart"] == sub]

        max_m = synoptic_map[key].marks
        if rows.empty or rows.iloc[0]["method"] == "not_attempted":
            lines.append(f"  {key:12}  0.0 / {max_m:.1f}  ❌  Not attempted")
        else:
            awarded = rows.iloc[0]["marks_awarded"]
            lines.append(f"  {key:12}  {awarded:.1f} / {max_m:.1f}")

    return "\n".join(lines)
