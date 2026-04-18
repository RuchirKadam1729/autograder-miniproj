"""
Synoptic (marking scheme) parsing utilities.
Converts raw Excel/CSV synoptic data into a structured lookup map.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SynopticEntry:
    """Single question/subpart entry from the marking scheme."""
    question: str
    subpart: str
    marks: float
    content: str
    key: str = field(init=False)

    def __post_init__(self) -> None:
        self.key = f"{self.question}.{self.subpart}"


SynopticMap = dict[str, SynopticEntry]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_synoptic(path: str | Path) -> pd.DataFrame:
    """
    Load a synoptic file (.xlsx or .csv) and return a normalised DataFrame
    with columns: ``question``, ``subpart``, ``max_marks``, ``content``.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported synoptic format: {path.suffix}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"question", "subpart", "max_marks", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Synoptic file is missing columns: {missing}")

    df["question"] = df["question"].astype(str).str.strip()
    df["subpart"] = df["subpart"].astype(str).str.strip().str.lower()
    df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0)
    df["content"] = df["content"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Map building
# ---------------------------------------------------------------------------


def build_synoptic_map(df: pd.DataFrame) -> SynopticMap:
    """Convert a synoptic DataFrame into a lookup dict keyed by ``Q.subpart``."""
    result: SynopticMap = {}
    for _, row in df.iterrows():
        entry = SynopticEntry(
            question=row["question"],
            subpart=row["subpart"],
            marks=float(row["max_marks"]),
            content=str(row["content"]),
        )
        result[entry.key] = entry
    return result


def get_all_topics(synoptic_map: SynopticMap) -> list[str]:
    """Return a flat list of all question key strings."""
    return list(synoptic_map.keys())


def find_entry(
    question: str, subpart: str, synoptic_map: SynopticMap
) -> Optional[SynopticEntry]:
    """Look up an entry; tries the exact key then a dash fallback."""
    exact = f"{question}.{subpart}"
    if exact in synoptic_map:
        return synoptic_map[exact]
    dash = f"{question}.-"
    if dash in synoptic_map:
        return synoptic_map[dash]
    return None


# ---------------------------------------------------------------------------
# LLM-assisted marks extraction
# ---------------------------------------------------------------------------

_MARKS_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*marks?\b", re.IGNORECASE)


def extract_max_marks(
    content: str,
    question_key: str,
    llm_client: LLMClient,
    max_content_chars: int = 1500,
) -> float:
    """
    Use the LLM to reliably extract the maximum marks for a question.
    Falls back to regex if the LLM call fails.
    """
    prompt = (
        f"You are analysing an exam marking scheme.\n\n"
        f"QUESTION: {question_key}\n\n"
        f"MARKING SCHEME CONTENT:\n{content[:max_content_chars]}\n\n"
        "TASK: What is the TOTAL maximum marks?\n"
        "Rules:\n"
        "- Look for patterns like '10 marks', '[10]', '(10 marks)'.\n"
        "- If there are sub-parts, ADD them for the total.\n\n"
        'OUTPUT (JSON only): {"max_marks": <number>}'
    )

    result = llm_client.evaluate(prompt)
    if isinstance(result, dict) and "max_marks" in result:
        try:
            return float(result["max_marks"])
        except (TypeError, ValueError):
            pass

    # Regex fallback
    matches = _MARKS_PATTERN.findall(content)
    if matches:
        return float(matches[0])
    return 0.0
