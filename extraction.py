"""
Answer extraction pipeline.
Converts student answer PDF pages → structured DataFrame of (question, subpart, text).
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image

from .llm_client import LLMClient
from .synoptic import SynopticMap
from .vision import VisionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PageData:
    """Raw data for a single PDF page."""
    page_number: int
    image: Image.Image
    ocr_text: str = ""
    confidence: str = "low"


@dataclass
class AnswerSegment:
    """A single extracted answer segment."""
    question: str
    subpart: str
    answer_text: str
    confidence: str = "medium"
    detection_method: str = "unknown"


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Render every page of a PDF to a PIL Image at the given DPI."""
    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def ocr_page(image: Image.Image, vision_model: VisionModel) -> str:
    """Run Florence-2 OCR on a single page image."""
    try:
        return vision_model.ocr(image)
    except Exception as exc:
        logger.warning("OCR failed for page: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# LLM-based structure detection
# ---------------------------------------------------------------------------


_STRUCTURE_PROMPT_TEMPLATE = """You are analysing OCR text from a student exam paper.
Identify which question and subpart each paragraph belongs to.

SYNOPTIC QUESTIONS AVAILABLE: {questions}

OCR TEXT:
{ocr_text}

Output ONLY a JSON array, one object per identified segment:
[
  {{
    "question": "Q1",
    "subpart": "a",
    "answer_text": "...",
    "confidence": "high|medium|low"
  }}
]
JSON array only:"""


def detect_structure(
    ocr_text: str,
    synoptic_map: SynopticMap,
    llm_client: LLMClient,
    page_number: int,
) -> list[AnswerSegment]:
    """Ask the LLM to split raw OCR text into labelled answer segments."""
    questions = ", ".join(synoptic_map.keys())
    prompt = _STRUCTURE_PROMPT_TEMPLATE.format(
        questions=questions,
        ocr_text=ocr_text[:3000],
    )
    result = llm_client.evaluate(prompt)
    if not isinstance(result, list):
        return []

    segments: list[AnswerSegment] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        try:
            segments.append(
                AnswerSegment(
                    question=str(item.get("question", "")).strip(),
                    subpart=str(item.get("subpart", "-")).strip().lower(),
                    answer_text=str(item.get("answer_text", "")).strip(),
                    confidence=str(item.get("confidence", "medium")),
                    detection_method=f"llm_page_{page_number}",
                )
            )
        except (KeyError, TypeError) as exc:
            logger.debug("Skipping malformed segment: %s", exc)
    return segments


# ---------------------------------------------------------------------------
# Validation / deduplication
# ---------------------------------------------------------------------------


def validate_and_merge(
    segments: list[AnswerSegment],
    synoptic_map: SynopticMap,
) -> list[AnswerSegment]:
    """
    Remove segments whose question/subpart is not in the synoptic map,
    then merge duplicate entries by concatenating their text.
    """
    merged: dict[str, AnswerSegment] = {}
    for seg in segments:
        key = f"{seg.question}.{seg.subpart}"
        # Accept if exact key or dash key exists
        if key not in synoptic_map and f"{seg.question}.-" not in synoptic_map:
            logger.debug("Dropping unknown key: %s", key)
            continue
        if key in merged:
            merged[key].answer_text += "\n" + seg.answer_text
        else:
            merged[key] = seg
    return list(merged.values())


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


def extract_answers(
    pdf_path: str | Path,
    synoptic_map: SynopticMap,
    vision_model: VisionModel,
    llm_client: LLMClient,
) -> tuple[pd.DataFrame, list[PageData]]:
    """
    Full extraction pipeline for a single student paper.

    Returns
    -------
    answers_df : DataFrame with columns question, subpart, answer_text,
                 confidence, detection_method.
    pages_data : List of raw PageData objects (for debugging / re-runs).
    """
    images = pdf_to_images(pdf_path)
    logger.info("Loaded %d pages from %s", len(images), pdf_path)

    pages_data: list[PageData] = []
    all_segments: list[AnswerSegment] = []

    for page_num, image in enumerate(images, start=1):
        logger.info("  OCR page %d/%d …", page_num, len(images))
        ocr_text = ocr_page(image, vision_model)
        page = PageData(page_number=page_num, image=image, ocr_text=ocr_text)
        pages_data.append(page)

        if not ocr_text.strip():
            logger.warning("  Page %d produced no OCR text, skipping.", page_num)
            continue

        segments = detect_structure(ocr_text, synoptic_map, llm_client, page_num)
        logger.info("  Page %d → %d segments detected.", page_num, len(segments))
        all_segments.extend(segments)

    validated = validate_and_merge(all_segments, synoptic_map)
    if not validated:
        return pd.DataFrame(), pages_data

    df = pd.DataFrame(
        [
            {
                "question": s.question,
                "subpart": s.subpart,
                "answer_text": s.answer_text,
                "confidence": s.confidence,
                "detection_method": s.detection_method,
            }
            for s in validated
        ]
    )
    return df, pages_data
