"""
Gradio frontend for the AI Exam Grader.
Entry point for Hugging Face Spaces (file: app.py).
LLM powered by Groq API — no server to keep running.
"""

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from src import (
    AppConfig,
    GradingPipeline,
    LLMClient,
    LLMConfig,
    VisionConfig,
    VisionModel,
    build_synoptic_map,
    load_synoptic,
    setup_logging,
)
from src.llm_client import GROQ_MODELS

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_vision_model: VisionModel | None = None
_app_config = AppConfig()


def _get_vision() -> VisionModel:
    global _vision_model
    if _vision_model is None:
        _vision_model = VisionModel(VisionConfig())
        _vision_model.load()
    return _vision_model


def _get_llm(api_key: str, model: str) -> LLMClient:
    cfg = LLMConfig(api_key=api_key.strip(), model=model.strip())
    return LLMClient(cfg)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def check_groq_connection(api_key: str, model: str) -> str:
    if not api_key.strip():
        return "⚠️ Paste your Groq API key first."
    try:
        client = _get_llm(api_key, model)
    except Exception as exc:
        return f"❌ {exc}"
    if client.is_available():
        return f"✅ Groq connected — model **{model}** is ready."
    return "❌ Groq responded but the test failed. Check your key."


def grade_paper(paper_file, synoptic_file, api_key: str, model: str):
    if paper_file is None:
        return "⚠️ Upload a student answer PDF.", None, ""
    if synoptic_file is None:
        return "⚠️ Upload a marking scheme (Excel/CSV).", None, ""
    if not api_key.strip():
        return "⚠️ Enter your Groq API key.", None, ""

    try:
        syn_df = load_synoptic(synoptic_file.name)
        synoptic_map = build_synoptic_map(syn_df)
    except Exception as exc:
        return f"❌ Failed to load synoptic: {exc}", None, ""

    try:
        vision = _get_vision()
        llm = _get_llm(api_key, model)
        pipeline = GradingPipeline(_app_config, vision, llm)
    except Exception as exc:
        return f"❌ Initialisation failed: {exc}", None, ""

    with tempfile.TemporaryDirectory() as tmp:
        result = pipeline.run(
            paper_path=paper_file.name,
            synoptic_map=synoptic_map,
            output_dir=tmp,
        )

    if result.status != "success":
        return f"❌ Pipeline failed: {result.reason}", None, ""

    status = (
        f"✅ **{result.paper_name}** graded →  "
        f"**{result.total_score:.1f} / {result.total_max:.1f}** "
        f"({result.percentage:.1f}%)"
    )
    display_df = result.graded_df[["question", "subpart", "marks_awarded", "max_marks"]].copy()
    display_df.columns = ["Question", "Subpart", "Awarded", "Max"]
    return status, display_df, result.report_text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 📝 AI Exam Grader
Automated marking of handwritten student answer PDFs using  
**Florence-2** (vision OCR) + **Groq LLaMA 3.3** (rubric grading).

No servers to keep alive — just paste your free [Groq API key](https://console.groq.com).
"""

SYNOPTIC_INFO = """
**Marking scheme columns required:**  
`question` · `subpart` · `max_marks` · `content`
"""

with gr.Blocks(theme=gr.themes.Soft(), title="AI Exam Grader") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left panel ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Groq API")
            api_key_box = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
            )
            model_dd = gr.Dropdown(
                label="Model",
                choices=GROQ_MODELS,
                value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            )
            test_btn = gr.Button("🔌 Test Connection", variant="secondary")
            conn_status = gr.Markdown("")
            test_btn.click(check_groq_connection, [api_key_box, model_dd], conn_status)

            gr.Markdown("---")
            gr.Markdown("### 📂 Upload Files")
            paper_upload = gr.File(label="Student Answer PDF", file_types=[".pdf"])
            synoptic_upload = gr.File(
                label="Marking Scheme (Excel / CSV)",
                file_types=[".xlsx", ".xls", ".csv"],
            )
            gr.Markdown(SYNOPTIC_INFO)
            grade_btn = gr.Button("🎓 Grade Paper", variant="primary", size="lg")

        # ── Right panel ──────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            status_box = gr.Markdown("")
            results_table = gr.Dataframe(
                label="Question Breakdown",
                headers=["Question", "Subpart", "Awarded", "Max"],
                interactive=False,
            )
            report_box = gr.Textbox(
                label="Full Report",
                lines=20,
                interactive=False,
                show_copy_button=True,
            )

    grade_btn.click(
        fn=grade_paper,
        inputs=[paper_upload, synoptic_upload, api_key_box, model_dd],
        outputs=[status_box, results_table, report_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)