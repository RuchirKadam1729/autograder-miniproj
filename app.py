"""
Gradio frontend for the AI Exam Grader.
Entry point for Hugging Face Spaces (file: app.py).
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

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Lazy-loaded globals so Spaces doesn't OOM on cold start
_vision_model: VisionModel | None = None
_llm_client: LLMClient | None = None
_app_config = AppConfig()


def _get_vision() -> VisionModel:
    global _vision_model
    if _vision_model is None:
        _vision_model = VisionModel(VisionConfig())
        _vision_model.load()
    return _vision_model


def _get_llm(llm_url: str, llm_model: str) -> LLMClient:
    """Return a (possibly cached) LLM client for the given endpoint."""
    cfg = LLMConfig(url=llm_url, model=llm_model)
    return LLMClient(cfg)


# ---------------------------------------------------------------------------
# Gradio callback handlers
# ---------------------------------------------------------------------------


def check_llm_connection(llm_url: str, llm_model: str) -> str:
    """Ping the LLM and report back to the UI."""
    if not llm_url.strip():
        return "⚠️ Please enter an LLM URL first."
    client = _get_llm(llm_url.strip(), llm_model.strip())
    if client.is_available():
        return f"✅ Connected to **{llm_model}** at `{llm_url}`"
    return f"❌ Could not reach LLM at `{llm_url}`. Check the URL and that Ollama is running."


def grade_paper(
    paper_file,
    synoptic_file,
    llm_url: str,
    llm_model: str,
) -> tuple[str, pd.DataFrame | None, str]:
    """
    Main grading callback.

    Returns (status_message, results_dataframe, report_text).
    """
    if paper_file is None:
        return "⚠️ Upload a student answer PDF.", None, ""
    if synoptic_file is None:
        return "⚠️ Upload a marking scheme (Excel/CSV).", None, ""
    if not llm_url.strip():
        return "⚠️ Enter the Ollama LLM URL.", None, ""

    try:
        syn_df = load_synoptic(synoptic_file.name)
        synoptic_map = build_synoptic_map(syn_df)
    except Exception as exc:
        return f"❌ Failed to load synoptic: {exc}", None, ""

    try:
        vision = _get_vision()
        llm = _get_llm(llm_url.strip(), llm_model.strip())
        pipeline = GradingPipeline(_app_config, vision, llm)
    except Exception as exc:
        return f"❌ Model initialisation failed: {exc}", None, ""

    with tempfile.TemporaryDirectory() as tmp:
        result = pipeline.run(
            paper_path=paper_file.name,
            synoptic_map=synoptic_map,
            output_dir=tmp,
        )

    if result.status != "success":
        return f"❌ Pipeline failed: {result.reason}", None, ""

    status = (
        f"✅ Graded **{result.paper_name}**  →  "
        f"**{result.total_score:.1f} / {result.total_max:.1f}** "
        f"({result.percentage:.1f}%)"
    )

    # Pretty-print the breakdown table
    display_df = result.graded_df[["question", "subpart", "marks_awarded", "max_marks"]].copy()
    display_df.columns = ["Question", "Subpart", "Awarded", "Max"]

    return status, display_df, result.report_text


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 📝 AI Exam Grader

Automated marking of handwritten student answer PDFs using **Florence-2** (vision OCR)
and an **Ollama LLM** (e.g. Qwen 2.5) for rubric-based grading.

### How to use
1. Set the **LLM URL** to your Ollama endpoint (ngrok or local).
2. Upload the **student answer PDF** and the **marking scheme** (`.xlsx` or `.csv`).
3. Click **Grade Paper** and wait for the results.
"""

SYNOPTIC_COLUMNS_INFO = """
**Marking scheme columns required:**
| column | description |
|---|---|
| `question` | e.g. `Q1`, `Q2` |
| `subpart` | e.g. `a`, `b`, `-` (for undivided questions) |
| `max_marks` | numeric |
| `content` | the full marking scheme text for this part |
"""

with gr.Blocks(theme=gr.themes.Soft(), title="AI Exam Grader") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            llm_url_box = gr.Textbox(
                label="Ollama LLM URL",
                placeholder="https://your-ngrok-url.ngrok-free.app/api/generate",
                value=os.getenv("LLM_URL", ""),
            )
            llm_model_box = gr.Textbox(
                label="Model name",
                value=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
            )
            test_btn = gr.Button("🔌 Test LLM Connection", variant="secondary")
            connection_status = gr.Markdown("")

            test_btn.click(
                fn=check_llm_connection,
                inputs=[llm_url_box, llm_model_box],
                outputs=connection_status,
            )

            gr.Markdown("---")
            gr.Markdown("### 📂 Upload Files")
            paper_upload = gr.File(
                label="Student Answer PDF",
                file_types=[".pdf"],
            )
            synoptic_upload = gr.File(
                label="Marking Scheme (Excel / CSV)",
                file_types=[".xlsx", ".xls", ".csv"],
            )
            gr.Markdown(SYNOPTIC_COLUMNS_INFO)
            grade_btn = gr.Button("🎓 Grade Paper", variant="primary", size="lg")

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
        inputs=[paper_upload, synoptic_upload, llm_url_box, llm_model_box],
        outputs=[status_box, results_table, report_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
