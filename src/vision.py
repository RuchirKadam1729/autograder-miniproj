"""
Florence-2 vision model wrapper.
Handles model loading and single-task inference (OCR, captioning, etc.).
"""

import logging
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .config import VisionConfig

logger = logging.getLogger(__name__)

# Florence-2 task tokens
TASK_OCR = "<OCR>"
TASK_CAPTION = "<CAPTION>"
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_OD = "<OD>"


class VisionModel:
    """Thin wrapper around a Florence-2 checkpoint."""

    def __init__(self, cfg: VisionConfig) -> None:
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForCausalLM | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download / load the model weights into memory."""
        logger.info("Loading Florence-2 from %s on %s …", self._cfg.model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(
            self._cfg.model_id, trust_remote_code=self._cfg.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._cfg.model_id,
            trust_remote_code=self._cfg.trust_remote_code,
            torch_dtype=getattr(torch, self._cfg.torch_dtype),
        ).to(self._device)
        self._model.eval()
        logger.info("Florence-2 ready.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_task(self, image: Image.Image, task: str = TASK_OCR) -> str:
        """
        Run a Florence-2 vision task on *image*.

        Parameters
        ----------
        image:  PIL Image (RGB preferred).
        task:   Florence task token, e.g. ``TASK_OCR``.

        Returns
        -------
        str — the model's text output for the task.
        """
        if not self.is_loaded:
            raise RuntimeError("VisionModel.load() must be called before inference.")

        inputs = self._processor(text=task, images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        raw_text = self._processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed: dict[str, Any] = self._processor.post_process_generation(
            raw_text,
            task=task,
            image_size=(image.width, image.height),
        )
        result = parsed.get(task, "")
        return result if isinstance(result, str) else str(result)

    def ocr(self, image: Image.Image) -> str:
        """Convenience wrapper: run OCR task."""
        return self.run_task(image, TASK_OCR)

    def caption(self, image: Image.Image, detailed: bool = False) -> str:
        """Convenience wrapper: run caption task."""
        task = TASK_DETAILED_CAPTION if detailed else TASK_CAPTION
        return self.run_task(image, task)
