"""
Centralized configuration for the Exam Grader application.
All environment variables and constants live here.
"""

import os
import logging
from dataclasses import dataclass, field


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class LLMConfig:
    """LLM (Ollama) connection settings."""
    url: str = field(
        default_factory=lambda: os.getenv("LLM_URL", "http://localhost:11434/api/generate")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    )
    max_tokens: int = 500
    timeout_seconds: int = 120
    stream: bool = True


@dataclass
class VisionConfig:
    """Florence-2 vision model settings."""
    model_id: str = "microsoft/Florence-2-large-ft"
    torch_dtype: str = "float32"
    trust_remote_code: bool = True


@dataclass
class AppConfig:
    """Top-level application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "outputs")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    max_synoptic_chars: int = 1500   # chars sent to LLM per question block
    half_mark_rounding: bool = True  # round awards to nearest 0.5


# Module-level singleton — import and use `config` everywhere
config = AppConfig()
