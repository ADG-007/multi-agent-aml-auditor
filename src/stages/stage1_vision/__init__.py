"""Stage 1 - ID document check (Tesseract OCR + field diff). Never scored; pauses only."""

from src.stages.stage1_vision.agent import run


__all__ = ["run"]
