"""Stage 0 - Triage & Routing. Deterministic; never calls a model or runs graph algorithms."""

from src.stages.stage0_triage.triage import TriageContext, first_alert, triage


__all__ = ["TriageContext", "first_alert", "triage"]
