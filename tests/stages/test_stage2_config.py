"""Stage 2's config: the decided numbers, and no rulebook-defined number copied in."""

from pathlib import Path

from src.stages.stage2_transaction_pattern import config


def test_decided_values():
    assert config.FRAME_DAYS == 90 and config.WINDOW_DAYS == 30 and config.RECENT_DAYS == 30
    assert dict(config.STRUCTURING_MAP) == {
        0: 0.0,
        1: 0.15,
        2: 0.30,
        3: 0.40,
        4: 0.50,
        5: 0.65,
        6: 0.80,
        7: 0.90,
        8: 1.0,
    }
    assert config.FAST_HOLD_HOURS == 12 and config.RUN_CLOSE_SHARE == 0.70
    assert dict(config.GATE) == {3: 0.6, 4: 0.8, 5: 1.0}
    assert (config.SHARE_LOW, config.SHARE_HIGH) == (0.25, 0.60)
    assert (config.NOVELTY_BASE, config.NOVELTY_BOOST) == (0.7, 0.3)
    assert config.MAX_TRAIL_STEPS == 8


def test_structuring_map_never_falls():
    scores = [score for _, score in config.STRUCTURING_MAP]
    assert scores == sorted(scores)


def test_no_rulebook_number_is_held_in_config():
    """The band and the mule floor are read from STRUCT-001 and MULE-002 at run time."""
    names = [n for n in vars(config) if n.isupper()]
    assert not [n for n in names if any(w in n for w in ("THRESHOLD", "FLOOR", "BAND", "CTR"))]
    source = Path(config.__file__).read_text(encoding="utf-8")
    assert "1_000_000" not in source and "1000000" not in source


def test_stage2_never_imports_the_llm_layer():
    for path in Path("src/stages/stage2_transaction_pattern").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "src.llm" not in source and "from src import llm" not in source, path
