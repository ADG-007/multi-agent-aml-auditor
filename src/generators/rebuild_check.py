"""Same-seed rebuild check (Amendment 4, V3): two builds give the same files.

    python -m src.generators.rebuild_check

Builds the dataset twice with the committed seed, each into its own temporary folder
(AML_DATA_DIR), and compares the SHA-256 of every file the generator writes that git tracks.
Both builds are also compared with the committed data/. The ChromaDB index and the document
images are not tracked, so they are not compared (S15). Exits non-zero on any difference.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.generators import config as cfg

#: Generator outputs that git tracks, relative to the data folder.
TRACKED_OUTPUTS: tuple[str, ...] = (
    "customers.json",
    "accounts.json",
    "transactions.json",
    "counterparties.json",
    "kyc_profiles.json",
    "adverse_media.json",
    "graph.graphml",
    "documents/manifest.json",
    "rulebook/clauses.json",
    "rulebook/README.md",
    "_ground_truth/evidence_ledger.json",
    "_ground_truth/fraud_rings.json",
)


def build(data_dir: Path, seed: int = cfg.SEED, embeddings: bool = False) -> None:
    """Run the full generator into `data_dir`, by default without the embedding index."""
    env = {**os.environ, "AML_DATA_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8"}
    skip = [] if embeddings else ["--skip-embeddings"]
    subprocess.run(
        [sys.executable, "-m", "src.generators", "--seed", str(seed), *skip],
        cwd=cfg.PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def digests(data_dir: Path) -> dict[str, str]:
    """SHA-256 of each tracked output, with line endings normalised as git stores them."""
    out = {}
    for name in TRACKED_OUTPUTS:
        content = (data_dir / name).read_bytes().replace(b"\r\n", b"\n")
        out[name] = hashlib.sha256(content).hexdigest()
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="aml-rebuild-") as tmp:
        runs = []
        for label in ("a", "b"):
            target = Path(tmp) / label / "data"
            print(f"Building copy {label} (seed {cfg.SEED}) into {target} ...", flush=True)
            build(target)
            runs.append(digests(target))
    committed = digests(cfg.PROJECT_ROOT / "data")

    failures = 0
    print(f"\n  {'File':<36} {'SHA-256 (first 16)':<18}  build a = b   = committed")
    for name in TRACKED_OUTPUTS:
        same = runs[0][name] == runs[1][name]
        matches = runs[0][name] == committed[name]
        failures += not (same and matches)
        print(
            f"  {name:<36} {runs[0][name][:16]:<18}  {'yes' if same else 'NO':<11} "
            f"{'yes' if matches else 'NO'}"
        )
    if failures:
        print(f"\nFAILED: {failures} file(s) differ.")
        return 1
    print(f"\nAll {len(TRACKED_OUTPUTS)} files are identical across both builds and the committed data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
