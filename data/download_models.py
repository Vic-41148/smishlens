#!/usr/bin/env python3
"""Download the public model weights the roadmap needs.

Weights are large and git-ignored (`models/`), so this script is the
reproducible artefact, not the payload. Nothing here is a model we trained.

Usage:
    python data/download_models.py           # fetch missing weights
    python data/download_models.py --force   # refetch everything
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# repo_id -> local directory name. Sizes are the last verified download.
HF_MODELS = {
    "jhu-clsp/mmBERT-base": "mmBERT-base",   # 1.2 GB, MIT, main encoder
    "jhu-clsp/mmBERT-small": "mmBERT-small",  # 0.6 GB, MIT, CPU fallback
    "telecomsxchange/OpenTextShield": "OpenTextShield",  # external baseline
}

OLLAMA_MODEL = "qwen3:4b-instruct"  # ~2.5 GB, Apache-2.0, optional explainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch existing models")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    for repo, name in HF_MODELS.items():
        dest = MODELS / name
        weights = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
        if weights and not args.force:
            print(f"HF {repo}: present ({dest})")
            continue
        print(f"HF {repo} -> {dest}")
        snapshot_download(repo_id=repo, local_dir=dest)

    if shutil.which("ollama") is None:
        print(f"ollama not installed; skipping {OLLAMA_MODEL}")
    else:
        print(f"ollama pull {OLLAMA_MODEL}")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=False)

    print("\nDone. Models live under models/ and are git-ignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
