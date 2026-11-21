#!/usr/bin/env python3
"""Download every public SMS/smishing source used by the SMS-2026 roadmap.

Raw data is git-ignored: several sources are PII-bearing and some licences do
not allow redistribution. This script is the reproducible artefact.

Usage:
    python data/download_data.py              # fetch missing sources
    python data/download_data.py --force      # refetch everything
    python data/download_data.py --only uci   # fetch only the UCI anchor
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

UCI_ZIP = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
IMC_REPO = "https://github.com/reportsmishing/Smishing-Dataset-IMC25.git"
EXAIS_REPO = "https://github.com/AbayomiAlli/SMS-Spam-Dataset.git"
WSPR_REPO = "https://github.com/wspr-ncsu/sms-phishing.git"
MENDELEY_FILES = "https://data.mendeley.com/api/datasets/f45bkkt8pr/files"
# Dataset I: 1,062 smishing samples, Timko & Rahman (CODASPY 2024).
# https://doi.org/10.1145/3626232.3653282
SMISHTANK_CSV = "https://smishtank.com/analysisdataset.csv"

HF_DATASETS = {
    "MOZNLP/MOZ-Smishing": "moz",
    "CloveAI/india-spam-sms": "hf/india-spam-sms",
    "Ngadou/Spam_SMS": "hf/ngadou-spam-sms",
    "daviiiidcpp1/sms-spam-combined": "hf/sms-spam-combined",
    "electricsheepafrica/africa-smishing-sms-phishing": "hf/africa-smishing",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    urllib.request.urlretrieve(url, dest)


def clone(repo: str, dest: Path) -> None:
    print(f"  git clone {repo}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo, str(dest)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def unzip(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)


def exists(path: Path, force: bool) -> bool:
    return path.exists() and not force


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch existing sources")
    ap.add_argument(
        "--only",
        choices=["uci"],
        default=None,
        help="fetch a single source only (uci: the Tier 1 anchor)",
    )
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. UCI SMS Spam Collection
    uci_file = RAW / "uci" / "SMSSpamCollection"
    if not exists(uci_file, args.force):
        print("[1/9] UCI SMS Spam Collection")
        zpath = RAW / "uci_sms_spam_collection.zip"
        fetch(UCI_ZIP, zpath)
        unzip(zpath, RAW / "uci")
        zpath.unlink(missing_ok=True)
    else:
        print("[1/9] UCI: present")

    if args.only == "uci":
        print("\nDone (--only uci).")
        return 0

    # 2. IMC'25 smishing
    imc_file = RAW / "imc25" / "dataset" / "final_dataset_output.csv"
    if not exists(imc_file, args.force):
        print("[2/9] IMC'25 smishing")
        shutil.rmtree(RAW / "imc25", ignore_errors=True)
        clone(IMC_REPO, RAW / "imc25")
    else:
        print("[2/9] IMC'25: present")

    # 3. Mishra & Soni (Mendeley data f45bkkt8pr)
    mishra = RAW / "mishra_soni" / "Dataset_5971.csv"
    if not exists(mishra, args.force):
        print("[3/9] Mishra & Soni (Mendeley)")
        files = json.load(urllib.request.urlopen(MENDELEY_FILES, timeout=60))
        target = next(f for f in files if f["filename"].startswith("Dataset_"))
        zpath = RAW / "mishra_soni" / target["filename"]
        fetch(target["content_details"]["download_url"], zpath)
        unzip(zpath, RAW / "mishra_soni")
        zpath.unlink(missing_ok=True)
    else:
        print("[3/9] Mishra & Soni: present")

    # 4. ExAIS
    exais = RAW / "exais" / "extracted"
    if not exists(exais, args.force):
        print("[4/9] ExAIS SMS Spam")
        clone(EXAIS_REPO, RAW / "exais")
        zpath = next((RAW / "exais").glob("*.zip"))
        unzip(zpath, exais)
    else:
        print("[4/9] ExAIS: present")

    # 5. WSPR Lab
    wspr = RAW / "wspr" / "phishing_messages.csv"
    if not exists(wspr, args.force):
        print("[5/9] WSPR Lab SMS phishing")
        shutil.rmtree(RAW / "wspr", ignore_errors=True)
        clone(WSPR_REPO, RAW / "wspr")
    else:
        print("[5/9] WSPR: present")

    # 6. Hugging Face datasets (including MOZ-Smishing)
    print("[6/9] Hugging Face datasets")
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    for repo, rel in HF_DATASETS.items():
        dest = RAW / rel
        if exists(dest / "README.md", args.force):
            print(f"  HF {repo}: present")
            continue
        print(f"  HF {repo}")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest)

    # 7. Smishtank Dataset I (full CSV, 1,062 smishing samples)
    st = RAW / "smishtank" / "analysisdataset.csv"
    if not exists(st, args.force):
        print("[7/9] Smishtank Dataset I")
        fetch(SMISHTANK_CSV, st)
    else:
        print("[7/9] Smishtank: present")

    # 8. NUS SMS Corpus
    print("[8/9] NUS SMS Corpus: NOT DOWNLOADABLE")
    print("      WING-NUS/nus-sms-corpus now returns 404 (deleted/private).")
    print("      UCI already contributes 3,375 NUS ham messages.")

    # 9. COVA / COVA-X
    print("[9/9] COVA / COVA-X: ACCESS BY REQUEST ONLY")
    print("      Papers: arXiv 2604.11752 (COVA), 2606.06879 (COVA-X).")
    print("      scamlingua.org releases both for non-commercial research on request;")
    print("      no public download. Email contact@scamlingua.org before Tier 3.")

    print("\nDone. Run `python data/audit_sources.py` to verify counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
