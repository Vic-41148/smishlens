#!/usr/bin/env python3
"""G3: can mmBERT fine-tune inside 6 GB, and how fast is it?

One-hour smoke test from roadmap §7.8. Trains on UCI **train** rows only — never
the pool, the gold holdout or the UCI test split — for 2000 rows and 1 epoch,
with the embedding matrix frozen, max_len 128 and bf16 autocast. Reports peak
VRAM, seconds per step, tokens per second, and CPU inference time for 100
messages. Writes ``reports/results/g3_smoke.json``.

Usage:
    python data/g3_smoke.py --model models/mmBERT-small
    python data/g3_smoke.py --model models/mmBERT-base --rows 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smishlens.normalise import normalise  # noqa: E402

TRAIN = ROOT / "data" / "processed" / "uci" / "train.csv"
OUT = ROOT / "reports" / "results" / "g3_smoke.json"
SEED = 42


def prep(texts):
    """Unicode fixes only: transformers keep case and natural text."""
    return [normalise(t, mask=False, lower=False) for t in texts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "mmBERT-small"))
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        print("no CUDA device; G3 needs a GPU")
        return 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    frame = pd.read_csv(TRAIN)
    frame = frame.sample(n=min(args.rows, len(frame)), random_state=SEED).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    model.get_input_embeddings().weight.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    steps_per_epoch = int(np.ceil(len(frame) / args.bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, 0.06 * steps_per_epoch)))

    texts = prep(frame["text"].tolist())
    y = frame["y"].to_numpy()
    tokens_seen = 0
    step_times: list[float] = []

    torch.cuda.reset_peak_memory_stats()
    model.train()
    t_start = time.perf_counter()
    for k in range(steps_per_epoch):
        batch = slice(k * args.bs, (k + 1) * args.bs)
        enc = tok(
            texts[batch],
            truncation=True,
            max_length=args.max_len,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        tokens_seen += int(enc["attention_mask"].sum())
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            logits = model(**enc).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.tensor(y[batch], device=dev)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t_start

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    median_step = float(np.median(step_times))

    # CPU inference speed on 100 messages, from the same freshly trained weights.
    cpu_texts = texts[:100]
    cpu_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    cpu_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(0, len(cpu_texts), 16):
            enc = tok(
                cpu_texts[i : i + 16],
                truncation=True,
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            )
            cpu_model(**enc)
        cpu_seconds = time.perf_counter() - t0

    result = {
        "model": str(Path(args.model).name),
        "rows": int(len(frame)),
        "epochs": args.epochs,
        "batch_size": args.bs,
        "max_len": args.max_len,
        "dtype": str(dtype).replace("torch.", ""),
        "embeddings_frozen": True,
        "params_total": int(n_params),
        "params_trainable": int(n_trainable),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "steps": steps_per_epoch,
        "train_seconds": round(train_seconds, 2),
        "median_s_per_step": round(median_step, 3),
        "tokens_per_second": int(tokens_seen / train_seconds),
        "peak_vram_alloc_gib": round(peak_alloc, 2),
        "peak_vram_reserved_gib": round(peak_reserved, 2),
        "cpu_inference_100_messages_seconds": round(cpu_seconds, 2),
        "cpu_ms_per_message": round(1000 * cpu_seconds / len(cpu_texts), 2),
        "source": "data/processed/uci/train.csv (UCI train only; test never read)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for key, value in result.items():
        print(f"{key:<38} {value}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
