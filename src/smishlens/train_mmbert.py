"""Fine-tune mmBERT with a frozen embedding matrix so it fits in 6 GB.

STATUS: syntax-checked only until the G3 smoke test runs on the GPU. Treat the
first run as that smoke test: it prints peak VRAM, which is the G3 number.
"""

from __future__ import annotations

import os

import numpy as np

from smishlens.normalise import normalise

MODEL = os.getenv("MMBERT_MODEL", "models/mmBERT-base")


def prep(texts):
    """Encoders keep case and natural text: unicode fixes only, no masks, no lowercasing."""
    return [normalise(t, mask=False, lower=False) for t in texts]


def _margins(model, tok, texts, max_len: int, bs: int, dev: str):
    """``logit_spam - logit_ham`` for a list of raw messages."""
    import torch

    texts = list(texts)
    if not texts:
        return np.empty(0)
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        for i in range(0, len(texts), bs):
            enc = tok(
                prep(texts[i : i + bs]),
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            logits = model(**enc).logits.float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def train(
    train_df,
    out_dir,
    model_name: str = MODEL,
    max_len: int = 128,
    bs: int = 16,
    accum: int = 1,
    epochs: int = 3,
    lr: float = 3e-5,
    seed: int = 42,
    freeze_emb: bool = True,
    grad_ckpt: bool = False,
    device: str | None = None,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)
    if freeze_emb:  # ~197M of 307M params: the memory saver
        model.get_input_embeddings().weight.requires_grad_(False)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    steps = epochs * int(np.ceil(len(train_df) / (bs * accum)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    texts, y = train_df["text"].tolist(), train_df["y"].values
    if not texts:
        raise ValueError("empty training frame")

    def step() -> None:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(texts))
        n_batches = int(np.ceil(len(order) / bs))
        for k, i in enumerate(range(0, len(order), bs)):
            batch = order[i : i + bs]
            enc = tok(
                prep([texts[j] for j in batch]),
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = (
                    torch.nn.functional.cross_entropy(
                        model(**enc).logits.float(), torch.tensor(y[batch], device=dev)
                    )
                    / accum
                )
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0:
                step()
        if n_batches % accum != 0:  # flush a partial accumulation window
            step()

    if dev == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def cal_margins(model_dir, texts, max_len: int = 128, bs: int = 64, device: str | None = None):
    """Margins for the calibration split, from a saved checkpoint."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    return _margins(model, tok, list(texts), max_len, bs, dev)


class MMBertScorer:
    """``predict_proba(list[str]) -> (n, 2)``, temperature-scaled.

    Normalises internally, like the classical pipeline, so train and serve cannot
    drift.
    """

    def __init__(self, model_dir, T: float = 1.0, max_len: int = 128, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.T = float(T)
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev)

    @classmethod
    def load(cls, model_dir, T: float = 1.0, **kw):
        return cls(model_dir, T=T, **kw)

    def predict_proba(self, texts):
        margins = _margins(self.model, self.tok, list(texts), self.max_len, 32, self.dev)
        if margins.size == 0:
            return np.empty((0, 2))
        p = 1 / (1 + np.exp(-np.clip(margins / self.T, -40, 40)))
        return np.c_[1 - p, p]
