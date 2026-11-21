# DATA CARD — Africa HF set (synthetic)

| Field | Value |
|---|---|
| Source | Hugging Face `electricsheepafrica/africa-smishing-sms-phishing` |
| Accessed | 2026-11-21 |
| Licence | **Unclear** — the card says "other" |
| Rows (raw) | 10,000 |
| Labels | 5,000 ham / 5,000 smishing |
| Synthetic | **Yes — `is_synthetic = 1` on every row** |
| Languages | English 2,104, Swahili 1,145, Yoruba 841, + others |
| Rows kept in the pool | **0 — excluded** |
| Role | None in Tier 2. At most a train-time augmentation later, at ≤30% of train |

## Why it is excluded

Every row is synthetic. Roadmap §4.1 forbids synthetic rows in a headline test set, and mixing
them into train would let the model learn the generator's style. It is loaded and tagged
`synthetic=True` so that any future use is automatically identifiable, but the pool builder does
not include it.

## Decisions taken in the loader

- `sms_text` is the message; `label` maps `0 → ham`, `1 → smish`.
- `synthetic=True` is forced by the adapter, regardless of the file's own flag.
- The 90-odd derived columns (`type_*`, `cta_*`, `lang_*`, `victim_*`) are ignored: they are
  derived from the label or from metadata no real deployment would have.

## Known biases

- Generated text with regular structure; likely easier than real SMS.
- "Synthetic" is a property of the whole set, not of individual rows: do not treat a subset as
  real.
