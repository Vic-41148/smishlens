# DATA CARD — WSPR Lab SMS phishing

| Field | Value |
|---|---|
| Source | GitHub `wspr-ncsu/sms-phishing`, `phishing_messages.csv` |
| Accessed | 2026-11-21 |
| Licence | MIT (repository) |
| Rows (raw) | 68,029 messages / 35,128 campaigns |
| Labels | phishing only |
| Languages | Mixed (English, German and others; not declared per row) |
| Rows kept in the pool | **0 — deliberately outside the pool** |
| Role | Recall-only OOD set for Tier 2 evaluation (E10 and the evasion suite) |

## Why it is not in the pool

68,029 positives against ~4,500 ham would make the pool's class balance meaningless. The
roadmap calls it an "optional extra OOD" source, so it is held out of the training pool and used
only to measure recall on a source the model has never seen.

## Decisions taken in the loader

- **The header is wrong.** It declares 6 columns but every row has 7: the campaign label
  (`VIP Club`, `Top 4You`, …) has no header. `pandas.read_csv` therefore shifts `message` and
  reports a mean message length of 9 characters. The adapter uses the stdlib `csv` reader and
  takes field 3 (the quoted message) directly.
- The campaign label is not used as a feature.
- `time` is a UNIX timestamp and is carried as a string; no time split.

## Known biases

- Gateway-collected bulk phishing, not hand-written smishing. 66,883 of 68,029 rows contain a
  URL (`has_url`), so it is a link-heavy set — useful for the URL-shortcut probe, misleading for
  link-less smishing.
- Repeated campaign templates mean near-duplicate rows are common; dedup is applied when the
  set is actually used as an eval fold.
