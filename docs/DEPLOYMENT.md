# Deployment plan

Status: **prepared, not pushed**. This repository has no remotes.

Primary target: **Streamlit Community Cloud**, deployed from a private
GitHub repository. Public repository: code and docs only, no artifact.

## Hosting facts (verified 2026-11-22)

### Hugging Face Spaces

- The **Streamlit SDK was deprecated on 2025-04-30**. Streamlit apps now
  run on the **Docker** template, and Docker Spaces require a paid plan
  (PRO for personal accounts). Free accounts get **static Spaces only**.
  There is no free private compute Space, and **protected** visibility
  also requires PRO.
  - https://huggingface.co/docs/hub/en/spaces-changelog
  - https://huggingface.co/docs/hub/en/spaces-overview

### Streamlit Community Cloud

- **Free.** The app inherits the GitHub repository's visibility: a
  private repository produces a private app by default, and it can be
  switched to public in the app settings.
- **One private-repository app at a time.** Invited viewers sign in.
  - https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

### Self-host fallback

- Coolify + Gitea on the home server (`192.168.0.101`) is a valid
  fallback. It is not used by the plan below.

## Two-repository split

| content | public repo | private repo |
|---|---|---|
| `src/`, `tests/`, `app/`, `serve/` | yes | yes |
| `docs/`, `README.md`, `MODEL_CARD.md` | yes | yes |
| `data/cards/`, `data/processed/modern/split_manifest.csv` | yes | yes |
| `reports/results/*.json`, `reports/verification_log.md` | yes | yes |
| `requirements.txt`, `requirements-app.txt` | yes | yes |
| `data/processed/modern/{train,cal,test}.csv` | **no** | **no** |
| `data/raw/`, `data/processed/review/` | **no** | **no** |
| `artifacts/serve/svm_bundle.joblib` | **no** | **yes** |
| `artifacts/serve/metadata.json` | **no** | **yes** |
| `artifacts/serve/MANIFEST.sha256` | **no** | **yes** |

The private repository is byte-identical to the public one except for
**one commit** that adds the three serving files and a scoped un-ignore.
Never merge that commit back into the public repository.

## Licence position

| source | rows | ham / positive | licence |
|---|---:|---|---|
| IMC'25 smishing | 24,462 | 3 / 24,459 | CC BY 4.0 |
| Mishra & Soni | 599 | 367 / 232 | CC BY 4.0 |
| MOZ-Smishing | 2,330 | 1,784 / 546 | **unverified** (card carries a *model* licence) |
| ExAIS SMS Spam | 2,330 | 2,330 / 0 | **unverified** (no licence file) |
| Smishtank Dataset I | 726 | 0 / 726 | **unverified** (cite only) |
| UCI SMS Spam | 5,159 | 4,517 / 642 | CC BY 4.0 |

The bundle is a **TF-IDF vocabulary plus SVC weights**, so it is derived
from the pool text, not just from the labels. MOZ, ExAIS and Smishtank
have unverified or missing text licences, so **redistributing the bundle
is not clearly safe**. Keeping it out of the public repository avoids
redistribution.

This does **not** clear the dataset licences for serving predictions,
and it does not make the model weights open. Treat the private repo as
the boundary: the artifact stays there, and the app is the only public
surface.

## Procedure (run these yourself; this repo has no remotes)

1. Create the **public** GitHub repository and push `main`.
2. Clone it into a second working directory.
3. Append `deploy/private-unignore.txt` to `.gitignore` in that clone.
4. Copy `artifacts/serve/` from the build machine, then commit the three
   files and push to the **private** repository.
5. In Streamlit Community Cloud, deploy from the **private** repository
   with entrypoint `app/app.py`. Choose app visibility in the app
   settings afterwards.
6. Before each push, run the matching check:

   ```bash
   python -m smishlens.public_check            # public repo
   python -m smishlens.public_check --private  # private repo
   ```

## What Streamlit Cloud needs

- Entrypoint: `app/app.py`.
- Dependencies: `requirements.txt` at the repository root. It has no
  torch or transformers, so the build stays small.
- Bundle: `artifacts/serve/` (private repository only).
- Artifact on disk: **3.8 MB**; peak RSS
  **227 MB**; batch-1 p50
  **2.84 ms** (E17).
- The pool CSVs are not needed at runtime. The bundle loads and scores
  with `train/cal/test.csv` absent.
- Streamlit Cloud limit: 2.7 GB RAM and 2 CPU cores maximum. The
  measured footprint is well inside that.

## Visibility

- Private repository -> private app by default. Only invited viewers
  can open it.
- Switching the app to public makes the running app public, but the
  **repository and the bundle stay private**.
- Deploying the artifact-bearing repository publicly would expose the
  bundle; do not do that.

## Evaluation context shown in the app

- Synthetic gold v2: 23 spam / 35 ham; svm FPR
  0.5429.
- Pool test: 6,090 rows, prevalence
  0.8529.

## Pre-push checklist

- [ ] `python -m smishlens.public_check` passes in the public repo.
- [ ] `git log --all -- artifacts/serve` is empty in the public repo.
- [ ] `python -m smishlens.public_check --private` passes in the private repo.
- [ ] The private repo has exactly the three serving files under
      `artifacts/serve/`.
- [ ] `MANIFEST.sha256` matches the bundle and metadata.
- [ ] The app loads and scores with the pool CSVs absent.
- [ ] The always-visible notice is present.
- [ ] `README.md` links to `MODEL_CARD.md` and `docs/DEPLOYMENT.md`.

## Fallback

If Streamlit Community Cloud is unavailable, self-host the same app with
Coolify + Gitea on the home server. The runtime requirements are the
same: `requirements.txt`, `app/app.py`, and `artifacts/serve/`.
