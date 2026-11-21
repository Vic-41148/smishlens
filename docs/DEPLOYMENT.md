# Deployment plan

Status: **prepared, not pushed**. No remote is created by this repo.

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
| `requirements.txt` | yes | yes |
| `data/processed/modern/{train,cal,test}.csv` | **no** | **no** |
| `data/raw/`, `data/processed/review/` | **no** | **no** |
| `artifacts/serve/svm_bundle.joblib` | **no** | **yes** |
| `artifacts/serve/metadata.json` | **no** | **yes** |
| `artifacts/serve/MANIFEST.sha256` | **no** | **yes** |

The private repository is byte-identical to the public one except for
**one commit** that adds the three serving files and a scoped un-ignore.
Never merge that commit back into the public repository.

## Row reconciliation and licence position

The **raw** column is what each source shipped, read from its committed
data card. The **in-pool** column is what survives the frozen build and
is counted in `data/processed/modern/split_manifest.csv`. Only the
in-pool column sums to the pool total.

| source | raw source rows | in-pool rows | in-pool ham / positive | licence |
|---|---:|---:|---|---|
| IMC'25 smishing | 33,869 | 24,462 | 3 / 24,459 | CC BY 4.0 |
| Mishra & Soni | 5,971 | 599 | 367 / 232 | CC BY 4.0 |
| MOZ-Smishing | 2,561 | 2,330 | 1,784 / 546 | **unverified** (card carries a *model* licence) |
| ExAIS SMS Spam | 5,240 | 2,330 | 2,330 / 0 | **unverified** (no licence file) |
| Smishtank Dataset I | 1,062 | 726 | 0 / 726 | **unverified** (cite only) |
| **pool total** | **48,703** | **30,447** | **4,484 / 25,963** | |
| UCI SMS Spam (separate anchor, **not in the pool**) | 5,574 raw / 5,159 kept | — | 4,517 / 642 | CC BY 4.0 |

The two 2,330 figures are not a copy error: MOZ and ExAIS independently
both keep 2,330 rows after cleaning. The raw counts differ (2,561 vs
5,240), so the columns do not reconcile row-for-row; only the in-pool
column reconciles to the pool.

### Licence-clean subset

Only IMC'25 and Mishra carry a clear licence (CC BY 4.0) **inside the
pool**. Their in-pool rows are:

- **25,061 rows** = 82.3% of the 30,447-row
  pool; **370 ham / 24,691 positive**.

That is the correct figure. It is **not** "close to the pool size":
about one pool row in six comes from a source whose text licence is
unverified. Adding the separate 2011 UCI anchor (CC BY 4.0, not part of
the pool) gives a combined training set of:

- **30,220 rows** = **4,887 ham / 25,333 positive**.

Its ham composition is entirely different: UCI plus Mishra, with no MOZ
or ExAIS bank/service-notification negatives. A model retrained on it
would lose exactly the negatives the modern pool was built to add.

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

The private artifact lives on a **local branch** in this same clone.
There is no second clone and no `private` remote until you add one.

1. Add the two remotes:

   ```bash
   git remote add origin <public-repo-url>
   git remote add private <private-repo-url>
   ```

2. Push the public branch:

   ```bash
   git push origin main
   ```

3. Push the artifact branch to the private repository's `main`:

   ```bash
   git push private private-deploy:main
   ```

4. In Streamlit Community Cloud, deploy from the **private** repository
   with entrypoint `app/app.py`. Choose app visibility in the app
   settings afterwards.

### Updating the private branch

Develop on `main`, never on `private-deploy`. When `main` changes:

```bash
git switch private-deploy
git rebase main          # or: git merge main
make build-serve         # rebuild artifacts/serve from your licensed data
git add artifacts/serve && git commit -S -m "deploy: refresh serving bundle"
git switch main
```

`make build-serve` rewrites `svm_bundle.joblib`, `metadata.json` and
`MANIFEST.sha256` together, so the manifest always matches the bundle.
Commit all three in the same commit.

### Before every push

```bash
python -m smishlens.public_check            # public mode, inspects main
python -m smishlens.public_check --private  # run on private-deploy
```

`make install-hooks` adds a `pre-push` hook that runs the matching check
on the refs being pushed, refuses `private-deploy` on any remote not
named `private`, and refuses any ref that touches `artifacts/serve` on
`origin`.

## What Streamlit Cloud needs

- Entrypoint: `app/app.py`.
- Dependencies: `requirements.txt` at the repository root. Streamlit
  Community Cloud reads that file, not `requirements-dev.txt`. It is
  app-minimal (no torch or transformers), so the build stays small.
- Bundle: `artifacts/serve/` (private repository only).
- Artifact on disk: **3.8 MB**; peak RSS
  **227 MB**; batch-1 p50
  **2.84 ms** (E17).
- The pool CSVs are not needed at runtime. The bundle loads and scores
  with `train/cal/test.csv` absent.
- Streamlit Cloud resource limit: **2.7 GB RAM and 2 CPU cores max,
  per the docs as of Feb 2024; this may change, so re-check before
  relying on it.** The measured footprint is well inside that figure.

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

- [ ] `python -m smishlens.public_check` passes on `main`.
- [ ] `git log main -- artifacts/serve` is empty.
- [ ] `python -m smishlens.public_check --private` passes on `private-deploy`.
- [ ] The private branch has exactly the three serving files under
      `artifacts/serve/`.
- [ ] `MANIFEST.sha256` matches the bundle and metadata.
- [ ] The app loads and scores with the pool CSVs absent.
- [ ] The always-visible notice is present.
- [ ] `README.md` links to `MODEL_CARD.md` and `docs/DEPLOYMENT.md`.

## Fallback

If Streamlit Community Cloud is unavailable, self-host the same app with
Coolify + Gitea on the home server. The runtime requirements are the
same: `requirements.txt`, `app/app.py`, and `artifacts/serve/`.
