# Deployment plan

Status: **prepared, not pushed**. `origin` already exists; you add `private`.

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
- **Python version.** Community Cloud supports released Python versions
  that are still receiving security updates, and **defaults to 3.12**.
  A version is chosen from the "Python version" dropdown in the
  "Advanced settings" modal; an app on an unsupported version is
  forcibly upgraded and may break. Do not assume 3.14 is selectable:
  check the dropdown, and prefer 3.12, which is the default and the
  version this bundle is verified against (`make deploy-env`).
  - https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy#optional-configure-secrets-and-python-version
  - https://docs.streamlit.io/deploy/streamlit-community-cloud/status#python-environments

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

## Deployment-environment check

Generated 2026-11-21T20:58:20+00:00 by `make deploy-env`. Every value below is read from `reports/results/deploy_env.json`.

| check | result |
|---|---|
| clean-venv Python | 3.12.14 |
| requirements file | `requirements.txt` (7 pins) |
| resolved packages | `joblib==1.6.0`, `numpy==2.5.5`, `pandas==2.3.3`, `pyarrow==26.0.1`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `streamlit==1.67.0` |
| artifact sha256 | `eeca262df368dcabd58d803038a876ca510c549b9947332a3cb2d88b96a28843` |
| matches `MANIFEST.sha256` | yes |
| app import path | yes |
| bundle loads | yes |
| AppTest (notice + score) | yes |
| pool CSVs absent during check | yes |
| CSVs restored byte-identical | yes |
| cross-version score (3.14.10 vs 3.12.14) | yes, max |Δ| 0.000000000000 (tolerance 0.000000001000) |
| Docker image | `python:3.12-slim`, 1090.2 MB, health yes, no pool data yes, removed yes |

The synthetic probe is a fixed invented string; its text is never
printed, logged or written. Only its band, conformal set and calibrated
score are recorded.

## Procedure (two repositories, one clone)

| remote | url | branch | contents |
|---|---|---|---|
| `origin` | `git@github.com:Vic-41148/smishlens.git` | `main` | code, docs, text-free data |
| `private` | you add it | `private-deploy` → `main` | the same tree plus the serving bundle |

`private-deploy` is a **local** branch created from `main`. It adds exactly
one commit: `deploy/private-unignore.txt` appended to `.gitignore` and the
three files under `artifacts/serve/`. Nothing else differs.

### 1. The first push to origin

`origin` already exists. Its only commit is GitHub's "Initial commit" and
it contains `LICENSE` alone:

```bash
git show origin/main --stat --format="%h %s"
git ls-tree -r --name-only origin/main     # LICENSE
```

In this clone that commit is the **root of `main`** (`git merge-base main
origin/main` returns it), so the histories are already related and the
push is a plain fast-forward:

```bash
git merge-base --is-ancestor origin/main main && echo fast-forward
git push origin main
```

Do **not** force-push. If a later clone does not share that root, merge
instead:

```bash
git fetch origin
git merge --allow-unrelated-histories origin/main
git commit -S            # only if the merge did not commit on its own
```

`git pull --rebase origin main` also works, but a rebase rewrites every
local commit hash and **invalidates every GPG signature**; each commit
then has to be re-signed. Prefer the merge unless the origin commit is
empty or trivial.

### 2. Add the private remote and push

```bash
git remote add private <private-repo-url>
git push origin main
git push private private-deploy:main
```

The pre-push hook dispatches on the remote name and reads the refs from
stdin, so it behaves the same from a linked worktree:

| Remote | Allowed ref | Checks |
|---|---|---|
| `origin` | `main` → `main` only | fast-forward, no `artifacts/serve` in history, `public_check` |
| `private` | `private-deploy` → `main` only | fast-forward, URL must equal `git config remote.private.url`, `public_check --private` |
| any other name | nothing | refused |

A force push is refused on both remotes. The URL check compares against
the URL already configured for `private`, so the hook never carries a
repository URL of its own.

### 3. Streamlit Community Cloud

1. share.streamlit.io → **Create app** → pick the **private** repository,
   branch `main`, entrypoint `app/app.py`.
2. **Advanced settings → Python version.** Choose the version recorded by
   `make deploy-env` (**3.12.14** here). The docs say Community
   Cloud defaults to 3.12 and offers released, security-supported versions
   from a dropdown; check the dropdown instead of assuming 3.14 is there.
3. Dependencies come from `requirements.txt` at the repository root. Do not
   add `requirements-dev.txt`; the cloud does not read it.
4. **Visibility.** A private repository produces a private app by default.
   You can switch the app to public in its settings; the repository and the
   bundle stay private.
5. **Viewers.** Invite them by email in the app settings; they must sign in.
   Only **one private-repository app** is allowed at a time.
6. The cloud overrides some `.streamlit/config.toml` keys, for example
   `browser.gatherUsageStats=true` and `client.showErrorDetails=false`.

### 4. Updating the private branch

Develop on `main`, never on `private-deploy`. When `main` moves:

```bash
git switch private-deploy
git merge main            # or: git rebase main
make build-serve          # needs your licensed copy of the pool data
git add artifacts/serve && git commit -S -m "deploy: refresh serving bundle"
git switch main
```

`make build-serve` rewrites the bundle, `metadata.json` and
`MANIFEST.sha256` together. Commit all three in one commit, then check that
`git diff --stat main..private-deploy` lists only `.gitignore` and those
three files.

### 5. Coolify fallback (home server)

If Community Cloud is unavailable, run the same image on `192.168.0.101`:

```bash
git switch private-deploy
docker build -t smishlens:local .
docker run -d --name smishlens -p 8501:8501 smishlens:local
```

In Coolify: new resource → Dockerfile → this repository and branch. Expose
port 8501 and health-check `/_stcore/health`. The public repository cannot
build this image: `artifacts/serve/` is absent, so the app starts in its
documented "bundle not distributed" state.

### 6. Pre-push checklist

- [ ] `make public-check` passes on `main`.
- [ ] `python -m smishlens.public_check --private` passes on `private-deploy`.
- [ ] `ruff check .` and the full `pytest` suite pass.
- [ ] `make summary` leaves no diff (generated docs are current).
- [ ] `git log main -- artifacts/serve` is empty.
- [ ] `git diff --stat main..private-deploy` lists only the four files.
- [ ] `MANIFEST.sha256` matches the bundle and metadata.
- [ ] `make install-hooks` has installed the pre-push hook.
- [ ] The app loads and scores with the pool CSVs absent, and the
      always-visible notice is present.

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
