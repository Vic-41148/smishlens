"""Generate docs/DEPLOYMENT.md from the result JSONs and the text-free manifest.

Deployment numbers (artifact size, RSS, latency, gold FPR, pool counts) are read
from ``reports/results/*.json``, ``split_manifest.csv`` or the committed data
cards. Hosting limits are external facts and are attributed to their official
pages. No message text is read or written.
"""

from __future__ import annotations

import json
import re

from smishlens.docs_gen import LICENCE, LIVE_URL, pool_stats
from smishlens.results_summary import ROOT, load, num

OUT = ROOT / "docs" / "DEPLOYMENT.md"
CARD_DIR = ROOT / "data" / "cards"
UCI_CARD = ROOT / "data" / "processed" / "uci" / "DATA_CARD.md"
DEPLOY_ENV = ROOT / "reports" / "results" / "deploy_env.json"

VERIFIED = "2026-11-22"

POOL_SOURCES = ("imc25", "mishra", "moz", "exais", "smishtank")
CLEARED_SOURCES = ("imc25", "mishra")


def _first_int(path, label: str) -> int:
    """First integer in the value cell of a ``| label | value |`` card row."""
    for line in path.read_text().splitlines():
        cells = [cell.strip().strip("*") for cell in line.split("|")]
        if len(cells) > 2 and cells[1] == label:
            match = re.search(r"[\d,]+", cells[2])
            if match:
                return int(match.group().replace(",", ""))
    raise ValueError(f"{label!r} row not found in {path}")


def raw_counts() -> dict:
    """Raw rows shipped by each pool source, read from its committed card."""
    return {key: _first_int(CARD_DIR / f"{key}.md", "Rows (raw)") for key in POOL_SOURCES}


def uci_counts() -> dict:
    """UCI anchor totals from the committed, text-free data card."""
    raw = _first_int(UCI_CARD, "Raw rows read")
    for line in UCI_CARD.read_text().splitlines():
        cells = [c.strip().strip("*").replace(",", "") for c in line.split("|")]
        if len(cells) > 4 and cells[1] == "total":
            return {
                "raw": raw,
                "rows": int(cells[2]),
                "spam": int(cells[3]),
                "ham": int(cells[4]),
            }
    raise ValueError("UCI total row not found in DATA_CARD.md")


def hosting_facts() -> list[str]:
    return [
        "## Hosting facts (verified " + VERIFIED + ")",
        "",
        "### Hugging Face Spaces",
        "",
        "- The **Streamlit SDK was deprecated on 2025-04-30**. Streamlit apps now",
        "  run on the **Docker** template, and Docker Spaces require a paid plan",
        "  (PRO for personal accounts). Free accounts get **static Spaces only**.",
        "  There is no free private compute Space, and **protected** visibility",
        "  also requires PRO.",
        "  - https://huggingface.co/docs/hub/en/spaces-changelog",
        "  - https://huggingface.co/docs/hub/en/spaces-overview",
        "",
        "### Streamlit Community Cloud",
        "",
        "- **Free.** The app inherits the GitHub repository's visibility: a",
        "  private repository produces a private app by default, and it can be",
        "  switched to public in the app settings.",
        "- **One private-repository app at a time.** Invited viewers sign in.",
        "  - https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app",
        "- **Python version.** Community Cloud supports released Python versions",
        "  that are still receiving security updates, and **defaults to 3.12**.",
        "  A version is chosen from the \"Python version\" dropdown in the",
        "  \"Advanced settings\" modal; an app on an unsupported version is",
        "  forcibly upgraded and may break. Do not assume 3.14 is selectable:",
        "  check the dropdown, and prefer 3.12, which is the default and the",
        "  version this bundle is verified against (`make deploy-env`).",
        "  - https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy#optional-configure-secrets-and-python-version",
        "  - https://docs.streamlit.io/deploy/streamlit-community-cloud/status#python-environments",
        "",
        "### Self-host fallback",
        "",
        "- Coolify + Gitea on the home server (`192.168.0.101`) is a valid",
        "  fallback. It is not used by the plan below.",
        "",
    ]


def split_table() -> list[str]:
    return [
        "## Two-repository split",
        "",
        "| content | public repo | private repo |",
        "|---|---|---|",
        "| `src/`, `tests/`, `app/`, `serve/` | yes | yes |",
        "| `docs/`, `README.md`, `MODEL_CARD.md` | yes | yes |",
        "| `data/cards/`, `data/processed/modern/split_manifest.csv` | yes | yes |",
        "| `reports/results/*.json`, `reports/verification_log.md` | yes | yes |",
        "| `requirements.txt` | yes | yes |",
        "| `data/processed/modern/{train,cal,test}.csv` | **no** | **no** |",
        "| `data/raw/`, `data/processed/review/` | **no** | **no** |",
        "| `artifacts/serve/svm_bundle.joblib` | **no** | **yes** |",
        "| `artifacts/serve/metadata.json` | **no** | **yes** |",
        "| `artifacts/serve/MANIFEST.sha256` | **no** | **yes** |",
        "",
        "The private repository is byte-identical to the public one except for",
        "**one commit** that adds the three serving files and a scoped un-ignore.",
        "Never merge that commit back into the public repository.",
        "",
    ]


def licence_lines(stats: dict) -> list[str]:
    """Raw rows vs in-pool rows, reconciled to the manifest total."""
    raw = raw_counts()
    uci = uci_counts()
    pool_total = sum(stats[key]["n"] for key in POOL_SOURCES)
    pool_ham = sum(stats[key]["neg"] for key in POOL_SOURCES)
    pool_spam = sum(stats[key]["pos"] for key in POOL_SOURCES)
    raw_total = sum(raw[key] for key in POOL_SOURCES)

    lines = [
        "## Row reconciliation and licence position",
        "",
        "The **raw** column is what each source shipped, read from its committed",
        "data card. The **in-pool** column is what survives the frozen build and",
        "is counted in `data/processed/modern/split_manifest.csv`. Only the",
        "in-pool column sums to the pool total.",
        "",
        "| source | raw source rows | in-pool rows | in-pool ham / positive | licence |",
        "|---|---:|---:|---|---|",
    ]
    for key in POOL_SOURCES:
        name, licence, _role = LICENCE[key]
        source = stats[key]
        lines.append(
            f"| {name} | {raw[key]:,} | {source['n']:,} | "
            f"{source['neg']:,} / {source['pos']:,} | {licence} |"
        )
    lines += [
        f"| **pool total** | **{raw_total:,}** | **{pool_total:,}** | "
        f"**{pool_ham:,} / {pool_spam:,}** | |",
        f"| UCI SMS Spam (separate anchor, **not in the pool**) | "
        f"{uci['raw']:,} raw / {uci['rows']:,} kept | — | "
        f"{uci['ham']:,} / {uci['spam']:,} | CC BY 4.0 |",
        "",
        "The two 2,330 figures are not a copy error: MOZ and ExAIS independently",
        "both keep 2,330 rows after cleaning. The raw counts differ (2,561 vs",
        "5,240), so the columns do not reconcile row-for-row; only the in-pool",
        "column reconciles to the pool.",
        "",
    ]

    cleared_rows = sum(stats[key]["n"] for key in CLEARED_SOURCES)
    cleared_ham = sum(stats[key]["neg"] for key in CLEARED_SOURCES)
    cleared_spam = sum(stats[key]["pos"] for key in CLEARED_SOURCES)
    cleared_share = 100.0 * cleared_rows / pool_total
    with_uci = cleared_rows + uci["rows"]
    with_uci_ham = cleared_ham + uci["ham"]
    with_uci_spam = cleared_spam + uci["spam"]

    lines += [
        "### Licence-clean subset",
        "",
        "Only IMC'25 and Mishra carry a clear licence (CC BY 4.0) **inside the",
        "pool**. Their in-pool rows are:",
        "",
        f"- **{cleared_rows:,} rows** = {cleared_share:.1f}% of the {pool_total:,}-row",
        f"  pool; **{cleared_ham:,} ham / {cleared_spam:,} positive**.",
        "",
        "That is the correct figure. It is **not** \"close to the pool size\":",
        "about one pool row in six comes from a source whose text licence is",
        "unverified. Adding the separate 2011 UCI anchor (CC BY 4.0, not part of",
        "the pool) gives a combined training set of:",
        "",
        f"- **{with_uci:,} rows** = **{with_uci_ham:,} ham / {with_uci_spam:,} positive**.",
        "",
        "Its ham composition is entirely different: UCI plus Mishra, with no MOZ",
        "or ExAIS bank/service-notification negatives. A model retrained on it",
        "would lose exactly the negatives the modern pool was built to add.",
        "",
        "The bundle is a **TF-IDF vocabulary plus SVC weights**, so it is derived",
        "from the pool text, not just from the labels. MOZ, ExAIS and Smishtank",
        "have unverified or missing text licences, so **redistributing the bundle",
        "is not clearly safe**. Keeping it out of the public repository avoids",
        "redistribution.",
        "",
        "This does **not** clear the dataset licences for serving predictions,",
        "and it does not make the model weights open. Treat the private repo as",
        "the boundary: the artifact stays there, and the app is the only public",
        "surface.",
        "",
    ]
    return lines


def _yes(value) -> str:
    return {True: "yes", False: "**no**"}.get(value, "—")


def deploy_env_lines() -> list[str]:
    """Deployment-environment checks, read from ``reports/results/deploy_env.json``."""
    lines = ["## Deployment-environment check", ""]
    if not DEPLOY_ENV.exists():
        return lines + [
            "Not yet run. Run `make deploy-env` to build a clean venv from",
            "`requirements.txt`, smoke the app and the Docker image, and write",
            "`reports/results/deploy_env.json`.",
            "",
        ]

    env = json.loads(DEPLOY_ENV.read_text())
    packages = env.get("packages", {})
    cross = env.get("cross_version", {})
    docker = env.get("docker", {})
    lines += [
        f"Generated {env.get('generated_utc', '—')} by `make deploy-env`. "
        "Every value below is read from `reports/results/deploy_env.json`.",
        "",
        "| check | result |",
        "|---|---|",
        f"| clean-venv Python | {env.get('python_version', '—')} |",
        f"| requirements file | `{env.get('requirements', '—')}` "
        f"({len(packages)} pins) |",
        "| resolved packages | "
        + ", ".join(f"`{name}=={version}`" for name, version in sorted(packages.items()))
        + " |",
        f"| artifact sha256 | `{env.get('artifact_sha256', '—')}` |",
        f"| matches `MANIFEST.sha256` | {_yes(env.get('artifact_matches_manifest'))} |",
        f"| app import path | {_yes(env.get('app_import_ok'))} |",
        f"| bundle loads | {_yes(env.get('bundle_loaded'))} |",
        f"| AppTest (notice + score) | {_yes(env.get('apptest_ok'))} |",
        f"| pool CSVs absent during check | {_yes(env.get('csvs_absent'))} |",
        f"| CSVs restored byte-identical | {_yes(env.get('csv_restore_verified'))} |",
    ]
    if cross:
        lines.append(
            f"| cross-version score ({cross.get('reference_python', '—')} vs "
            f"{env.get('python_version', '—')}) | {_yes(cross.get('match'))}, "
            f"max |Δ| {num(cross.get('max_abs_diff'), 12)} "
            f"(tolerance {num(cross.get('tolerance'), 12)}) |"
        )
    if docker:
        if docker.get("tested"):
            lines.append(
                f"| Docker image | `{docker.get('base_image', '—')}`, "
                f"{num(docker.get('image_size_mb'), 1)} MB, health "
                f"{_yes(docker.get('health_ok'))}, no pool data "
                f"{_yes(docker.get('no_pool_csvs'))}, removed "
                f"{_yes(docker.get('image_removed'))} |"
            )
        else:
            lines.append(
                f"| Docker image | not tested: {docker.get('note', docker.get('error', '—'))} |"
            )
    lines += [
        "",
        "The synthetic probe is a fixed invented string; its text is never",
        "printed, logged or written. Only its band, conformal set and calibrated",
        "score are recorded.",
        "",
    ]
    return lines


def bundle_pin_lines() -> list[str]:
    """How the loader's version pins are recorded, and this bundle's provenance."""
    return [
        "### Bundle version provenance",
        "",
        "`load_bundle()` refuses a pickle whose recorded `sklearn_version` or",
        "`joblib_version` differs from the running environment, and refuses a",
        "bundle that records neither. `make build-serve` writes both at build",
        "time.",
        "",
        "This bundle predates the `joblib_version` field. The value now in",
        "`artifacts/serve/metadata.json` was recorded **post hoc** from the",
        "verified loading environment, not at build time: the clean Python 3.12",
        "venv above (pins in the table above) loaded the frozen pickle and",
        "scored the probe, and that joblib version is the one recorded. The",
        "bundle was **not** rebuilt or retrained to add the field, so",
        "`svm_bundle.joblib` and its sha256 are unchanged. The next",
        "`make build-serve` records the field at build time and retires this",
        "note.",
        "",
    ]


def procedure() -> list[str]:
    env = json.loads(DEPLOY_ENV.read_text()) if DEPLOY_ENV.exists() else {}
    python_version = env.get("python_version", "3.12")
    return [
        "## Procedure (two repositories, one clone)",
        "",
        "| remote | url | branch | contents |",
        "|---|---|---|---|",
        "| `origin` | `git@github.com:Vic-41148/smishlens.git` | `main` | code, docs, text-free data |",
        "| `private` | you add it | `private-deploy` → `main` | the same tree plus the serving bundle |",
        "",
        "`private-deploy` is a **local** branch created from `main`. It adds exactly",
        "one commit: `deploy/private-unignore.txt` appended to `.gitignore` and the",
        "three files under `artifacts/serve/`. Nothing else differs.",
        "",
        "### 1. The first push to origin",
        "",
        "`origin` already exists. Its only commit is GitHub's \"Initial commit\" and",
        "it contains `LICENSE` alone:",
        "",
        "```bash",
        "git show origin/main --stat --format=\"%h %s\"",
        "git ls-tree -r --name-only origin/main     # LICENSE",
        "```",
        "",
        "In this clone that commit is the **root of `main`** (`git merge-base main",
        "origin/main` returns it), so the histories are already related and the",
        "push is a plain fast-forward:",
        "",
        "```bash",
        "git merge-base --is-ancestor origin/main main && echo fast-forward",
        "git push origin main",
        "```",
        "",
        "Do **not** force-push. If a later clone does not share that root, merge",
        "instead:",
        "",
        "```bash",
        "git fetch origin",
        "git merge --allow-unrelated-histories origin/main",
        "git commit -S            # only if the merge did not commit on its own",
        "```",
        "",
        "`git pull --rebase origin main` also works, but a rebase rewrites every",
        "local commit hash and **invalidates every GPG signature**; each commit",
        "then has to be re-signed. Prefer the merge unless the origin commit is",
        "empty or trivial.",
        "",
        "### 2. Add the private remote and push",
        "",
        "```bash",
        "git remote add private <private-repo-url>",
        "git push origin main",
        "git push private private-deploy:main",
        "```",
        "",
        "The pre-push hook dispatches on the remote name and reads the refs from",
        "stdin, so it behaves the same from a linked worktree:",
        "",
        "| Remote | Allowed ref | Checks |",
        "|---|---|---|",
        "| `origin` | `main` → `main` only | fast-forward, no `artifacts/serve` in history, `public_check` |",
        "| `private` | `private-deploy` → `main` only | fast-forward, URL must equal `git config remote.private.url`, `public_check --private` |",
        "| any other name | nothing | refused |",
        "",
        "A force push is refused on both remotes. The URL check compares against",
        "the URL already configured for `private`, so the hook never carries a",
        "repository URL of its own.",
        "",
        "### 3. Streamlit Community Cloud",
        "",
        "1. share.streamlit.io → **Create app** → pick the **private** repository,",
        "   branch `main`, entrypoint `app/app.py`.",
        "2. **Advanced settings → Python version.** Choose the version recorded by",
        f"   `make deploy-env` (**{python_version}** here). The docs say Community",
        "   Cloud defaults to 3.12 and offers released, security-supported versions",
        "   from a dropdown; check the dropdown instead of assuming 3.14 is there.",
        "3. Dependencies come from `requirements.txt` at the repository root. Do not",
        "   add `requirements-dev.txt`; the cloud does not read it.",
        "4. **Visibility.** A private repository produces a private app by default.",
        "   You can switch the app to public in its settings; the repository and the",
        "   bundle stay private.",
        "5. **Viewers.** Invite them by email in the app settings; they must sign in.",
        "   Only **one private-repository app** is allowed at a time.",
        "6. The cloud overrides some `.streamlit/config.toml` keys, for example",
        "   `browser.gatherUsageStats=true` and `client.showErrorDetails=false`.",
        "",
        "### 4. Updating the private branch",
        "",
        "Develop on `main`, never on `private-deploy`. When `main` moves:",
        "",
        "```bash",
        "git switch private-deploy",
        "git merge main            # or: git rebase main",
        "make build-serve          # needs your licensed copy of the pool data",
        "git add artifacts/serve && git commit -S -m \"deploy: refresh serving bundle\"",
        "git switch main",
        "```",
        "",
        "`make build-serve` rewrites the bundle, `metadata.json` and",
        "`MANIFEST.sha256` together. Commit all three in one commit, then check that",
        "`git diff --stat main..private-deploy` lists only `.gitignore` and those",
        "three files.",
        "",
        "### 5. Coolify fallback (home server)",
        "",
        "If Community Cloud is unavailable, run the same image on `192.168.0.101`:",
        "",
        "```bash",
        "git switch private-deploy",
        "docker build -t smishlens:local .",
        "docker run -d --name smishlens -p 8501:8501 smishlens:local",
        "```",
        "",
        "In Coolify: new resource → Dockerfile → this repository and branch. Expose",
        "port 8501 and health-check `/_stcore/health`. The public repository cannot",
        "build this image: `artifacts/serve/` is absent, so the app starts in its",
        "documented \"bundle not distributed\" state.",
        "",
        "### 6. Pre-push checklist",
        "",
        "- [ ] `make public-check` passes on `main`.",
        "- [ ] `python -m smishlens.public_check --private` passes on `private-deploy`.",
        "- [ ] `ruff check .` and the full `pytest` suite pass.",
        "- [ ] `make summary` leaves no diff (generated docs are current).",
        "- [ ] `git log main -- artifacts/serve` is empty.",
        "- [ ] `git diff --stat main..private-deploy` lists only the four files.",
        "- [ ] `MANIFEST.sha256` matches the bundle and metadata.",
        "- [ ] `make install-hooks` has installed the pre-push hook.",
        "- [ ] The app loads and scores with the pool CSVs absent, and the",
        "      always-visible notice is present.",
        "",
    ]


def runtime_lines(ev: dict, boot: dict, base: dict) -> list[str]:
    svm = ev["arms"]["svm"]
    gold = boot["gold"]["v2"]
    return [
        "## What Streamlit Cloud needs",
        "",
        "- Entrypoint: `app/app.py`.",
        "- Dependencies: `requirements.txt` at the repository root. Streamlit",
        "  Community Cloud reads that file, not `requirements-dev.txt`. It is",
        "  app-minimal (no torch or transformers), so the build stays small.",
        "- Bundle: `artifacts/serve/` (private repository only).",
        f"- Artifact on disk: **{num(svm['disk_mb'], 1)} MB**; peak RSS",
        f"  **{num(svm['rss_mb'], 0)} MB**; batch-1 p50",
        f"  **{num(svm['batch_1_ms']['p50'], 2)} ms** (E17).",
        "- The pool CSVs are not needed at runtime. The bundle loads and scores",
        "  with `train/cal/test.csv` absent.",
        "- Streamlit Cloud resource limit: **2.7 GB RAM and 2 CPU cores max,",
        "  per the docs as of Feb 2024; this may change, so re-check before",
        "  relying on it.** The measured footprint is well inside that figure.",
        "",
        "## Visibility",
        "",
        "- Private repository -> private app by default. Only invited viewers",
        "  can open it.",
        "- Switching the app to public makes the running app public, but the",
        "  **repository and the bundle stay private**.",
        "- Deploying the artifact-bearing repository publicly would expose the",
        "  bundle; do not do that.",
        "",
        "## Evaluation context shown in the app",
        "",
        f"- Synthetic gold v2: {gold['n_pos']} spam / {gold['n_neg']} ham; svm FPR",
        f"  {num(gold['models']['svm']['fpr']['point'])}.",
        f"- Pool test: {base['pool_test_rows']:,} rows, prevalence",
        f"  {num(base['pool_test_prevalence'])}.",
        "",
    ]


def deployment() -> str:
    stats, _ = pool_stats()
    ev = load("e17_latency.json")
    boot = load("e6_gold_v2_bootstrap.json")
    base = load("e3_pool.json")["base_rate"]
    lines = [
        "# Deployment plan",
        "",
        "Status: **live** at "
        f"<{LIVE_URL}>, served from the private repository. `origin` exists; "
        "`private` holds the bundle.",
        "",
        "Primary target: **Streamlit Community Cloud**, deployed from a private",
        "GitHub repository. Public repository: code and docs only, no artifact.",
        "",
    ]
    lines += hosting_facts()
    lines += split_table()
    lines += licence_lines(stats)
    lines += deploy_env_lines()
    lines += bundle_pin_lines()
    lines += procedure()
    lines += runtime_lines(ev, boot, base)
    return "\n".join(lines)


def main() -> int:
    OUT.write_text(deployment())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
