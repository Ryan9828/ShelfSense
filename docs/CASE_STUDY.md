# ShelfSense — Case Study

A retrospective on building this project end to end: what was tried, what
actually worked, what didn't, and why. Written as interview prep material as
much as documentation — the process is arguably more interesting than the
final architecture.

## Objective

Fill a real gap in a data-science portfolio otherwise built entirely on
regression, classification, time-series, and one LLM-extraction project:
recommender systems. Requirements set at the start: real data (not a toy
dataset), an actual A/B-testing component (not just accuracy), and a
deployed, clickable end product — not just a notebook.

## Dataset and scope decision

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle) — real anonymized retail transactions, ~31M rows / 1.4M customers /
105k articles in full. Two scope cuts were made deliberately:

- **Images skipped.** The competition ships a ~25GB `images.zip`; only the
  tabular CSVs (articles/customers/transactions) were used. Content-based
  similarity runs on text metadata (product name, type, department,
  description) instead of visual features — a smaller, faster, and honestly
  more explainable signal for a portfolio project.
- **Subsampled to 60,000 customers.** The full transaction log is too large
  to iterate on quickly. A random customer subsample (seeded, reproducible)
  keeps the sparsity and cold-start characteristics realistic while running
  the full pipeline in under a minute.

## Architecture (final)

| Component | Approach |
|---|---|
| Control baseline | Global popularity — recent-window best-sellers |
| Warm-customer personalization | **Category-affinity popularity** — best-sellers within a customer's own favorite product category |
| Cold-start (1-2 purchases) | TF-IDF content-based similarity to what they've already bought |
| Fully cold (0 purchases) | Falls back to global popularity |
| Benchmarked, not shipped | Item-based collaborative filtering (ALS) |

Evaluation: Recall@12 / NDCG@12 against a real held-out week of purchases,
with a paired bootstrap (2,000 resamples) giving a 95% CI on every model
comparison — framed explicitly as an **offline A/B test**, since there's no
live traffic to split.

## Results (real data, 3,066 customers with purchases in the held-out week)

| Model | Recall@12 | NDCG@12 | vs. popularity |
|---|---|---|---|
| Popularity (control) | 0.0185 | 0.0107 | — |
| **Hybrid (shipped)** | 0.0155 | 0.0101 | **ties** — 95% CI [-0.0066, +0.0007], p=0.12 |
| Item-CF / ALS (benchmarked) | 0.0091 | 0.0055 | **loses** — 95% CI [-0.0139, -0.0051], p<0.001 |

The headline number is not "hybrid wins" — it's that a properly-run
statistical test told the truth about two different personalization
strategies, and the codebase reflects that honestly instead of hiding it.

## What worked

- **Treating the offline A/B test as load-bearing, not decorative.** Every
  model comparison in this project goes through the same paired-bootstrap
  function (`evaluate.bootstrap_paired_diff`). That discipline is what
  caught the item-CF failure instead of a single point-estimate metric
  quietly shipping a worse model.
- **Category-affinity over item-level CF.** Once item-CF's failure was
  diagnosed (see below), recommending best-sellers within a customer's own
  favorite category — a coarser, category-level signal — statistically tied
  the popularity baseline instead of losing to it, while still tailoring
  what each customer sees.
- **A tiered cold-start strategy.** Routing by purchase-history length
  (0 / 1-2 / 3+) rather than one model for everyone meant every customer
  segment had a sensible strategy, including the ones a pure-CF system has
  nothing to say to.
- **Verifying with a real browser, not just pytest.** A synthetic-data smoke
  test caught a real bug before real data was even downloaded (see below).
  Later, driving the actual Streamlit UI with Playwright — not just curling
  the API — caught two UI bugs that unit tests structurally cannot see
  (a mislabeled form field, a wrapping button).
- **Keeping the failed approach in the codebase.** `collaborative.py` (ALS)
  is still fit and benchmarked on every `python -m shelfsense.train` run,
  and exposed as a labeled "benchmarked, not shipped" option in both the API
  and the demo UI. Deleting it would have made the negative result
  unfalsifiable — reviewers would have to take the README's word for it.

## What didn't work (and how each was actually diagnosed)

**1. Item-based collaborative filtering lost to a popularity baseline.**
First hybrid design routed warm customers straight to ALS. On real data:
Recall@12 0.0090 vs. popularity's 0.0185 — worse by roughly half, with a
95% CI nowhere near zero. Root cause, not a bug: fashion repurchase rates
are low and the catalog turns over fast, so item-level co-purchase patterns
are too sparse over a single-week holdout to compete with "what's trending
right now." Verified this wasn't just a naive-blend artifact by
implementing a proper weighted score blend (ALS score + popularity score,
swept from 100% ALS to 100% popularity across a shared candidate pool) —
**any** non-zero ALS weight made results worse; the relationship was
monotonic. That swept experiment is what justified replacing ALS entirely
rather than tuning it further.

**2. Kaggle authentication — three wrong turns before the real fix.** The
Kaggle CLI installed was v2.2.3, which replaced the old `kaggle.json`
scheme with OAuth login / a plain token file / an env var — none of which
matches most existing documentation or tutorials online.
- First attempt (correct instinct, wrong execution): saved a token to
  `~/.kaggle/access_token`, but the wrong filename format was assumed.
- Second attempt: switched to the old `kaggle.json` approach based on
  outdated knowledge of the CLI — this actually regressed things.
- Third attempt: ran `kaggle auth login` (OAuth) — the CLI reported success
  ("logged in as ryan9827"), but downloads *still* failed with the same
  generic "Authentication required" message.
- Diagnosis required reading the installed package's source and calling its
  internal Python API directly (bypassing the CLI's error-swallowing
  wrapper) to surface the real error: a `401 Unauthorized` from Kaggle's
  server, caused by a stale, previously-valid `kaggle.json` that the CLI was
  still trying (and Kaggle allows only one live API token per account, so
  the OAuth login had silently invalidated it).
- Fix: delete the stale `kaggle.json`, generate a fresh token, save it to
  `~/.kaggle/access_token` — the original approach, done right.

**3. Content-based model could recommend an item as "similar to itself."**
Caught during the first synthetic-data smoke test, before any real data was
downloaded: `ContentModel.recommend_similar` never excluded its own seed
items from the candidate pool, so the highest-scoring "recommendation" for
an item was often the item itself (cosine similarity 1.0 with itself).
Fixed by excluding seed IDs, not just the caller-supplied exclusion set.

**4. The download script silently skipped unzipping.** `set -euo pipefail`
combined with a `kaggle` CLI call that can exit non-zero on a benign warning
(even after a successful download) meant the script aborted before its own
unzip step ran. Fixed by removing `-e` and checking for the expected output
file explicitly per download, rather than trusting the tool's exit code.

**5. The Dockerfile would have failed on a real deploy.** It `COPY`s
`artifacts/` and `data/processed/` into the image, but both were fully
gitignored — a Render/Fly build pulling straight from GitHub would have
failed at that step with no local training run first. Fixed with a
targeted `.gitignore` carve-out: only the ~118MB of files the API actually
reads at serving time are committed (4 model artifacts + the 2 parquet
files it loads), not the full ~3.7GB raw dataset or the training-only
files.

## Talking points this project earns in an interview

- "Tell me about a time a model didn't work" — this project has a real,
  specific, quantified answer instead of a hypothetical.
- Cold-start is answered architecturally (tiered routing by history length),
  not hand-waved.
- The evaluation methodology (paired bootstrap, not a single point estimate)
  generalizes directly to reading a real live A/B test result.
- The debugging process itself (Kaggle auth, the Dockerfile gap) is evidence
  of actually operating the system end to end, not just training a model in
  a notebook.

## Links

- Repo: https://github.com/Ryan9828/ShelfSense
- Live API: _pending deployment_
- Live demo: _pending deployment_
