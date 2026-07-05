import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# --- subsampling (H&M's full transactions_train.csv is ~31M rows / 1.4M customers,
# too large to iterate on quickly on a laptop — a random customer sample preserves
# the interaction-sparsity and cold-start characteristics that make the eval realistic) ---
N_CUSTOMERS_SAMPLE = 60_000
SUBSAMPLE_SEED = 42

# --- time-based train/test split, mirroring the competition's own protocol:
# hold out the last week of transactions as "future" purchases to predict ---
HOLDOUT_DAYS = 7

# a customer needs at least this many *train* transactions before we trust
# collaborative-filtering scores for them; below this, hybrid falls back
# to content-based / popularity (see shelfsense.hybrid)
MIN_INTERACTIONS_FOR_CF = 3

TOP_K = 12  # H&M's own competition metric is MAP@12 — keep eval comparable

ALS_FACTORS = 64
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 15
ALS_ALPHA = 40  # confidence scaling on implicit counts, per Hu/Koren/Volinsky

N_BOOTSTRAP = 2000

# als.joblib is ~65MB in memory and is the single largest artifact loaded at serving
# time, for a model that's explicitly benchmarked-but-not-shipped (see docs/ab_test_results.md).
# On a memory-constrained deployment (Render's free tier is 512MB) it's not worth the
# room. Defaults to loaded (matches local dev, where memory isn't a constraint) —
# set LOAD_ITEM_CF=false as a Render environment variable to skip it there.
LOAD_ITEM_CF = os.environ.get("LOAD_ITEM_CF", "true").lower() != "false"
