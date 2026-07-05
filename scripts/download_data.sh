#!/usr/bin/env bash
# Downloads the H&M Personalized Fashion Recommendations dataset (tabular files only —
# the images.zip is ~25GB and is not needed for this project's collaborative/content models).
#
# Prerequisites (one-time, manual — cannot be scripted, needs your Kaggle account):
#   1. Create a Kaggle account and accept the competition rules at:
#      https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
#   2. Authenticate the Kaggle CLI (v2.x) with ONE of:
#        kaggle auth login                      # OAuth, recommended — caches creds for you
#        export KAGGLE_API_TOKEN=<token>        # from kaggle.com/settings/api "Generate New Token"
#        echo <token> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
# This script doesn't check which one you used — it just calls `kaggle`, which will print
# its own auth error (with these same options) if none of the above is set up yet.
set -uo pipefail  # not -e: `kaggle` can exit non-zero on a benign warning even after a good download

cd "$(dirname "$0")/.."

mkdir -p data/raw
cd data/raw

fetch() {
  local f="$1"
  echo "Downloading $f..."
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f "$f" -p .
  if [ -f "$f.zip" ]; then
    unzip -o "$f.zip" && rm "$f.zip"
  fi
  if [ ! -f "$f" ]; then
    echo "ERROR: $f did not end up in data/raw/ — check the kaggle output above." >&2
    exit 1
  fi
}

fetch articles.csv
fetch customers.csv
fetch transactions_train.csv

echo "Downloaded to data/raw/. Next: python -m shelfsense.build_features"
