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
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/raw
cd data/raw

kaggle competitions download -c h-and-m-personalized-fashion-recommendations \
  -f articles.csv -p .
kaggle competitions download -c h-and-m-personalized-fashion-recommendations \
  -f customers.csv -p .
kaggle competitions download -c h-and-m-personalized-fashion-recommendations \
  -f transactions_train.csv -p .

for f in articles.csv customers.csv transactions_train.csv; do
  if [ -f "$f.zip" ]; then
    unzip -o "$f.zip" && rm "$f.zip"
  fi
done

echo "Downloaded to data/raw/. Next: python -m shelfsense.build_features"
