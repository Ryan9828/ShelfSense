#!/usr/bin/env bash
# Downloads the H&M Personalized Fashion Recommendations dataset (tabular files only —
# the images.zip is ~25GB and is not needed for this project's collaborative/content models).
#
# Prerequisites (one-time, manual — cannot be scripted, needs your Kaggle account):
#   1. Create a Kaggle account and accept the competition rules at:
#      https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
#   2. Go to kaggle.com -> Account -> Create New API Token. This downloads kaggle.json.
#   3. Place it at ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "Missing ~/.kaggle/kaggle.json — see the header of this script for setup steps." >&2
  exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

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
