#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# Run both symbols
bash configs/spx.cmd
bash configs/ndx.cmd

# Zip outputs for the run
mkdir -p artifacts
zip -rq artifacts/levels_${STAMP}.zip out

# Keep just the distributables (don’t ship raw by default)
mkdir -p dist
cp -f out/levels_plot_*.csv out/levels_pine_*.txt dist/
