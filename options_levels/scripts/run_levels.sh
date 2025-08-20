#!/usr/bin/env bash
set -euo pipefail

mkdir -p out raw artifacts dist

# NDX (your tuned command)
python src/options_levels.py --symbols NDX --out out --raw raw --raw-ladders raw --basis "" --wall-fallback on --em-mult 3.0 --em-floor-ndx 250 --smooth-win 3 --vt-inner 0.30 --vt-outer 1.55 --oi-floor-ndx 350 --plot-count-ndx 16 --min-spacing-ndx 40 --lg-fwhm-frac 0.78 --wall-fwhm-frac 0.65 --box-hw-cap-ndx 15 --box-hw-frac-em-ndx 0.08

# SPX (your tuned command)
python src/options_levels.py --symbols SPX --out out --raw raw --raw-ladders raw --basis "" --wall-fallback on --em-mult 3.0 --em-floor-spx 100 --smooth-win 3 --vt-inner 0.30 --vt-outer 1.55 --oi-floor-spx 200 --plot-count-spx 16 --min-spacing-spx 10 --lg-fwhm-frac 0.78 --wall-fwhm-frac 0.65 --box-hw-cap-spx 2.5 --box-hw-frac-em-spx 0.07

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
zip -rq artifacts/levels_${STAMP}.zip out
cp -f out/levels_plot_*.csv out/levels_pine_*.txt dist/ || true
