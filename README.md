# Stuff+

A pitch-quality metric modeled on Fangraphs' Stuff+ concept. One XGBoost model is trained per pitch type (FF, SI, SL, CH, CU, FC, ST, FS, KC, SV), using only pure pitch-physics features — velocity, movement, spin, release point, approach angle, tunneling — with no location or command signals. Scores are normalized to a 100-scale: 100 = league average, 110 = one standard deviation above average.

## How it works

1. **Fetch** — Downloads Statcast pitch-level data (2021–2025) via [pybaseball](https://github.com/jldbc/pybaseball) and caches it locally as Parquet files.
2. **Preprocess** — Filters to qualified pitchers, computes a linear-weight run value target (`lw_run_value`) from pitch outcomes.
3. **Feature engineering** — Derives induced vertical break, horizontal break, vertical/horizontal approach angles, spin efficiency, release consistency, and pitch tunneling distances. Each feature is also z-scored within (pitch type × season) to produce era-stable relative features.
4. **Aggregate** — Collapses millions of pitch rows to pitcher-season-pitch_type level (~thousands of rows). This is the training unit: each row is one pitcher's average physics profile for a single pitch type in one season.
5. **Train** — Fits one XGBoost regressor per pitch type, with pitch-count sample weights. Standard and small-sample hyperparameter sets handle common vs. rare pitch types.
6. **Evaluate** — Scores every pitcher-season-pitch_type row, normalizes to 100-scale, and exports leaderboard CSVs and plots to `outputs/`.

## Project structure

```
Stuff+/
├── pipeline.py              # End-to-end CLI entry point
├── config.yaml              # All hyperparameters, paths, and feature lists
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── fetch.py         # Statcast download + Parquet cache
│   │   └── preprocess.py    # Cleaning, run-value target construction
│   ├── features/
│   │   ├── engineer.py      # Per-pitch feature derivation
│   │   ├── aggregate.py     # Pitch-level → pitcher-season-type aggregation
│   │   └── tunnel.py        # Tunneling distance / ratio computation
│   └── models/
│       ├── train.py         # XGBoost training loop
│       └── evaluate.py      # Scoring, normalization, leaderboards, plots
├── models/                  # Trained model artifacts (JSON) — one folder per pitch type
├── notebooks/
│   └── 01_quickstart.ipynb  # Interactive walkthrough
├── data/
│   ├── raw/                 # Cached Statcast Parquet files (gitignored — ~600 MB)
│   └── processed/           # Intermediate pipeline outputs (mostly gitignored)
└── outputs/
    ├── leaderboards/        # Per-pitch-type and group CSVs
    └── plots/               # Feature importance and distribution PNGs
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (fetches data on first run — takes a while)
python pipeline.py

# 3. Run specific steps only
python pipeline.py --steps fetch preprocess
python pipeline.py --steps aggregate train evaluate --val-season 2025

# 4. Train on specific pitch types
python pipeline.py --steps train evaluate --groups FF SL CH

# 5. Include the current (in-progress) season
python pipeline.py --include-current
```

Outputs land in `outputs/leaderboards/` (CSV) and `outputs/plots/` (PNG).

If you have the pre-trained models checked in (`models/`) and the aggregated data file (`data/processed/statcast_aggregated.parquet`), you can skip straight to scoring:

```bash
python pipeline.py --steps evaluate
```

## Feature design decisions

**No location features** — `plate_x` and `plate_z` are intentionally excluded. Including them would penalize pitchers with poor command: a 102 mph fastball that misses its spot would score lower than an identical pitch that found the corner, conflating command with stuff quality.

**Era-stable z-scores** — Raw features are supplemented with within-(pitch_type × season) z-scores. A 98 mph fastball was elite in 2021; it's above-average in 2025. The z-score captures the same information era-stably.

**Pitch tunneling** — `tunnel_dist_nearest` measures how similar each pitch looks to the pitcher's nearest other pitch type at the batter's decision point (~23 ft from home plate). `tunnel_ratio_nearest` captures how much the pitches diverge after that point. These are NaN for one-pitch arsenals; XGBoost handles missing values natively.

**Aggregation before training** — Training on pitcher-season averages rather than individual pitches dramatically reduces noise and makes the metric reflect sustainable stuff quality rather than single-pitch variation.

## Training seasons

2021–2025 by default. Pre-2021 data is excluded because:
- 2020 was a 60-game COVID season with unusual conditions.
- Pre-2021 Trackman optical tracking is less accurate than 2021+ Hawk-Eye.
- Era drift (velocity inflation, sweeper proliferation) makes older comps less meaningful.

## Data source

All pitch data comes from [Baseball Savant](https://baseballsavant.mlb.com/) via the [pybaseball](https://github.com/jldbc/pybaseball) library. No credentials or API keys are required.

## Requirements

Python 3.10+. Key dependencies: `xgboost`, `scikit-learn`, `pandas`, `pyarrow`, `pybaseball`. See `requirements.txt` for pinned versions.
