"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Full end-to-end Stuff+ pipeline:
  fetch → preprocess → features → aggregate → train → evaluate

Run the full pipeline:
    python pipeline.py

Run specific steps:
    python pipeline.py --steps fetch preprocess
    python pipeline.py --steps aggregate train evaluate --val-season 2025
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step 1: Fetch ─────────────────────────────────────────────────────────────

def step_fetch(cfg: dict, args: argparse.Namespace) -> None:
    from src.data.fetch import load_seasons, cache_summary
    logger.info("── STEP 1: FETCH ────────────────────────────────────────")
    seasons = args.seasons or cfg["seasons"]["train"]
    load_seasons(seasons=seasons, cfg=cfg, include_current=args.include_current)
    cache_summary(cfg)


# ── Step 2: Preprocess ────────────────────────────────────────────────────────

def step_preprocess(cfg: dict, args: argparse.Namespace) -> None:
    from src.data.fetch import load_seasons
    from src.data.preprocess import clean, save_processed
    logger.info("── STEP 2: PREPROCESS ───────────────────────────────────")

    seasons = args.seasons or cfg["seasons"]["train"]
    df = load_seasons(seasons=seasons, cfg=cfg, include_current=args.include_current)
    df = clean(df, cfg=cfg)
    save_processed(df, cfg=cfg, filename="statcast_clean.parquet")


# ── Step 3: Feature engineering ───────────────────────────────────────────────

def step_features(cfg: dict, args: argparse.Namespace) -> None:
    from src.data.preprocess import load_processed
    from src.features.engineer import build_features
    logger.info("── STEP 3: FEATURE ENGINEERING ──────────────────────────")

    df = load_processed(cfg=cfg, filename="statcast_clean.parquet")
    df = build_features(df, cfg=cfg)

    out_path = Path(cfg["paths"]["processed_data"]) / "statcast_features.parquet"
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        out_path,
        compression="snappy",
        row_group_size=200_000,
    )
    logger.info("Saved pitch-level features → %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)


# ── Step 4: Aggregate ─────────────────────────────────────────────────────────

def step_aggregate(cfg: dict, args: argparse.Namespace) -> None:
    """
    Collapse pitch-level feature data to pitcher-season-pitch_type level.

    Input:  data/processed/statcast_features.parquet  (~300 MB, millions of rows)
    Output: data/processed/statcast_aggregated.parquet (~1 MB, thousands of rows)

    Each output row represents one pitcher's season-long profile for a single
    pitch type, with averaged physics features and mean lw_run_value as target.
    n_pitches is retained as the sample weight for XGBoost training.
    """
    from src.data.preprocess import load_processed
    from src.features.aggregate import aggregate_to_pitch_type_level, save_aggregated
    logger.info("── STEP 4: AGGREGATE ────────────────────────────────────")

    df = load_processed(cfg=cfg, filename="statcast_features.parquet")
    agg = aggregate_to_pitch_type_level(df, cfg=cfg)
    save_aggregated(agg, cfg=cfg)


# ── Step 5: Train ─────────────────────────────────────────────────────────────

def step_train(cfg: dict, args: argparse.Namespace) -> None:
    from src.models.train import run_training
    logger.info("── STEP 5: TRAIN ────────────────────────────────────────")
    run_training(
        groups=args.groups,
        val_season=args.val_season,
        cfg=cfg,
    )


# ── Step 6: Evaluate ──────────────────────────────────────────────────────────

def step_evaluate(cfg: dict, args: argparse.Namespace) -> None:
    """
    Load aggregated data, score every pitcher-season-pitch_type row with the
    per-pitch-type model for that row, then print leaderboards and save
    CSVs + diagnostic plots.
    """
    import numpy as np
    from src.features.aggregate import load_aggregated
    from src.models.evaluate import (
        predict_stuff_plus,
        build_leaderboard,
        save_leaderboard,
        plot_feature_importance,
        plot_stuff_plus_distribution,
        validate_whiff_rate_correlation,
    )
    logger.info("── STEP 6: EVALUATE ─────────────────────────────────────")

    df = load_aggregated(cfg=cfg)
    df = predict_stuff_plus(df, cfg=cfg)

    # Collect all trained pitch types so we know what's available
    model_pitch_types_cfg = cfg.get("model_pitch_types", {})
    all_pitch_types: list[str] = (
        model_pitch_types_cfg.get("standard", [])
        + model_pitch_types_cfg.get("small", [])
    )

    # ── Full leaderboard (all pitch types, most recent season) ────────────────
    current_season = int(df["season"].max()) if "season" in df.columns else None
    df_recent = df[df["season"] == current_season] if current_season else df

    lb_all = build_leaderboard(df_recent, min_pitches=50)
    save_leaderboard(lb_all, "stuff_plus_all.csv", cfg)
    print(f"\nTop 20 pitches by Stuff+ ({current_season} season):")
    print(lb_all.head(20).to_string(index=False))

    # ── Per-pitch-type leaderboards (most recent season) ──────────────────────
    for pt in all_pitch_types:
        lb_pt = build_leaderboard(df_recent, pitch_type=pt, min_pitches=50)
        if len(lb_pt) > 0:
            save_leaderboard(lb_pt, f"stuff_plus_{pt}.csv", cfg)

    # ── Per-group leaderboards (most recent season) ───────────────────────────
    for group in cfg["pitch_groups"]:
        lb_g = build_leaderboard(df_recent, group=group, min_pitches=50)
        if len(lb_g) > 0:
            save_leaderboard(lb_g, f"stuff_plus_{group}.csv", cfg)

    # ── Multi-season career leaderboard (pitch-count-weighted average) ────────
    if "n_pitches" in df.columns:
        career_rows = (
            df.dropna(subset=["stuff_plus"])
            .groupby(["pitcher", "player_name", "pitch_type"])
            .apply(
                lambda g: (g["stuff_plus"] * g["n_pitches"]).sum() / g["n_pitches"].sum(),
                include_groups=False,
            )
            .reset_index(name="stuff_plus_career")
        )
        career_rows["stuff_plus_career"] = career_rows["stuff_plus_career"].round(1)
        # Attach total pitch count
        total_pitches = (
            df.groupby(["pitcher", "player_name", "pitch_type"])["n_pitches"]
            .sum()
            .reset_index(name="total_pitches")
        )
        career_rows = career_rows.merge(
            total_pitches, on=["pitcher", "player_name", "pitch_type"], how="left"
        )
        career_rows = career_rows.sort_values("stuff_plus_career", ascending=False)
        save_leaderboard(career_rows, "stuff_plus_career.csv", cfg)

    # ── Feature importance plots ───────────────────────────────────────────────
    for pt in all_pitch_types:
        try:
            plot_feature_importance(pt, cfg, save=True)
        except FileNotFoundError:
            logger.warning("No model for pitch type '%s' — skipping importance plot", pt)

    # ── Whiff / CSW rate validation ───────────────────────────────────────────
    validate_whiff_rate_correlation(df, cfg=cfg, save=True)

    # ── Distribution plot (all seasons) ───────────────────────────────────────
    plot_stuff_plus_distribution(df, cfg=cfg, save=True)


# ── Step registry ─────────────────────────────────────────────────────────────

STEPS: dict[str, callable] = {
    "fetch":      step_fetch,
    "preprocess": step_preprocess,
    "features":   step_features,
    "aggregate":  step_aggregate,
    "train":      step_train,
    "evaluate":   step_evaluate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stuff+ pipeline")
    parser.add_argument(
        "--steps", nargs="+",
        choices=list(STEPS.keys()),
        default=list(STEPS.keys()),
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        help="Override training seasons from config",
    )
    parser.add_argument(
        "--groups", nargs="+",
        help="Pitch types to train, e.g. FF SL CH (default: all from config)",
    )
    parser.add_argument(
        "--val-season", type=int,
        help="Season to hold out for validation (default: most recent)",
    )
    parser.add_argument(
        "--include-current", action="store_true",
        help="Also fetch current (2026) season data",
    )
    args = parser.parse_args()

    cfg = load_config()

    for step_name in args.steps:
        STEPS[step_name](cfg, args)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
