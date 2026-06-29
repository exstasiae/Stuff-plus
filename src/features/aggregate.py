"""
src/features/aggregate.py
─────────────────────────────────────────────────────────────────────────────
Aggregate pitch-level engineered data to pitcher-season-pitch_type level.

WHY AGGREGATE?
──────────────
Training on individual pitches (~7M rows for 5 seasons) creates two problems:

  1. Noise: a single pitch outcome is dominated by luck (a 102 mph fastball
     thrown for ball 3 of a 3-2 count barely moves run expectancy regardless
     of how elite the pitch is). Averaging ~700 pitches per pitcher-season
     pitch type compresses this noise and exposes the true signal.

  2. If a pitcher has elite velocity but poor command,
     his individual pitches score inconsistently. Averaged features give a
     stable physics fingerprint; averaged lw_run_value reflects his true
     season-long performance with that pitch type across all outcomes.

OUTPUT SCHEMA (one row per pitcher-season-pitch_type)
─────────────────────────────────────────────────────
  pitcher       int        — Statcast pitcher ID
  player_name   str        — display name
  season        int        — season year
  pitch_type    str        — e.g. "FF", "SL"
  pitch_group   str        — "fastball" / "breaking" / "offspeed"
  p_throws_enc  int        — 1=RHP, 0=LHP
  n_pitches     int        — total pitches of this type (used as sample weight)
  <features>    float      — mean of every model input feature
  lw_run_value  float      — mean linear-weight run value (the training target)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.features.tunnel import add_tunnel_features, _TUNNEL_INTERMEDIATE_COLS

logger = logging.getLogger(__name__)

_AGGREGATED_FILENAME = "statcast_aggregated.parquet"


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def aggregate_to_pitch_type_level(
    df: pd.DataFrame,
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
) -> pd.DataFrame:
    """
    Collapse a pitch-level feature DataFrame to one row per
    (pitcher, season, pitch_type).

    Parameters
    ----------
    df : pd.DataFrame
        Output of src.features.engineer.build_features().
        Must contain game_date, pitcher, pitch_type, pitch_group,
        p_throws_enc, lw_run_value, and all model_input features.
    cfg : dict, optional
        Parsed config.yaml.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with n_pitches column and averaged features.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    min_pitches = cfg.get("aggregation", {}).get("min_pitches_per_type", 50)
    feature_cols = cfg["features"]["model_inputs"]
    target_col   = cfg["target"]   # "lw_run_value"

    # ── Extract season year from game_date ────────────────────────────────────
    df = df.copy()
    if "game_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["season"] = df["game_date"].dt.year.astype("Int64")
    elif "season" not in df.columns:
        raise ValueError(
            "DataFrame must contain either 'game_date' or 'season' column."
        )

    # ── Resolve feature columns present in data ───────────────────────────────
    # p_throws_enc is a group key — do not re-average it (it's 0 or 1 per pitcher)
    group_keys = ["pitcher", "season", "pitch_type", "pitch_group", "p_throws_enc"]
    cols_to_avg = [
        c for c in feature_cols
        if c in df.columns and c not in group_keys
    ]
    # z_* features are computed POST-aggregation — they don't exist in pitch-level
    # data yet, so suppress the warning for them specifically.
    missing_features = [
        c for c in feature_cols
        if c not in df.columns and not c.startswith("z_")
    ]
    if missing_features:
        logger.warning("Features not found in pitch-level DataFrame: %s", missing_features)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            "Run preprocess.clean() before aggregating."
        )

    # ── Pre-aggregate: binary outcome columns for validation metrics ─────────
    # Whiff = any swing-and-miss (bat misses the ball entirely)
    # CSW   = called strike + whiff (any pitch that takes a strike without contact)
    # These are NOT model features — they're validation targets to check that
    # high Stuff+ correlates with high whiff/CSW rates as expected.
    if "description" in df.columns:
        df = df.copy()
        df["_is_whiff"] = df["description"].isin(
            {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
        ).astype("float32")
        df["_is_csw"] = df["description"].isin(
            {"swinging_strike", "swinging_strike_blocked", "foul_tip",
             "missed_bunt", "called_strike"}
        ).astype("float32")
        validation_aggs: dict = {
            "whiff_rate": ("_is_whiff", "mean"),
            "csw_rate":   ("_is_csw",   "mean"),
        }
    else:
        logger.warning("'description' column missing — skipping whiff/CSW rate computation")
        validation_aggs = {}

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logger.info(
        "Aggregating %d pitch rows to pitcher-season-pitch_type level…", len(df)
    )

    # Release-point consistency: std dev in x and z directions (feet).
    # Computed inside the groupby — we derive the combined RMS metric after.
    rp_std_aggs: dict = {}
    if "release_pos_x" in df.columns and "release_pos_z" in df.columns:
        rp_std_aggs = {
            "_rp_std_x": ("release_pos_x", "std"),
            "_rp_std_z": ("release_pos_z", "std"),
        }

    # Tunnel intermediate columns: mean (x, z) at decision point and plate.
    # These are NOT model inputs — they're used by add_tunnel_features() to
    # compute pairwise cross-pitch distances, then dropped.
    tunnel_pos_aggs: dict = {
        col: (col, "mean")
        for col in _TUNNEL_INTERMEDIATE_COLS
        if col in df.columns
    }

    agg = (
        df.groupby(group_keys, observed=True)
        .agg(
            n_pitches    = (target_col, "count"),
            lw_run_value = (target_col, "mean"),
            **{col: (col, "mean") for col in cols_to_avg},
            **rp_std_aggs,
            **validation_aggs,
            **tunnel_pos_aggs,
        )
        .reset_index()
    )

    # Combine x/z std devs into a single release_consistency metric (RMS, feet).
    # Smaller = tighter release point = more deceptive.
    if "_rp_std_x" in agg.columns and "_rp_std_z" in agg.columns:
        agg["release_consistency"] = np.sqrt(
            agg["_rp_std_x"].fillna(0) ** 2 + agg["_rp_std_z"].fillna(0) ** 2
        )
        agg = agg.drop(columns=["_rp_std_x", "_rp_std_z"])

    # Attach display names (use most-common name per pitcher to handle typos)
    if "player_name" in df.columns:
        names = (
            df.groupby("pitcher")["player_name"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
            .rename("player_name")
        )
        agg = agg.join(names, on="pitcher")

    # ── Minimum pitch count filter ────────────────────────────────────────────
    n_before = len(agg)
    agg = agg[agg["n_pitches"] >= min_pitches].reset_index(drop=True)
    logger.info(
        "Aggregation complete: %d pitch-level rows → %d pitcher-season-pitch_type rows "
        "(removed %d with < %d pitches)",
        len(df), len(agg), n_before - len(agg), min_pitches,
    )

    # ── Drop rows where any required model feature is NaN ────────────────────
    # Validation-only and tunnel columns are excluded from this check:
    #   • whiff_rate / csw_rate / release_consistency: validation metrics
    #   • tunnel_dist_nearest / tunnel_ratio_nearest: NaN for single-pitch-type
    #     pitchers; XGBoost handles them natively (added by add_tunnel_features
    #     which runs after this filter)
    _non_required_cols = {
        "whiff_rate", "csw_rate", "release_consistency",
        "tunnel_dist_nearest", "tunnel_ratio_nearest",
    }
    present_features = [
        c for c in cols_to_avg
        if c in agg.columns and c not in _non_required_cols
    ]
    n_before = len(agg)
    agg = agg.dropna(subset=present_features + [target_col])
    if len(agg) < n_before:
        logger.info(
            "Dropped %d aggregated rows with NaN in features or target",
            n_before - len(agg),
        )

    # ── Tunneling features ────────────────────────────────────────────────────
    # Cross-pitch distance at decision point and divergence ratio.
    # Must run BEFORE z-scores so that z_tunnel_dist_nearest is generated.
    agg = add_tunnel_features(agg, cfg=cfg)

    # ── Within-type z-score features ─────────────────────────────────────────
    agg = add_within_type_zscores(agg, cfg=cfg)

    # ── Summary stats ─────────────────────────────────────────────────────────
    logger.info(
        "Final aggregated dataset: %d rows | "
        "target mean=%.4f  std=%.4f | "
        "avg pitches/row=%.0f",
        len(agg),
        agg[target_col].mean(),
        agg[target_col].std(),
        agg["n_pitches"].mean(),
    )
    logger.info(
        "Rows per pitch type:\n%s",
        agg.groupby("pitch_type")["n_pitches"]
           .agg(rows="count", total_pitches="sum")
           .sort_values("total_pitches", ascending=False)
           .to_string(),
    )

    return agg


def add_within_type_zscores(
    agg: pd.DataFrame,
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
) -> pd.DataFrame:
    """
    Add within-(pitch_type × season) z-score columns to the aggregated DataFrame.

    For each column listed in config aggregation.zscore_columns, compute:
        z_<col> = (value − group_mean) / group_std

    where the group is every pitcher who threw that pitch type in that season.
    This makes features era-stable and directly comparable across seasons:
    a 98 mph FF in 2021 (+1.2σ) is correctly scored higher than 98 mph in
    2025 (+0.3σ) when the league average has risen.

    Edge cases
    ──────────
    - Single pitcher in a group: std = NaN → z-score set to 0.0
    - All pitchers identical value: std = 0 → z-score set to 0.0
    Both cases are rare but handled gracefully rather than propagating NaN.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    zscore_cols: list[str] = cfg.get("aggregation", {}).get("zscore_columns", [])
    if not zscore_cols:
        logger.warning("aggregation.zscore_columns not set in config — skipping z-scores")
        return agg

    agg = agg.copy()

    for col in zscore_cols:
        if col not in agg.columns:
            logger.debug("Column '%s' absent from aggregated data — skipping z-score", col)
            continue

        grp  = agg.groupby(["pitch_type", "season"])[col]
        mean = grp.transform("mean")
        std  = grp.transform("std").fillna(0.0)   # NaN → 0 for single-pitcher groups

        agg[f"z_{col}"] = np.where(std > 0, (agg[col] - mean) / std, 0.0)

    added = [f"z_{c}" for c in zscore_cols if c in agg.columns]
    logger.info("Z-score features added (%d): %s", len(added), added)

    return agg


def save_aggregated(
    df: pd.DataFrame,
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
    compression: str = "snappy",
) -> Path:
    """Save the aggregated DataFrame to data/processed/statcast_aggregated.parquet."""
    if cfg is None:
        cfg = _load_config(config_path)

    out_dir  = Path(cfg["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _AGGREGATED_FILENAME

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression=compression)

    logger.info(
        "Saved aggregated data → %s  (%.2f MB,  %d rows)",
        out_path, out_path.stat().st_size / 1e6, len(df),
    )
    return out_path


def load_aggregated(
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the aggregated Parquet file from data/processed/."""
    if cfg is None:
        cfg = _load_config(config_path)

    path = Path(cfg["paths"]["processed_data"]) / _AGGREGATED_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregated file not found: {path}\n"
            "Run the aggregate step first:\n"
            "  python pipeline.py --steps aggregate"
        )
    return pq.read_table(path, columns=columns).to_pandas()
