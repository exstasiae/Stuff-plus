"""
src/data/preprocess.py
─────────────────────────────────────────────────────────────────────────────
Clean and standardize raw Statcast data before feature engineering.

Steps (in order):
  1.  Normalize pitch type codes (e.g. legacy "FT" → "SI").
  2.  Parse game_date to datetime — done early so filters can use season year.
  3.  Drop rows missing critical pitch-characteristic fields.
  4.  Apply velocity floors per pitch type to remove position players /
      Statcast tracking errors.
  5.  Filter pitcher-seasons with fewer than min_pitches_per_season — removes
      position players who appear too rarely to have meaningful data and whose
      unusual feature combinations contaminate the model.
  6.  Compute lw_run_value — the linear-weight run value target.
  7.  Drop rows where lw_run_value could not be computed.
  8.  Assign each pitch to a pitch group (fastball / breaking / offspeed).
  9.  Drop pitches with unrecognised type codes.
  10. Clip extreme outliers (sensor / entry errors).
  11. Encode pitcher handedness as 0/1.

TARGET: lw_run_value
────────────────────
We replaced the raw `delta_run_exp` target with a *linear-weight run value*
computed from the pitch event type.  Here is why this matters:

  delta_run_exp problems
  ──────────────────────
  • ~85% of pitches mid-at-bat have delta_run_exp ≈ 0.000 (the game state
    barely changed on ball 2 of a 3-2 count).  Only the last pitch of each
    at-bat has a large value, creating a highly imbalanced, noisy label.
  • The value is heavily situational: the same excellent curveball thrown for
    a called strike gets different delta_run_exp depending on whether the
    bases are loaded or empty.

  lw_run_value advantages
  ───────────────────────
  • Every pitch gets a non-trivial label:
      swinging strike   → −0.130  (great for pitcher)
      called strike     → −0.068
      foul              → −0.038
      ball              → +0.056
      ball in play      → (estimated_woba − league_woba) / woba_scale
  • Situational noise is stripped out — the same pitch quality always gets
    the same base weight, regardless of count or base state.
  • For balls in play we use estimated_woba_using_speedangle (xwOBA), which
    removes defensive variance: a hard-hit lineout is still punished, and a
    well-placed pitch that generates weak contact is still rewarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Pitch type normalization ──────────────────────────────────────────────────
_PITCH_TYPE_MAP: dict[str, str | None] = {
    "FT": "SI",   # two-seam fastball → sinker
    "PO": "FF",   # pitchout → 4-seam (rare; drop via velocity floor anyway)
    "EP": "CH",   # eephus → changeup
    "SC": "CH",   # screwball → changeup
    "AB": None,   # automatic ball — drop
    "IN": None,   # intentional ball — drop
    "FA": "FF",   # ambiguous fastball → 4-seam
}

# ── Required pitch-characteristic columns ────────────────────────────────────
# Pitches missing any of these cannot be used as training examples.
_REQUIRED_COLUMNS = [
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "pitch_type",
    "description",   # needed to compute lw_run_value
]

# ── Linear weight run values per pitch event ─────────────────────────────────
# Each event type maps to an approximate run value (positive = good for batter,
# negative = good for pitcher).  None = ball in play, handled via xwOBA below.
#
# Values based on Tom Tango's linear weight methodology as applied in
# Baseball Prospectus / FanGraphs pitch-level run value work.
_LINEAR_WEIGHTS: dict[str, float | None] = {
    # ── Balls ─────────────────────────────────────────────────────────────────
    "ball":                      0.056,
    "blocked_ball":              0.056,
    "pitchout":                  0.056,
    "intent_ball":               0.056,

    # ── Called strikes ────────────────────────────────────────────────────────
    "called_strike":            -0.068,

    # ── Swinging strikes ──────────────────────────────────────────────────────
    # foul_tip counts as a swinging strike only if the catcher holds it with
    # two strikes; in practice it's almost always a strike event so we treat
    # it identically.
    "swinging_strike":          -0.130,
    "swinging_strike_blocked":  -0.130,
    "missed_bunt":              -0.130,
    "foul_tip":                 -0.130,

    # ── Foul balls ────────────────────────────────────────────────────────────
    # A foul can never record the third strike (except foul_tip), so it is
    # worth roughly half a called strike on average across all counts.
    "foul":                     -0.038,
    "foul_bunt":                -0.038,
    "foul_pitchout":            -0.038,

    # ── Hit by pitch ──────────────────────────────────────────────────────────
    "hit_by_pitch":              0.350,

    # ── Balls in play ─────────────────────────────────────────────────────────
    # Value computed separately using estimated_woba_using_speedangle.
    # None is a sentinel — do not change to 0.
    "hit_into_play":             None,
    "hit_into_play_no_out":      None,
    "hit_into_play_score":       None,
}

# wOBA-to-run-value conversion for balls in play.
# rv_bip = (xwOBA − WOBA_MEAN) / WOBA_SCALE
# A pitch generating league-average contact (xwOBA = 0.320) → rv = 0.
# A barrel (xwOBA ≈ 1.200) → rv ≈ +0.76 (bad for pitcher).
# A weak grounder (xwOBA ≈ 0.050) → rv ≈ −0.23 (good for pitcher).
_WOBA_MEAN  = 0.320   # approximate league-average wOBA (2017–2025 era)
_WOBA_SCALE = 1.157   # wOBA → runs conversion factor

# ── Count leverage weights ───────────────────────────────────────────────────
# Each pitch outcome is multiplied by the leverage of the count in which it
# was thrown.  A swinging strike on 3-2 (leverage 2.16) is worth much more
# than one on 0-2 (leverage 0.75) because the former can end the PA while
# the latter merely runs the count full.
#
# Values are approximate count-based leverage indices derived from RE24
# (run-expectancy 24 tables) and PA-completion probability tables.
# Reference: Tango / Lichtman / Dolphin "The Book" + public WOBA count work.
_COUNT_LEVERAGE: dict[tuple[int, int], float] = {
    # (balls, strikes) → leverage multiplier (normalised so league mean ≈ 1.0)
    (0, 0): 0.87,
    (1, 0): 0.99,
    (2, 0): 1.01,
    (3, 0): 0.67,   # batters usually take; pitcher has limited leverage
    (0, 1): 0.78,
    (1, 1): 0.99,
    (2, 1): 1.25,
    (3, 1): 1.44,
    (0, 2): 0.75,
    (1, 2): 1.00,
    (2, 2): 1.35,
    (3, 2): 2.16,   # highest leverage — every pitch can end the AB
}
# Pre-compute as a 12-element array indexed by balls*3 + strikes (0-11)
_LEVERAGE_ARRAY: list[float] = [
    _COUNT_LEVERAGE.get((b, s), 1.0)
    for b in range(4) for s in range(3)
]

# ── Velocity floors per pitch type ───────────────────────────────────────────
# Pitches below these velocities are almost certainly position players pitching
# in blowouts, or Statcast radar tracking errors.  Values in mph.
#
# Rationale per type:
#   Fastball family: even the slowest MLB starter throws 80+ on a 4-seam.
#   Breaking balls:  a real slider is 65+; below that it's an eephus or error.
#   Offspeed:        changeups/splitters sit 65+ for true pitchers.
_VELOCITY_FLOORS: dict[str, float] = {
    "FF": 80.0,   # 4-seam fastball
    "SI": 78.0,   # sinker
    "FC": 75.0,   # cutter
    "SL": 65.0,   # slider
    "CU": 55.0,   # curveball (some elite yakkers sit 68–72, floor is conservative)
    "KC": 55.0,   # knuckle-curve
    "ST": 65.0,   # sweeper
    "SV": 58.0,   # slurve
    "CH": 65.0,   # changeup
    "FS": 63.0,   # splitter
    "FO": 60.0,   # forkball
}

# ── Physical clip bounds (sensor / entry error guard) ────────────────────────
_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "release_speed":      (50.0,  105.0),
    "pfx_x":             (-30.0,   30.0),
    "pfx_z":             (-30.0,   30.0),
    "release_spin_rate":  (500.0, 3500.0),
    "release_pos_x":      (-5.0,    5.0),
    "release_pos_z":       (2.5,    8.5),
    "release_extension":   (3.0,    8.0),
    "plate_x":            (-3.0,    3.0),
    "plate_z":            (-1.0,    6.5),
}


# ── Target computation ────────────────────────────────────────────────────────

def compute_linear_weight_rv(df: pd.DataFrame) -> pd.Series:
    """
    Compute a linear-weight run value for every pitch in df.

    Returns
    -------
    pd.Series (float, same index as df)
        NaN for pitches whose description is unrecognised (these will be
        dropped by the caller as missing-target rows).

    Algorithm
    ---------
    1. Map the `description` column to the fixed linear weight in
       _LINEAR_WEIGHTS.  Event types that resolve to None (balls in play)
       are left as NaN at this stage.
    2. For balls in play (type == 'X'), replace NaN with:
           (estimated_woba_using_speedangle − 0.320) / 1.157
       If xwOBA is itself NaN (launch data missing), fall back to 0.0
       (league-average contact quality, neutral run impact).
    """
    # Step 1: map event-type weights (skip None sentinel values)
    weight_map = {k: v for k, v in _LINEAR_WEIGHTS.items() if v is not None}
    rv = df["description"].map(weight_map)   # NaN for unknown + BIP events

    # Step 2: balls in play via xwOBA
    if "type" in df.columns:
        bip_mask = df["type"] == "X"
    else:
        # Fallback if 'type' column missing
        bip_mask = df["description"].isin(
            {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
        )

    if bip_mask.any():
        if "estimated_woba_using_speedangle" in df.columns:
            xwoba    = df.loc[bip_mask, "estimated_woba_using_speedangle"]
            bip_rv   = (xwoba - _WOBA_MEAN) / _WOBA_SCALE
            rv.loc[bip_mask] = bip_rv.fillna(0.0)
        else:
            # No xwOBA available — assign neutral (0) for all BIP
            logger.warning(
                "estimated_woba_using_speedangle not found; "
                "assigning rv=0 for all balls in play."
            )
            rv.loc[bip_mask] = 0.0

    n_unknown = rv.isna().sum() - bip_mask.sum()   # BIP NaN were just filled
    if n_unknown > 0:
        unknown_descs = df.loc[rv.isna(), "description"].value_counts().head(10)
        logger.debug(
            "%d pitches have unrecognised description (will be dropped):\n%s",
            n_unknown, unknown_descs.to_string()
        )

    return rv


# ── Per-season pitcher filter ─────────────────────────────────────────────────

def _filter_min_pitches_per_season(df: pd.DataFrame,
                                   min_pitches: int) -> pd.DataFrame:
    """
    Drop every pitch belonging to a pitcher-season where the pitcher threw
    fewer than `min_pitches` total pitches that season.

    Requires game_date to already be a datetime column.
    """
    df = df.copy()
    df["_season"] = df["game_date"].dt.year

    # Count pitches per (pitcher, season) and broadcast back to each row
    df["_season_count"] = df.groupby(
        ["pitcher", "_season"]
    )["pitcher"].transform("count")

    before = len(df)
    df = df[df["_season_count"] >= min_pitches]
    dropped = before - len(df)

    if dropped:
        logger.info(
            "Position-player filter: dropped %d pitches from pitcher-seasons "
            "with < %d pitches (%.1f%% of data)",
            dropped, min_pitches, 100 * dropped / before
        )

    return df.drop(columns=["_season", "_season_count"])


# ── Velocity floor filter ─────────────────────────────────────────────────────

def _filter_velocity_floors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop pitches whose velocity is below the realistic minimum for that
    pitch type.  Targets two populations:
      • Position players pitching (e.g. 55 mph lob changeups)
      • Statcast radar tracking errors (occasional sub-60 mph readings)
    """
    mask = pd.Series(False, index=df.index)
    for pt, floor in _VELOCITY_FLOORS.items():
        mask |= (df["pitch_type"] == pt) & (df["release_speed"] < floor)

    n_dropped = mask.sum()
    if n_dropped:
        logger.info(
            "Velocity floor filter: dropped %d pitches below per-type minimums",
            n_dropped
        )
        # Log a breakdown so it's easy to see which types were hit
        breakdown = (
            df.loc[mask]
            .groupby("pitch_type")["release_speed"]
            .agg(["count", "max"])
            .rename(columns={"count": "n_dropped", "max": "max_velo_dropped"})
        )
        logger.debug("Velocity floor breakdown:\n%s", breakdown.to_string())

    return df[~mask]


# ── Main clean function ───────────────────────────────────────────────────────

def clean(df: pd.DataFrame,
          cfg: dict | None = None,
          config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """
    Apply all cleaning steps and return a training-ready DataFrame with a
    `lw_run_value` target column and a `pitch_group` label column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Statcast data (output of fetch.load_seasons).
    cfg : dict, optional
        Parsed config.yaml.  Loaded automatically if None.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame, reset index.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    min_pitches = (
        cfg.get("preprocessing", {}).get("min_pitches_per_season", 100)
    )

    n_start = len(df)
    logger.info("Preprocessing: starting with %d rows", n_start)

    df = df.copy()

    # ── Step 1: Normalize pitch type codes ────────────────────────────────────
    df["pitch_type"] = df["pitch_type"].map(
        lambda pt: _PITCH_TYPE_MAP.get(pt, pt) if isinstance(pt, str) else pt
    )
    df = df[df["pitch_type"].notna()]

    # ── Step 2: Parse game_date early (needed by the season filter below) ─────
    if "game_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # ── Step 3: Drop rows missing required pitch-characteristic columns ────────
    present_required = [c for c in _REQUIRED_COLUMNS if c in df.columns]
    missing_required = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        logger.warning("Required columns not found in data: %s", missing_required)
    n_before = len(df)
    df = df.dropna(subset=present_required)
    logger.info("Dropped %d rows missing required columns", n_before - len(df))

    # ── Step 4: Velocity floors — removes position players + tracking errors ──
    df = _filter_velocity_floors(df)

    # ── Step 5: Per-season pitch count filter — removes remaining position players
    if "game_date" in df.columns and "pitcher" in df.columns:
        df = _filter_min_pitches_per_season(df, min_pitches=min_pitches)
    else:
        logger.warning(
            "Cannot apply per-season pitcher filter: "
            "missing 'game_date' or 'pitcher' column."
        )

    # ── Step 6: Compute linear-weight run value target ────────────────────────
    df["lw_run_value"] = compute_linear_weight_rv(df)

    # ── Step 6b: Scale by count leverage ─────────────────────────────────────
    # A swinging strike on 3-2 counts for more than one on 0-2 because it
    # ends the AB.  Multiply each pitch's run value by the count's leverage
    # index so the model rewards pitchers who execute in high-leverage counts.
    if "balls" in df.columns and "strikes" in df.columns:
        import numpy as _np
        _leverage_arr = _np.array(_LEVERAGE_ARRAY)
        balls_int   = df["balls"].fillna(0).astype(int).clip(0, 3)
        strikes_int = df["strikes"].fillna(0).astype(int).clip(0, 2)
        count_idx   = (balls_int * 3 + strikes_int).values          # 0-11
        df["lw_run_value"] = df["lw_run_value"] * _leverage_arr[count_idx]
    else:
        logger.warning("balls/strikes columns missing — skipping count leverage")

    # ── Step 7: Drop rows where target could not be computed ──────────────────
    n_before = len(df)
    df = df.dropna(subset=["lw_run_value"])
    n_dropped_target = n_before - len(df)
    if n_dropped_target:
        logger.info(
            "Dropped %d rows with uncomputable lw_run_value "
            "(unrecognised description values)",
            n_dropped_target
        )

    # ── Step 8: Assign pitch group ────────────────────────────────────────────
    group_map: dict[str, str] = {}
    for group, types in cfg["pitch_groups"].items():
        for pt in types:
            group_map[pt] = group
    df["pitch_group"] = df["pitch_type"].map(group_map)

    # ── Step 9: Drop unrecognised pitch types ─────────────────────────────────
    unknown_mask = df["pitch_group"].isna()
    if unknown_mask.any():
        unknown_types = df.loc[unknown_mask, "pitch_type"].value_counts()
        logger.info(
            "Dropping %d pitches with unrecognised pitch type:\n%s",
            unknown_mask.sum(), unknown_types.to_string()
        )
    df = df[~unknown_mask]

    # ── Step 10: Clip physical outliers (sensor / entry errors) ───────────────
    for col, (lo, hi) in _CLIP_BOUNDS.items():
        if col in df.columns:
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_clipped:
                logger.info("Clipping %d extreme values in '%s'", n_clipped, col)
            df[col] = df[col].clip(lo, hi)

    # ── Step 11: Encode pitcher handedness ────────────────────────────────────
    if "p_throws" in df.columns:
        df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)

    n_end = len(df)
    logger.info(
        "Preprocessing complete: %d → %d rows (dropped %d, %.1f%%)\n"
        "  Target: lw_run_value  mean=%.4f  std=%.4f",
        n_start, n_end, n_start - n_end,
        100 * (n_start - n_end) / n_start,
        df["lw_run_value"].mean(),
        df["lw_run_value"].std(),
    )

    return df.reset_index(drop=True)


# ── Parquet save / load ───────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame,
                   cfg: dict | None = None,
                   config_path: str | Path = "config.yaml",
                   filename: str = "statcast_clean.parquet",
                   compression: str = "snappy") -> Path:
    """Save the cleaned DataFrame to data/processed/ as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if cfg is None:
        cfg = _load_config(config_path)

    out_dir  = Path(cfg["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression=compression, row_group_size=200_000)

    logger.info("Saved processed data → %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)
    return out_path


def load_processed(cfg: dict | None = None,
                   config_path: str | Path = "config.yaml",
                   filename: str = "statcast_clean.parquet",
                   columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned Parquet file from data/processed/."""
    import pyarrow.parquet as pq

    if cfg is None:
        cfg = _load_config(config_path)

    path = Path(cfg["paths"]["processed_data"]) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            "Run preprocess.clean() and preprocess.save_processed() first."
        )
    return pq.read_table(path, columns=columns).to_pandas()
