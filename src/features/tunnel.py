"""
src/features/tunnel.py
─────────────────────────────────────────────────────────────────────────────
Tunneling feature computation for the Stuff+ model.

WHAT IS TUNNELING?
──────────────────
A pitch's deceptiveness depends not just on its own movement and velocity,
but on how similar it looks to the pitcher's other pitches until the moment
the batter must commit to a swing.

A batter has roughly 170 ms to decide swing/take. At 94 mph that decision
window closes when the ball is ~26.5 ft from home plate — the "tunnel point."
Two pitches that look identical at 26.5 ft but diverge sharply after it are
extremely hard to distinguish. Two pitches that look different from the start
give the batter more information.

       Release   Tunnel point    Plate
       ~55 ft     ~26.5 ft      ~1.4 ft
          ●──────────●─────────-─●
          |<-- batter gathers -->|<-- break happens -->|
                                ↑ decision committed here

FEATURES ADDED (per pitcher-season-pitch_type row)
────────────────────────────────────────────────────
  tunnel_dist_nearest  (ft)
      Euclidean distance at the tunnel point between this pitch type's
      mean trajectory and the nearest other pitch type the same pitcher
      threw that season.  LOWER = pitches look more similar = harder to read.

  tunnel_ratio_nearest  (dimensionless)
      (plate separation) / (tunnel separation) for that nearest pair.
      HIGHER = pitches that look the same at decision, then diverge most
      at the plate = maximum deception.  A ratio of 8 means the pitches
      are 8× further apart at the plate than at the decision point.

PIPELINE POSITION
─────────────────
  Step 1 – pitch level (called from engineer.py):
      compute_tunnel_pos(df)
      Adds tunnel_x, tunnel_z, plate_x_kin, plate_z_kin to each pitch row
      using Statcast kinematic equations. These are intermediate columns
      that get averaged at the aggregation step and then dropped.

  Step 2 – arsenal level (called from aggregate.py, after main groupby):
      add_tunnel_features(agg)
      Uses averaged tunnel/plate positions to compute pairwise distances
      across pitch types within each pitcher-season. Drops intermediate
      columns after use.

HANDLING SINGLE-PITCH-TYPE PITCHERS
─────────────────────────────────────
Relievers who throw only fastballs (no tunnel partner) get NaN for both
tunnel features. This is intentional: XGBoost's native NaN handling learns
the optimal branch direction for missing values, so these rows are kept in
training rather than dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Y coordinate (feet from back of home plate, Statcast convention) at the
# batter's decision point.  At 94 mph the ball is ~170 ms from the plate
# when it crosses this threshold — the minimum time a batter needs to swing.
TUNNEL_Y_FT: float = 26.5

# Y coordinate at the front of home plate (matches the value used in VAA/HAA).
PLATE_Y_FT: float = 1.417

# Intermediate column names added at pitch level and dropped after aggregation.
_TUNNEL_INTERMEDIATE_COLS: tuple[str, ...] = (
    "tunnel_x", "tunnel_z", "plate_x_kin", "plate_z_kin"
)


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Kinematic helpers ─────────────────────────────────────────────────────────

def _time_at_y(
    vy0: pd.Series,
    ay: pd.Series,
    release_pos_y: pd.Series,
    y_target: float,
) -> pd.Series:
    """
    Solve for the time t at which the pitch reaches y = y_target.

    Kinematic equation:
        y(t) = release_pos_y + vy0·t + ½·ay·t²  =  y_target
    Rearranged:
        ½·ay·t² + vy0·t + (release_pos_y − y_target) = 0

    Both roots of the quadratic are computed; the smaller positive root
    is returned (the ball reaching y_target on its way to the plate,
    not on a hypothetical return trip).

    Sign conventions (typical Statcast values):
        vy0 < 0   (ball flying toward plate, i.e. decreasing y)
        ay  > 0   (air drag, decelerating the ball)
    """
    a_coef = 0.5 * ay
    b_coef = vy0
    c_coef = release_pos_y - y_target

    # discriminant: b² - 4ac;  clamp to 0 for floating-point safety
    discriminant = (b_coef ** 2 - 4 * a_coef * c_coef).clip(lower=0.0)
    t = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
    return t.clip(lower=0.0)


def _pos_at_t(
    t: pd.Series,
    pos0: pd.Series,
    v0: pd.Series,
    a: pd.Series,
) -> pd.Series:
    """Kinematic position: pos(t) = pos0 + v0·t + ½·a·t²"""
    return pos0 + v0 * t + 0.5 * a * t ** 2


# ── Pitch-level computation ───────────────────────────────────────────────────

def compute_tunnel_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the (x, z) position of each pitch at:
      • the tunnel point  (y = TUNNEL_Y_FT ≈ 26.5 ft from plate)
      • the plate         (y = PLATE_Y_FT  ≈  1.4 ft from plate)

    All coordinates in feet (Statcast frame: x is horizontal from catcher's
    view, z is vertical, y decreases toward the plate).

    Adds four columns:
        tunnel_x, tunnel_z      — position at batter decision point
        plate_x_kin, plate_z_kin — kinematic plate position (not Statcast
                                   plate_x/plate_z; those encode command)

    Returns a copy of df with the new columns.  Returns NaN columns if any
    required kinematic column is missing.
    """
    required = [
        "vx0", "vy0", "vz0",
        "ax",  "ay",  "az",
        "release_pos_x", "release_pos_y", "release_pos_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "compute_tunnel_pos: missing Statcast kinematic columns %s — "
            "tunnel features will be NaN", missing
        )
        df = df.copy()
        for col in _TUNNEL_INTERMEDIATE_COLS:
            df[col] = np.nan
        return df

    df = df.copy()

    # ── Time to tunnel point ──────────────────────────────────────────────────
    t_tun = _time_at_y(df["vy0"], df["ay"], df["release_pos_y"], TUNNEL_Y_FT)
    df["tunnel_x"] = _pos_at_t(t_tun, df["release_pos_x"], df["vx0"], df["ax"])
    df["tunnel_z"] = _pos_at_t(t_tun, df["release_pos_z"], df["vz0"], df["az"])

    # ── Time to front of home plate ───────────────────────────────────────────
    t_plt = _time_at_y(df["vy0"], df["ay"], df["release_pos_y"], PLATE_Y_FT)
    df["plate_x_kin"] = _pos_at_t(t_plt, df["release_pos_x"], df["vx0"], df["ax"])
    df["plate_z_kin"] = _pos_at_t(t_plt, df["release_pos_z"], df["vz0"], df["az"])

    return df


# ── Arsenal-level computation ─────────────────────────────────────────────────

def add_tunnel_features(
    agg: pd.DataFrame,
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
) -> pd.DataFrame:
    """
    Add per-row tunneling features to an already-aggregated DataFrame.

    For each (pitcher, season, pitch_type) row, this function:
      1. Looks up every other pitch type the same pitcher threw that season.
      2. Computes the Euclidean distance at the tunnel point between this
         pitch type's mean trajectory and each of the others.
      3. Identifies the *nearest* pitch type (minimum tunnel distance).
      4. Computes how much more the pitch pair separates at the plate than
         at the tunnel point (the divergence ratio).

    Requires that the aggregated DataFrame contains mean tunnel_x, tunnel_z,
    plate_x_kin, plate_z_kin columns (computed at pitch level by
    compute_tunnel_pos and averaged in aggregate_to_pitch_type_level).

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated pitcher-season-pitch_type data with tunnel intermediate cols.
    cfg : dict, optional
        Parsed config.yaml.  Not used yet; reserved for future knobs.

    Returns
    -------
    pd.DataFrame
        agg enriched with tunnel_dist_nearest and tunnel_ratio_nearest, with
        intermediate columns dropped.
    """
    if "tunnel_x" not in agg.columns or "tunnel_z" not in agg.columns:
        logger.warning(
            "add_tunnel_features: tunnel_x / tunnel_z not in aggregated data. "
            "Re-run the 'features' pipeline step to populate them."
        )
        agg = agg.copy()
        agg["tunnel_dist_nearest"]  = np.nan
        agg["tunnel_ratio_nearest"] = np.nan
        return agg

    has_plate = ("plate_x_kin" in agg.columns and "plate_z_kin" in agg.columns)

    agg = agg.copy()
    agg["tunnel_dist_nearest"]  = np.nan
    agg["tunnel_ratio_nearest"] = np.nan

    # ── Per pitcher-season, compute pairwise tunnel / plate distances ─────────
    for (pitcher, season), grp in agg.groupby(["pitcher", "season"], observed=True):

        # Only keep rows with valid tunnel positions
        valid = grp.dropna(subset=["tunnel_x", "tunnel_z"])
        if len(valid) < 2:
            # Solo pitch type — no tunnel partner exists; leave as NaN
            continue

        tx = valid["tunnel_x"].values        # shape (n,)
        tz = valid["tunnel_z"].values
        row_idx = valid.index.to_numpy()

        if has_plate:
            px = valid["plate_x_kin"].values
            pz = valid["plate_z_kin"].values

        n = len(valid)

        for i in range(n):
            # Euclidean distances at tunnel point to all other pitch types
            diff_x = tx - tx[i]
            diff_z = tz - tz[i]
            tdists = np.sqrt(diff_x ** 2 + diff_z ** 2)
            tdists[i] = np.inf               # exclude self

            nearest = int(np.argmin(tdists))
            min_td  = tdists[nearest]

            agg.at[row_idx[i], "tunnel_dist_nearest"] = float(min_td)

            if has_plate and min_td > 1e-4:  # guard against exact-duplicate positions
                pdist = np.sqrt(
                    (px[nearest] - px[i]) ** 2 +
                    (pz[nearest] - pz[i]) ** 2
                )
                # Clip at 60 — ratios above that are physically implausible
                # and would dominate the feature scale unnecessarily.
                agg.at[row_idx[i], "tunnel_ratio_nearest"] = min(
                    float(pdist / min_td), 60.0
                )

    # ── Log summary ───────────────────────────────────────────────────────────
    n_with = agg["tunnel_dist_nearest"].notna().sum()
    n_nan  = agg["tunnel_dist_nearest"].isna().sum()
    logger.info(
        "Tunnel features computed: %d rows with data, %d NaN "
        "(single-pitch-type pitchers — XGBoost will handle NaN natively)",
        n_with, n_nan,
    )
    if n_with > 0:
        logger.info(
            "  tunnel_dist_nearest  mean=%.3f ft  median=%.3f ft",
            float(agg["tunnel_dist_nearest"].mean()),
            float(agg["tunnel_dist_nearest"].median()),
        )
        logger.info(
            "  tunnel_ratio_nearest mean=%.2f     median=%.2f",
            float(agg["tunnel_ratio_nearest"].mean()),
            float(agg["tunnel_ratio_nearest"].median()),
        )

    # ── Drop intermediate columns ─────────────────────────────────────────────
    drop_cols = [c for c in _TUNNEL_INTERMEDIATE_COLS if c in agg.columns]
    if drop_cols:
        agg = agg.drop(columns=drop_cols)

    return agg
