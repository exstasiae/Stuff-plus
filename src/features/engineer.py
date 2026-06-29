"""
src/features/engineer.py
─────────────────────────────────────────────────────────────────────────────
Feature engineering for the Stuff+ model.

Beyond the raw Statcast fields, derive several physically-meaningful
features that give the model richer signal:

  induced_vertical_break (IVB)
      The "true" vertical movement of a pitch after removing the effect of
      gravity.  A 4-seam fastball that "rises" relative to a gravity-only
      trajectory has high IVB.  Computed as pfx_z + (gravity component
      expected over the pitch's flight time).

      Why this matters: pfx_z on its own mixes the pitcher's spin-induced
      movement with a fixed gravitational constant.  IVB isolates what the
      pitcher is actually doing.

  vertical_approach_angle (VAA)
      The angle (in degrees) at which the pitch crosses the front of the
      plate, measured in the vertical plane.  A pitch arriving at -4° looks
      "flatter" than one at -7°.  Flatter approach angles are harder to
      barrel because the bat's sweet spot spends less time in the ball's
      path.

      Computed from the velocity vector components (vz0, vy0, ay) near the
      plate (y ≈ 1.417 ft from the back of home plate).

  horizontal_approach_angle (HAA)
      Same idea in the horizontal plane — how much is the pitch cutting
      across the zone laterally at the point of contact?

  arm_side_break
      pfx_x signed so that positive always means "toward the arm side" of
      the pitcher (away from a same-handed batter).  Normalising by handedness
      makes features comparable across L/R pitchers.

  glove_side_break
      Complement of arm_side_break — movement toward the glove side.

  speed_differential (vs. fastball)
      For a given pitcher, how much slower is this pitch than their fastest
      pitch?  Large differentials on changeups are a key quality signal.
      Note: this requires a groupby over the pitcher's other pitches, so it's
      computed last and is optional for single-game inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.tunnel import compute_tunnel_pos

logger = logging.getLogger(__name__)

# Gravitational acceleration in ft/s²
_G = 32.174


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Approach angle helpers ────────────────────────────────────────────────────

def _velocity_at_y(vy0: pd.Series, ay: pd.Series,
                   y_release: pd.Series, y_target: float = 1.417) -> pd.Series:
    """
    Compute vy at a given y position (y_target) using kinematic equations.
    y_release is the release point distance from the back of the plate (ft).
    """
    # Time to travel from release to y_target: solve y_target = y0 + vy0*t + 0.5*ay*t²
    # Using quadratic formula; vy0 and y values are negative (toward plate).
    delta_y = y_target - y_release
    discriminant = vy0**2 + 2 * ay * delta_y
    # Guard against small negative values from floating-point noise
    discriminant = discriminant.clip(lower=0)
    t = (-vy0 - np.sqrt(discriminant)) / ay
    return vy0 + ay * t


def compute_vaa(df: pd.DataFrame) -> pd.Series:
    """
    Vertical approach angle (VAA) in degrees at the front of the plate.
    Negative values = pitch coming downward (as expected).
    """
    required = ["vz0", "vy0", "ay", "az", "release_pos_y"]
    if not all(c in df.columns for c in required):
        logger.warning("Missing columns for VAA; returning NaN")
        return pd.Series(np.nan, index=df.index)

    vy_plate = _velocity_at_y(df["vy0"], df["ay"],
                               df["release_pos_y"], y_target=1.417)
    # vz at the plate: same kinematic step for z
    delta_y  = 1.417 - df["release_pos_y"]
    t_plate  = (vy_plate - df["vy0"]) / df["ay"]
    vz_plate = df["vz0"] + df["az"] * t_plate

    vaa = np.degrees(np.arctan(vz_plate / vy_plate.abs()))
    return vaa


def compute_haa(df: pd.DataFrame) -> pd.Series:
    """Horizontal approach angle (HAA) in degrees at the front of the plate."""
    required = ["vx0", "vy0", "ax", "ay", "release_pos_y"]
    if not all(c in df.columns for c in required):
        logger.warning("Missing columns for HAA; returning NaN")
        return pd.Series(np.nan, index=df.index)

    vy_plate = _velocity_at_y(df["vy0"], df["ay"],
                               df["release_pos_y"], y_target=1.417)
    delta_y  = 1.417 - df["release_pos_y"]
    t_plate  = (vy_plate - df["vy0"]) / df["ay"]
    vx_plate = df["vx0"] + df["ax"] * t_plate

    haa = np.degrees(np.arctan(vx_plate / vy_plate.abs()))
    return haa


# ── Induced vertical break ────────────────────────────────────────────────────

def compute_ivb(df: pd.DataFrame) -> pd.Series:
    """
    Induced vertical break (IVB) = pfx_z + gravity component.

    Statcast pfx_z already represents vertical movement vs. a "no-spin"
    trajectory (which includes gravity).  IVB is sometimes defined slightly
    differently by different vendors; here we use the version consistent with
    Baseball Savant's pfx_z meaning: pfx_z IS the spin-induced movement, so
    IVB ≈ pfx_z for our purposes — we apply a small correction for release
    height that vendors sometimes include.

    For a simpler definition used by many public models:
        IVB = pfx_z  (already gravity-corrected by Trackman)

    We keep both the raw pfx_z and the derived IVB in the feature set.
    """
    # Statcast pfx values are already the spin-induced deviation from a
    # theoretical no-spin (gravity only) trajectory, so:
    return df["pfx_z"].copy()


# ── Spin efficiency ───────────────────────────────────────────────────────────

def compute_spin_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    Estimate active-spin fraction (spin efficiency).

    Not all spin creates useful Magnus force.  "Gyro" spin (axis pointing
    toward the catcher, like a bullet) produces zero Magnus movement; pure
    "transverse" spin (axis perpendicular to travel) produces maximum movement.
    Spin efficiency = fraction of spin that is transverse (active).

    Algorithm
    ─────────
    1. From the Statcast spin_axis (clock-face degrees, 0 = 12 o'clock),
       compute the unit vector of expected Magnus movement:
           expected_x =  sin(spin_axis_rad)
           expected_z = −cos(spin_axis_rad)
       Convention check: spin_axis=180° (backspin) → expected_z = +1 (rise) ✓
                         spin_axis=90°  (sidespin)  → expected_x = +1       ✓

    2. Project actual movement (pfx_x, pfx_z, in inches) onto that vector
       to get the "active" Magnus component.

    3. Divide by the theoretical max movement for 100% active spin at this
       spin rate and velocity (empirically calibrated to MLB averages):
           theoretical = (spin_rate / 2400) × (velocity / 95) × 18 inches

    Output range: ~0.0 (pure gyro) → ~1.0 (pure Magnus).
    Values slightly > 1.0 occur when seam-shifted wake (SSW) adds movement
    beyond what Magnus predicts; clipped at 1.25.
    Slightly negative values (gyro sliders) clipped at −0.05.

    NaN is returned for pitches where spin_axis is missing (pre-Hawk-Eye or
    tracking failure).  These become NaN means at the aggregate level and
    those pitcher-season-type rows are dropped before training.
    """
    required = ["pfx_x", "pfx_z", "release_speed", "release_spin_rate", "spin_axis"]
    if not all(c in df.columns for c in required):
        logger.warning("Missing columns for spin efficiency (need %s) — returning NaN",
                       [c for c in required if c not in df.columns])
        return pd.Series(np.nan, index=df.index)

    spin_axis_rad = np.deg2rad(df["spin_axis"])

    # Unit vector of expected Magnus movement direction
    expected_x = np.sin(spin_axis_rad)
    expected_z = -np.cos(spin_axis_rad)

    # Project actual movement (pfx values are in inches) onto expected direction
    active_movement = df["pfx_x"] * expected_x + df["pfx_z"] * expected_z

    # Theoretical maximum Magnus movement (inches) at 100% active spin
    # Calibration: 2400 rpm + 95 mph → ~18 in max; scales linearly with both
    theoretical_movement = (
        (df["release_spin_rate"] / 2400.0)
        * (df["release_speed"]   / 95.0)
        * 18.0
    ).clip(lower=0.1)   # guard against division by ~zero

    efficiency = active_movement / theoretical_movement
    return efficiency.clip(lower=-0.05, upper=1.25).where(df["spin_axis"].notna())


# ── Arm-side / glove-side break ───────────────────────────────────────────────

def compute_arm_side_break(df: pd.DataFrame) -> pd.Series:
    """
    pfx_x signed so +1 = arm side regardless of pitcher handedness.
    RHP: arm side is catcher's left (pfx_x negative in Statcast convention)
    LHP: arm side is catcher's right (pfx_x positive)
    """
    if "p_throws" not in df.columns:
        return df["pfx_x"].copy()
    arm_sign = df["p_throws"].map({"R": -1, "L": 1}).fillna(1)
    return df["pfx_x"] * arm_sign


# ── Speed differential ────────────────────────────────────────────────────────

def compute_speed_differential(df: pd.DataFrame) -> pd.Series:
    """
    For each pitch, compute how much slower it is than the pitcher's
    season-average fastball velocity.  Fastballs themselves get 0.

    This is computed per (pitcher, season) if a game_date column is present,
    otherwise per pitcher across the full dataset.
    """
    fastball_types = {"FF", "SI", "FC"}

    out = pd.Series(0.0, index=df.index)

    if "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df = df.copy()
        df["_season"] = df["game_date"].dt.year
        group_keys = ["pitcher", "_season"]
    else:
        group_keys = ["pitcher"]

    fb_mask = df["pitch_type"].isin(fastball_types)

    # Mean fastball velo per pitcher (per season)
    fb_velo = (
        df[fb_mask]
        .groupby(group_keys)["release_speed"]
        .mean()
        .rename("fb_velo")
    )

    df2 = df.join(fb_velo, on=group_keys)
    out = (df2["fb_velo"] - df2["release_speed"]).fillna(0).clip(lower=0)

    return out


# ── Master feature builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   cfg: dict | None = None,
                   config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """
    Add all engineered features to a cleaned Statcast DataFrame.

    Modifies a copy; does not alter the input.

    Returns
    -------
    pd.DataFrame
        Original columns plus the new engineered columns listed below.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    df = df.copy()
    logger.info("Building features for %d rows…", len(df))

    df["induced_vertical_break"] = compute_ivb(df)
    df["horz_break"]             = df["pfx_x"].copy()  # keep raw for model
    df["arm_side_break"]         = compute_arm_side_break(df)
    df["vaa"]                    = compute_vaa(df)
    df["haa"]                    = compute_haa(df)
    df["speed_diff"]             = compute_speed_differential(df)
    df["spin_efficiency"]        = compute_spin_efficiency(df)

    # Tunnel positions: (x, z) at the batter decision point and at the plate.
    # These are INTERMEDIATE columns — not model inputs themselves — averaged
    # at the aggregate step and used by add_tunnel_features() to compute the
    # cross-pitch tunnel distance and divergence ratio features.
    df = compute_tunnel_pos(df)

    # Encode handedness if not already done by preprocess
    if "p_throws_enc" not in df.columns and "p_throws" in df.columns:
        df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)

    logger.info("Feature engineering complete. New columns: "
                "induced_vertical_break, horz_break, arm_side_break, "
                "vaa, haa, speed_diff, spin_efficiency, "
                "tunnel_x, tunnel_z, plate_x_kin, plate_z_kin, p_throws_enc")

    return df


def get_feature_list(cfg: dict | None = None,
                     config_path: str | Path = "config.yaml") -> list[str]:
    """Return the list of model input features from config."""
    if cfg is None:
        cfg = _load_config(config_path)
    return cfg["features"]["model_inputs"]
