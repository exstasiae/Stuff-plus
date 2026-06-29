"""
src/models/train.py
─────────────────────────────────────────────────────────────────────────────
Train a Stuff+ model (one XGBoost model per pitch group) and save artifacts.

Pipeline
────────
  1. Load cleaned + engineered data from data/processed/
  2. Split by pitch group (fastball / breaking / offspeed)
  3. Time-based train/validation split (hold out most recent season)
  4. Train XGBoost with hyperparameters from config.yaml
  5. Evaluate on validation set
  6. Normalize predictions to 100-scale
  7. Save: model JSON, scaler params, feature list → models/

Usage
─────
  python -m src.models.train
  python -m src.models.train --val-season 2024 --groups fastball breaking
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(cfg: dict) -> pd.DataFrame:
    """Load the aggregated (pitcher-season-pitch_type level) Parquet file."""
    import pyarrow.parquet as pq
    path = Path(cfg["paths"]["processed_data"]) / "statcast_aggregated.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregated file not found: {path}\n"
            "Run the full pipeline first:\n"
            "  python pipeline.py --steps fetch preprocess features aggregate"
        )
    df = pq.read_table(path).to_pandas()
    logger.info(
        "Loaded %d aggregated rows from %s  "
        "(pitcher-season-pitch_type level, avg %.0f pitches/row)",
        len(df), path, df["n_pitches"].mean() if "n_pitches" in df.columns else float("nan"),
    )
    return df


# ── Train / validation split ──────────────────────────────────────────────────

def time_split(df: pd.DataFrame,
               val_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a full season as validation data.
    All prior seasons form the training set.

    Supports both aggregated data (has 'season' int column) and pitch-level
    data (has 'game_date' datetime column).
    """
    if "season" in df.columns:
        # Aggregated data path — season is an explicit integer column
        train = df[df["season"] < val_season].copy()
        val   = df[df["season"] == val_season].copy()
    elif "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        train = df[df["game_date"].dt.year < val_season].copy()
        val   = df[df["game_date"].dt.year == val_season].copy()
    else:
        logger.warning("No season/game_date column; falling back to 80/20 split")
        cutoff = int(len(df) * 0.8)
        train, val = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()

    logger.info(
        "Train: %d rows (seasons < %d) | Val: %d rows (season %d)",
        len(train), val_season, len(val), val_season,
    )
    return train, val


# ── Model training ────────────────────────────────────────────────────────────

def train_group_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict,
    group_name: str,
    sample_weight_col: str = "n_pitches",
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """
    Train one XGBoost model for a single pitch group.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (aggregated rows for this pitch group only).
    val_df : pd.DataFrame
        Validation split.
    features : list[str]
        Input feature column names.
    target : str
        Target column name ("lw_run_value").
    params : dict
        XGBoost hyperparameters from config.
    group_name : str
        "fastball", "breaking", or "offspeed" — for logging.
    sample_weight_col : str
        Column to use as sample weights.  With aggregated data this is
        "n_pitches" so that a row representing 2 000 pitches has 20×
        more influence than one representing 100 pitches.

    Returns
    -------
    model : xgb.XGBRegressor
    metrics : dict[str, float]
        RMSE and correlation on the validation set.
    """
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_val   = val_df[features].values
    y_val   = val_df[target].values

    # Sample weights: pitch count per row (aggregated data only)
    w_train = (
        train_df[sample_weight_col].values
        if sample_weight_col in train_df.columns
        else None
    )

    n_train_pitches = (
        int(train_df[sample_weight_col].sum())
        if w_train is not None else len(train_df)
    )
    n_val_pitches = (
        int(val_df[sample_weight_col].sum())
        if sample_weight_col in val_df.columns else len(val_df)
    )
    logger.info(
        "[%s] Train: %d rows / %d pitches | Val: %d rows / %d pitches",
        group_name,
        len(X_train), n_train_pitches,
        len(X_val),   n_val_pitches,
    )

    model = xgb.XGBRegressor(
        **params,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
    corr  = float(np.corrcoef(y_val, preds)[0, 1])

    metrics = {"rmse": rmse, "corr": corr, "best_iteration": model.best_iteration}
    logger.info("[%s] Val RMSE=%.4f  Corr=%.4f  (best iter=%d)",
                group_name, rmse, corr, model.best_iteration)

    return model, metrics


# ── Normalization ─────────────────────────────────────────────────────────────

def compute_scaler(raw_preds: np.ndarray,
                   center: float = 100.0,
                   scale: float = 10.0) -> dict[str, float]:
    """
    Compute scaling parameters so that:
        normalized = center + scale * (raw - mean_raw) / std_raw

    This gives 100 = league average, and ~10 points per standard deviation.
    The parameters are saved alongside the model so they can be applied at
    inference time consistently.
    """
    mean_raw = float(np.mean(raw_preds))
    std_raw  = float(np.std(raw_preds))
    return {"mean_raw": mean_raw, "std_raw": std_raw,
            "center": center, "scale": scale}


def apply_scaler(raw_preds: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    """
    Apply saved scaler params to raw model output.

    The model predicts lw_run_value where LOWER (more negative) = better pitch.
    We invert the z-score so that elite pitchers score ABOVE 100:
        stuff_plus = center − scale × z
    """
    z = (raw_preds - scaler["mean_raw"]) / scaler["std_raw"]
    return scaler["center"] - scaler["scale"] * z


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def save_model_artifacts(
    group: str,
    model: xgb.XGBRegressor,
    scaler: dict[str, float],
    features: list[str],
    metrics: dict[str, float],
    cfg: dict,
) -> Path:
    """
    Save all artifacts for one pitch group under models/<group>/.

    Saved files:
      model.json      — XGBoost model (portable, version-stable)
      scaler.json     — normalization params
      features.json   — ordered feature list (must match at inference time)
      metrics.json    — validation metrics
    """
    out_dir = Path(cfg["paths"]["models"]) / group
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(out_dir / "model.json"))

    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    with open(out_dir / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved artifacts → %s", out_dir)
    return out_dir


# ── Full training pipeline ────────────────────────────────────────────────────

def run_training(
    groups: list[str] | None = None,
    val_season: int | None = None,
    cfg: dict | None = None,
    config_path: str | Path = "config.yaml",
) -> dict[str, dict]:
    """
    End-to-end training: load data, train one XGBoost model per pitch type,
    and save artifacts.

    Parameters
    ----------
    groups : list[str], optional
        Pitch *types* to train (e.g. ["FF", "SL"]).  Defaults to all types
        listed in config.model_pitch_types (standard + small).
    val_season : int, optional
        Season to hold out for validation.  Defaults to max season in data.
    cfg : dict, optional
        Parsed config.

    Returns
    -------
    dict
        Mapping of pitch_type → {"model", "scaler", "features", "metrics"}.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    # Collect pitch types to train
    model_pitch_types_cfg = cfg.get("model_pitch_types", {})
    all_pitch_types: list[str] = (
        model_pitch_types_cfg.get("standard", [])
        + model_pitch_types_cfg.get("small", [])
    )
    if not all_pitch_types:
        # Fallback: derive from pitch_groups values
        for pts in cfg["pitch_groups"].values():
            all_pitch_types.extend(pts)

    pitch_types = groups if groups else all_pitch_types

    target   = cfg["target"]
    features = cfg["features"]["model_inputs"]

    df = load_training_data(cfg)

    # Determine validation season (default: most recent season in data)
    if val_season is None:
        if "season" in df.columns:
            val_season = int(df["season"].max())
        elif "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
            val_season = int(df["game_date"].dt.year.max())
        else:
            val_season = cfg["seasons"]["train"][-1]
    logger.info("Validation season: %d", val_season)

    # Threshold for choosing standard vs. small-sample hyperparameters
    _SMALL_SAMPLE_THRESHOLD = 150

    results: dict[str, dict] = {}

    for pitch_type in pitch_types:
        logger.info("=" * 60)
        logger.info("Training pitch type: %s", pitch_type)

        type_df = df[df["pitch_type"] == pitch_type].copy()
        if len(type_df) < 20:
            logger.warning("Very few rows for pitch type '%s' (%d) — skipping",
                           pitch_type, len(type_df))
            continue

        # Drop rows where any required feature or target is NaN.
        # Tunnel features are optional: single-pitch-type pitchers have no
        # tunnel partner and get NaN. XGBoost handles NaN natively (learns
        # the optimal branch direction for missing values), so keep these rows.
        _TUNNEL_OPTIONAL = frozenset({
            "tunnel_dist_nearest", "z_tunnel_dist_nearest",
            "tunnel_ratio_nearest", "z_tunnel_ratio_nearest",
        })
        needed = features + [target]
        required_subset = [
            c for c in needed
            if c in type_df.columns and c not in _TUNNEL_OPTIONAL
        ]
        type_df = type_df.dropna(subset=required_subset)

        train_df, val_df = time_split(type_df, val_season)

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(
                "Pitch type '%s' has no data in train (%d) or val (%d) split — skipping",
                pitch_type, len(train_df), len(val_df),
            )
            continue

        # Adaptive hyperparameters: extra regularisation for thin pitch types
        if len(type_df) < _SMALL_SAMPLE_THRESHOLD:
            param_key = "small_sample"
        else:
            param_key = "standard"
        params = cfg["xgboost"][param_key]
        logger.info(
            "[%s] Using '%s' hyperparameters (%d total rows)",
            pitch_type, param_key, len(type_df),
        )

        model, metrics = train_group_model(
            train_df, val_df, features, target, params, pitch_type
        )

        # Build scaler from training-set predictions (so scale reflects
        # the distribution of pitches seen during training, not just val)
        train_preds = model.predict(train_df[features].values)
        scaler = compute_scaler(
            train_preds,
            center=cfg["normalization"]["center"],
            scale=cfg["normalization"]["scale"],
        )

        save_model_artifacts(pitch_type, model, scaler, features, metrics, cfg)

        results[pitch_type] = {
            "model":    model,
            "scaler":   scaler,
            "features": features,
            "metrics":  metrics,
        }

    logger.info("=" * 60)
    logger.info("Training complete. Summary:")
    for pt, res in results.items():
        m = res["metrics"]
        logger.info("  %-6s  RMSE=%.4f  Corr=%.4f", pt, m["rmse"], m["corr"])

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train Stuff+ models.")
    parser.add_argument("--groups", nargs="+",
                        help="Pitch types to train, e.g. FF SL CH (default: all)")
    parser.add_argument("--val-season", type=int,
                        help="Season to hold out for validation")
    args = parser.parse_args()

    run_training(groups=args.groups, val_season=args.val_season)
