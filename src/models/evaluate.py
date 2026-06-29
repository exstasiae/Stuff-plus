"""
src/models/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Model evaluation: leaderboards, correlation tables, and diagnostic plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.models.train import apply_scaler

logger = logging.getLogger(__name__)


def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_artifacts(pitch_type: str,
                          cfg: dict) -> tuple[xgb.XGBRegressor, dict, list[str]]:
    """Load model, scaler, and feature list for a pitch type (e.g. "FF", "SL")."""
    model_dir = Path(cfg["paths"]["models"]) / pitch_type

    model = xgb.XGBRegressor()
    model.load_model(str(model_dir / "model.json"))

    with open(model_dir / "scaler.json") as f:
        scaler = json.load(f)
    with open(model_dir / "features.json") as f:
        features = json.load(f)

    return model, scaler, features


def predict_stuff_plus(df: pd.DataFrame,
                        cfg: dict | None = None,
                        config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """
    Score every row in the aggregated DataFrame with the appropriate per-pitch-type
    model and add a `stuff_plus` column.

    Each pitch type has its own model (FF → models/FF/model.json, etc.).
    Rows whose pitch_type has no trained model get NaN.
    """
    if cfg is None:
        cfg = _load_config(config_path)

    model_pitch_types_cfg = cfg.get("model_pitch_types", {})
    all_pitch_types: list[str] = (
        model_pitch_types_cfg.get("standard", [])
        + model_pitch_types_cfg.get("small", [])
    )
    if not all_pitch_types:
        # Fallback: derive from pitch_groups values
        for pts in cfg["pitch_groups"].values():
            all_pitch_types.extend(pts)

    df = df.copy()
    df["stuff_plus"] = np.nan

    for pitch_type in all_pitch_types:
        model_dir = Path(cfg["paths"]["models"]) / pitch_type
        if not (model_dir / "model.json").exists():
            logger.warning("No model found for pitch type '%s' — skipping", pitch_type)
            continue

        model, scaler, features = load_model_artifacts(pitch_type, cfg)
        mask = df["pitch_type"] == pitch_type
        if not mask.any():
            continue

        X = df.loc[mask, features].values
        raw_preds = model.predict(X)
        df.loc[mask, "stuff_plus"] = apply_scaler(raw_preds, scaler)
        logger.info("Scored %d rows for pitch type %s", int(mask.sum()), pitch_type)

    return df


def build_leaderboard(
    df: pd.DataFrame,
    min_pitches: int = 50,
    group: str | None = None,
    pitch_type: str | None = None,
) -> pd.DataFrame:
    """
    Build a Stuff+ leaderboard from aggregated data.

    With aggregated data each row is already one pitcher-season-pitch_type,
    so no further grouping is needed — just filter, sort, and return.

    Parameters
    ----------
    df : pd.DataFrame
        Output of predict_stuff_plus() on aggregated data.
        Must have a stuff_plus column and an n_pitches column.
    min_pitches : int
        Minimum pitches of that type in that season to appear in results.
    group : str, optional
        If provided, filter to one pitch *group* (fastball/breaking/offspeed).
    pitch_type : str, optional
        If provided, filter to one pitch *type* (e.g. "FF", "SL").
        Takes precedence over group if both are supplied.

    Returns
    -------
    pd.DataFrame
        Leaderboard sorted by Stuff+ descending.
    """
    if pitch_type:
        df = df[df["pitch_type"] == pitch_type]
    elif group:
        df = df[df["pitch_group"] == group]

    df = df.dropna(subset=["stuff_plus"])

    if "n_pitches" in df.columns:
        df = df[df["n_pitches"] >= min_pitches]

    # Select display columns in a logical order
    want = ["player_name", "pitcher", "season", "pitch_type", "pitch_group",
            "n_pitches", "stuff_plus"]
    cols = [c for c in want if c in df.columns]

    return (
        df[cols]
        .assign(stuff_plus=lambda x: x["stuff_plus"].round(1))
        .sort_values("stuff_plus", ascending=False)
        .reset_index(drop=True)
    )


def save_leaderboard(leaderboard: pd.DataFrame,
                      filename: str,
                      cfg: dict) -> Path:
    """Save leaderboard to outputs/leaderboards/ as CSV."""
    out_dir = Path(cfg["paths"]["outputs"]) / "leaderboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    leaderboard.to_csv(path, index=False)
    logger.info("Leaderboard saved → %s", path)
    return path


def plot_feature_importance(pitch_type: str,
                             cfg: dict,
                             top_n: int = 15,
                             save: bool = True) -> None:
    """Bar chart of XGBoost feature importances for a pitch type (e.g. 'FF')."""
    model, _, features = load_model_artifacts(pitch_type, cfg)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    feat_imp = feat_imp.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance — {pitch_type} Model")
    ax.set_xlabel("XGBoost gain (normalized)")
    plt.tight_layout()

    if save:
        out_dir = Path(cfg["paths"]["outputs"]) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"feature_importance_{pitch_type}.png"
        fig.savefig(path, dpi=150)
        logger.info("Saved feature importance plot → %s", path)
    plt.show()


def validate_whiff_rate_correlation(
    df: pd.DataFrame,
    cfg: dict | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Sanity-check: Stuff+ should correlate positively with whiff rate and CSW rate.
    A good pure-stuff model should have r ≥ 0.40 for most pitch types.

    Prints a table and optionally saves a scatter plot per pitch type.

    Returns
    -------
    pd.DataFrame
        Correlation summary with columns: pitch_type, n, r_whiff, r_csw.
    """
    from scipy.stats import spearmanr

    rows = []
    for pt in sorted(df["pitch_type"].dropna().unique()):
        sub = df[df["pitch_type"] == pt].copy()

        n = len(sub.dropna(subset=["stuff_plus"]))
        r_whiff = r_csw = float("nan")

        if "whiff_rate" in sub.columns:
            mask = sub[["stuff_plus", "whiff_rate"]].notna().all(axis=1)
            if mask.sum() >= 15:
                r_whiff, _ = spearmanr(sub.loc[mask, "stuff_plus"],
                                       sub.loc[mask, "whiff_rate"])
        if "csw_rate" in sub.columns:
            mask = sub[["stuff_plus", "csw_rate"]].notna().all(axis=1)
            if mask.sum() >= 15:
                r_csw, _ = spearmanr(sub.loc[mask, "stuff_plus"],
                                     sub.loc[mask, "csw_rate"])

        rows.append({"pitch_type": pt, "n": n,
                     "r_whiff": round(r_whiff, 3), "r_csw": round(r_csw, 3)})

    summary = pd.DataFrame(rows)

    print("\nStuff+ vs. outcome-rate validation (Spearman ρ):")
    print("  pitch_type    n   r_whiff   r_csw")
    print("  " + "-" * 38)
    for _, row in summary.iterrows():
        flag = "  ⚠" if (not np.isnan(row["r_whiff"]) and row["r_whiff"] < 0.35) else ""
        print(f"  {row['pitch_type']:<12} {row['n']:4d}   {row['r_whiff']:+.3f}   {row['r_csw']:+.3f}{flag}")

    if save and cfg is not None:
        out_dir = Path(cfg["paths"]["outputs"]) / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "whiff_correlation.csv", index=False)
        logger.info("Saved whiff correlation summary → %s", out_dir / "whiff_correlation.csv")

        # Scatter: Stuff+ vs whiff_rate, one panel per pitch type
        if "whiff_rate" in df.columns:
            pts = sorted(df["pitch_type"].dropna().unique())
            n_pts = len(pts)
            fig, axes = plt.subplots(2, (n_pts + 1) // 2,
                                     figsize=(5 * ((n_pts + 1) // 2), 8))
            axes = np.array(axes).ravel()
            for ax, pt in zip(axes, pts):
                sub = df[df["pitch_type"] == pt].dropna(subset=["stuff_plus", "whiff_rate"])
                ax.scatter(sub["stuff_plus"], sub["whiff_rate"],
                           alpha=0.4, s=18, color="steelblue")
                ax.set_title(f"{pt} (n={len(sub)})")
                ax.set_xlabel("Stuff+")
                ax.set_ylabel("Whiff rate")
            for ax in axes[len(pts):]:
                ax.set_visible(False)
            fig.suptitle("Stuff+ vs. Whiff Rate by Pitch Type", fontsize=13)
            plt.tight_layout()
            path = out_dir / "stuff_plus_vs_whiff.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved scatter plot → %s", path)

    return summary


def plot_stuff_plus_distribution(df: pd.DataFrame,
                                  group: str | None = None,
                                  cfg: dict | None = None,
                                  save: bool = True) -> None:
    """Histogram of Stuff+ values, one panel per pitch type."""
    if group:
        df = df[df["pitch_group"] == group]
    if cfg is None:
        cfg = _load_config()

    pitch_types = df["pitch_type"].dropna().unique()
    n = len(pitch_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, pt in zip(axes, sorted(pitch_types)):
        vals = df.loc[df["pitch_type"] == pt, "stuff_plus"].dropna()
        ax.hist(vals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(100, color="red", linestyle="--", linewidth=1.5, label="Avg (100)")
        ax.set_title(f"{pt}  (n={len(vals):,})")
        ax.set_xlabel("Stuff+")
        ax.legend(fontsize=8)

    fig.suptitle("Stuff+ Distribution by Pitch Type", fontsize=13, y=1.02)
    plt.tight_layout()

    if save and cfg:
        out_dir = Path(cfg["paths"]["outputs"]) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{group}" if group else ""
        path = out_dir / f"stuff_plus_distribution{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved distribution plot → %s", path)
    plt.show()
