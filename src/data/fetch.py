"""
src/data/fetch.py
─────────────────────────────────────────────────────────────────────────────
Statcast data fetching with a Parquet-based local cache.

CACHE ARCHITECTURE
──────────────────
data/raw/
├── manifest.json                  ← tracks every file that has been written
├── statcast_2017.parquet          ← one file per completed season
├── statcast_2018.parquet
│   ...
├── statcast_2024.parquet
└── current/                       ← current season stored in monthly chunks
    ├── statcast_2026_03.parquet   ←   so we only re-fetch the latest month
    ├── statcast_2026_04.parquet
    └── statcast_2026_05.parquet

WHY MONTHLY CHUNKS FOR THE CURRENT SEASON?
───────────────────────────────────────────
If we stored the current season as one file we would need to re-download the
entire season every time we want to add a few new days. Monthly chunking
means only the *current month's* chunk is ever re-fetched; all prior months
are treated as immutable once the month is over.

WHY PARQUET?
────────────
Statcast data has ~90 columns per pitch row. A full season is ~750 000 rows.

  Format         Typical size    Cold read (all cols)
  ─────────────────────────────────────────────────────
  CSV (gzip)     ~100 MB         ~8 s
  Parquet/snappy  ~45 MB         ~1.2 s   ← we use this
  Parquet/zstd    ~35 MB         ~1.5 s

Parquet advantages relevant here:
  1. Columnar layout — reading 15 columns out of 90 only touches those 15
     column chunks on disk.  I/O is proportional to columns used, not total.
  2. Typed schema in file footer — floats stay floats, dates stay dates.
     No dtype-coercion bugs on reload.
  3. Row-group statistics — min/max stored per ~128 MB block, enabling
     predicate pushdown (e.g. filter by date range before loading into RAM).
  4. Snappy compression is near-lossless speed with ~3-4x size reduction
     over raw CSV for this kind of numeric/string mix.
  5. Works natively with both pandas (via pyarrow) and polars.

MANIFEST
────────
manifest.json is a dict keyed by cache filename:
  {
    "statcast_2023.parquet": {
      "season": 2023,
      "rows": 748212,
      "fetched_at": "2024-11-01T14:32:00",
      "is_complete": true,     ← completed seasons are never re-fetched
      "size_bytes": 47382910
    },
    "current/statcast_2026_04.parquet": { ..., "is_complete": true },
    "current/statcast_2026_05.parquet": { ..., "is_complete": false }
  }

`is_complete: false` means the chunk may grow (current month) — it will be
re-fetched on next call to load_season() or refresh_current_season().
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pybaseball import statcast
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Path helpers ──────────────────────────────────────────────────────────────

def _raw_dir(cfg: dict) -> Path:
    p = Path(cfg["paths"]["raw_data"])
    p.mkdir(parents=True, exist_ok=True)
    return p

def _current_dir(cfg: dict) -> Path:
    p = _raw_dir(cfg) / "current"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _season_path(cfg: dict, season: int) -> Path:
    return _raw_dir(cfg) / f"statcast_{season}.parquet"

def _month_path(cfg: dict, year: int, month: int) -> Path:
    return _current_dir(cfg) / f"statcast_{year}_{month:02d}.parquet"

def _manifest_path(cfg: dict) -> Path:
    return _raw_dir(cfg) / "manifest.json"


# ── Manifest ──────────────────────────────────────────────────────────────────

def _read_manifest(cfg: dict) -> dict:
    """Load manifest; returns empty dict if it doesn't exist yet."""
    p = _manifest_path(cfg)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

def _write_manifest(cfg: dict, manifest: dict) -> None:
    with open(_manifest_path(cfg), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

def _update_manifest(cfg: dict, key: str, path: Path, is_complete: bool,
                     season: int | None = None) -> None:
    """Update a single entry in the manifest after a successful write."""
    manifest = _read_manifest(cfg)
    manifest[key] = {
        "season":       season,
        "rows":         pq.read_metadata(path).num_rows,
        "fetched_at":   datetime.utcnow().isoformat(timespec="seconds"),
        "is_complete":  is_complete,
        "size_bytes":   path.stat().st_size,
    }
    _write_manifest(cfg, manifest)
    logger.debug("Manifest updated: %s → %s", key, manifest[key])


# ── Season date ranges ────────────────────────────────────────────────────────

# Approximate MLB season start/end dates.  Exact game dates don't matter much
# because Statcast returns an empty DataFrame for off-days.
_SEASON_DATES: dict[int, tuple[str, str]] = {
    2015: ("2015-04-05", "2015-11-01"),
    2016: ("2016-04-03", "2016-11-03"),
    2017: ("2017-04-02", "2017-11-02"),
    2018: ("2018-03-29", "2018-10-28"),
    2019: ("2019-03-28", "2019-10-30"),
    2020: ("2020-07-23", "2020-10-27"),   # COVID short season
    2021: ("2021-04-01", "2021-11-02"),
    2022: ("2022-04-07", "2022-11-05"),
    2023: ("2023-03-30", "2023-11-04"),
    2024: ("2024-03-20", "2024-10-30"),
    2025: ("2025-03-27", "2025-10-31"),
    2026: ("2026-03-26", "2026-10-31"),   # estimate — update if opening day shifts
}

def _season_date_range(season: int) -> tuple[str, str]:
    if season not in _SEASON_DATES:
        raise ValueError(f"No date range defined for season {season}. "
                         f"Add it to _SEASON_DATES in fetch.py.")
    return _SEASON_DATES[season]


# ── Parquet I/O helpers ───────────────────────────────────────────────────────

def _write_parquet(df: pd.DataFrame, path: Path,
                   compression: str = "snappy") -> None:
    """
    Write a DataFrame to Parquet.

    compression choices:
      "snappy"  — fastest read/write, ~3-4x size reduction   ← default
      "zstd"    — slower write, ~20% smaller than snappy
      "gzip"    — slowest, smallest; rarely worth it here
      "none"    — no compression, maximum read speed if disk I/O is bottleneck

    We use snappy because Statcast data is loaded frequently during training
    experiments; read speed matters more than absolute file size.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        # row_group_size controls how many rows per internal block.
        # Smaller = faster filtered reads; larger = better compression.
        # 200_000 rows is a reasonable middle ground for ~750k row seasons.
        row_group_size=200_000,
    )
    logger.info("Written %d rows → %s (%.1f MB)",
                len(df), path, path.stat().st_size / 1e6)

def _read_parquet(path: Path,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a Parquet file, optionally projecting to a subset of columns.

    Passing `columns` triggers column pruning at the storage layer — only
    the requested column chunks are read from disk, everything else is
    skipped.  On a 90-column Statcast file reading 15 columns, this gives
    roughly a 5-6x I/O reduction.
    """
    return pq.read_table(path, columns=columns).to_pandas()


# ── pybaseball fetch with retry ───────────────────────────────────────────────

def _fetch_statcast_range(start: str, end: str,
                          max_retries: int = 3,
                          retry_delay: float = 10.0) -> pd.DataFrame:
    """
    Fetch Statcast data for a date range with exponential-backoff retry.

    pybaseball can time out or return partial data when Baseball Savant is
    under load.  We retry up to `max_retries` times before giving up.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching Statcast %s → %s (attempt %d/%d)",
                        start, end, attempt, max_retries)
            df = statcast(start_dt=start, end_dt=end, verbose=False)
            if df is None or df.empty:
                logger.warning("Empty result for %s → %s", start, end)
                return pd.DataFrame()
            return df
        except Exception as exc:
            if attempt == max_retries:
                logger.error("All retries exhausted for %s → %s: %s",
                             start, end, exc)
                raise
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed (%s). Retrying in %.0fs…",
                           attempt, exc, wait)
            time.sleep(wait)
    return pd.DataFrame()  # unreachable, satisfies type checker


# ── Column filtering ──────────────────────────────────────────────────────────

def _filter_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only the columns listed in config.yaml keep_columns."""
    wanted: list[str] = []
    for group in cfg["keep_columns"].values():
        wanted.extend(group)
    available = [c for c in wanted if c in df.columns]
    missing   = [c for c in wanted if c not in df.columns]
    if missing:
        logger.debug("Columns not found in Statcast data (may be new/removed): %s",
                     missing)
    return df[available]


# ── Historical season fetch ───────────────────────────────────────────────────

def fetch_season(season: int,
                 cfg: dict | None = None,
                 force: bool = False,
                 compression: str = "snappy") -> pd.DataFrame:
    """
    Fetch (or load from cache) a complete historical MLB season.

    Parameters
    ----------
    season : int
        The season year (e.g. 2023).
    cfg : dict, optional
        Parsed config.yaml.  Loaded automatically if None.
    force : bool
        If True, re-download even if a cache file already exists.
    compression : str
        Parquet compression codec.  See _write_parquet for options.

    Returns
    -------
    pd.DataFrame
        Pitch-level Statcast data for the season.

    Cache behaviour
    ───────────────
    Historical seasons (prior years) are fetched exactly once.  The manifest
    marks them `is_complete: true` and subsequent calls return the cached
    Parquet immediately — no network request.
    """
    if cfg is None:
        cfg = _load_config()

    path     = _season_path(cfg, season)
    manifest = _read_manifest(cfg)
    key      = path.name

    # ── Cache hit ──────────────────────────────────────────────────────────
    if not force and key in manifest and manifest[key]["is_complete"] and path.exists():
        logger.info("Cache hit: %s (%d rows, %.1f MB)",
                    key,
                    manifest[key]["rows"],
                    manifest[key]["size_bytes"] / 1e6)
        return _read_parquet(path)

    # ── Cache miss — download ──────────────────────────────────────────────
    start, end = _season_date_range(season)
    df = _fetch_statcast_range(start, end)
    if df.empty:
        logger.warning("No data returned for season %d", season)
        return df

    df = _filter_columns(df, cfg)
    _write_parquet(df, path, compression=compression)

    # Only mark complete if it's a past season (current year may be ongoing)
    is_complete = (season < date.today().year)
    _update_manifest(cfg, key, path, is_complete=is_complete, season=season)

    return df


# ── Current season (incremental monthly chunks) ───────────────────────────────

def _month_date_range(year: int, month: int) -> tuple[str, str]:
    """Return the first and last day of a given month as YYYY-MM-DD strings."""
    start = date(year, month, 1)
    # Last day: first of next month minus one day
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    # Don't ask for future dates
    end = min(end, date.today())
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def refresh_current_season(cfg: dict | None = None,
                            compression: str = "snappy") -> pd.DataFrame:
    """
    Fetch any missing or incomplete months for the current MLB season and
    return the full season as a single DataFrame.

    Strategy
    ────────
    1. Walk through each calendar month from season start through today.
    2. For each month:
        - If a chunk exists in the manifest as `is_complete: true` → skip it.
        - Otherwise → re-fetch that month and overwrite the chunk.
    3. Mark a month as complete once the calendar month has fully passed.
    4. Concatenate all monthly chunks and return.

    This ensures we only ever re-download the *current* month, not the
    whole season.
    """
    if cfg is None:
        cfg = _load_config()

    current_year = cfg["seasons"]["current"]
    season_start_str, _ = _season_date_range(current_year)
    season_start = datetime.strptime(season_start_str, "%Y-%m-%d").date()
    today = date.today()

    manifest = _read_manifest(cfg)
    frames: list[pd.DataFrame] = []

    # Iterate over every month in the season up to the current month
    cursor = season_start.replace(day=1)
    while cursor <= today.replace(day=1):
        year, month = cursor.year, cursor.month
        path = _month_path(cfg, year, month)
        key  = f"current/{path.name}"

        month_is_in_past = (date(year, month, 1) < today.replace(day=1))
        cached_complete  = (key in manifest
                            and manifest[key]["is_complete"]
                            and path.exists())

        if cached_complete:
            logger.info("Cache hit (complete month): %s", key)
            frames.append(_read_parquet(path))
        else:
            start_str, end_str = _month_date_range(year, month)
            df = _fetch_statcast_range(start_str, end_str)

            if not df.empty:
                df = _filter_columns(df, cfg)
                _write_parquet(df, path, compression=compression)
                # A past month that now has data is permanently complete
                _update_manifest(cfg, key, path,
                                 is_complete=month_is_in_past,
                                 season=year)
                frames.append(df)
            else:
                logger.info("No data yet for %d-%02d (season may not have started)",
                            year, month)

        # Advance to next month
        if month == 12:
            cursor = date(year + 1, 1, 1)
        else:
            cursor = date(year, month + 1, 1)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Multi-season loader ───────────────────────────────────────────────────────

def load_seasons(seasons: list[int] | None = None,
                 cfg: dict | None = None,
                 include_current: bool = False,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load multiple seasons of Statcast data, fetching and caching anything
    that isn't already on disk.

    Parameters
    ----------
    seasons : list[int], optional
        Which historical seasons to load.  Defaults to config.yaml train list.
    cfg : dict, optional
        Parsed config.  Loaded automatically if None.
    include_current : bool
        If True, also fetch/refresh the current season and append it.
    columns : list[str], optional
        Subset of columns to load from Parquet.  If None, loads all columns
        defined in config keep_columns.  Passing a small list dramatically
        reduces RAM usage via Parquet column pruning.

    Returns
    -------
    pd.DataFrame
        All seasons concatenated into a single DataFrame.
    """
    if cfg is None:
        cfg = _load_config()
    if seasons is None:
        seasons = cfg["seasons"]["train"]

    frames: list[pd.DataFrame] = []

    for season in tqdm(seasons, desc="Loading seasons"):
        path = _season_path(cfg, season)
        manifest = _read_manifest(cfg)
        key = path.name

        if key in manifest and manifest[key]["is_complete"] and path.exists():
            # Fast path: read directly from Parquet with optional column pruning
            logger.info("Loading cached season %d (%.1f MB)",
                        season, manifest[key]["size_bytes"] / 1e6)
            frames.append(_read_parquet(path, columns=columns))
        else:
            # Slow path: download, cache, then return
            df = fetch_season(season, cfg=cfg)
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            frames.append(df)

    if include_current:
        current_df = refresh_current_season(cfg=cfg)
        if not current_df.empty:
            if columns:
                current_df = current_df[[c for c in columns
                                         if c in current_df.columns]]
            frames.append(current_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total pitches across %d seasons",
                len(combined), len(seasons) + int(include_current))
    return combined


# ── Cache inspection ──────────────────────────────────────────────────────────

def cache_summary(cfg: dict | None = None) -> pd.DataFrame:
    """
    Print a summary table of what's currently cached.

    Returns a DataFrame so it can be displayed in a notebook or piped.
    """
    if cfg is None:
        cfg = _load_config()
    manifest = _read_manifest(cfg)
    if not manifest:
        print("Cache is empty. Run fetch_season() or load_seasons() to populate it.")
        return pd.DataFrame()

    rows = []
    for key, meta in manifest.items():
        rows.append({
            "file":        key,
            "season":      meta.get("season"),
            "rows":        meta.get("rows"),
            "size_mb":     round(meta.get("size_bytes", 0) / 1e6, 1),
            "complete":    meta.get("is_complete"),
            "fetched_at":  meta.get("fetched_at"),
        })
    df = pd.DataFrame(rows).sort_values("file")
    print(df.to_string(index=False))
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch and cache Statcast data as Parquet files."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        help="Seasons to fetch (e.g. --seasons 2022 2023 2024). "
             "Defaults to config train list."
    )
    parser.add_argument(
        "--current", action="store_true",
        help="Also refresh the current season's monthly chunks."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cache exists."
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print cache summary and exit."
    )
    args = parser.parse_args()

    cfg = _load_config()

    if args.summary:
        cache_summary(cfg)
    else:
        seasons = args.seasons or cfg["seasons"]["train"]
        for s in seasons:
            fetch_season(s, cfg=cfg, force=args.force)
        if args.current:
            refresh_current_season(cfg=cfg)
        cache_summary(cfg)
