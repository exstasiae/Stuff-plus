"""
src/utils/io.py
─────────────────────────────────────────────────────────────────────────────
Shared I/O helpers used across the project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet(path: str | Path,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
    columns : list[str], optional
        If provided, only these columns are read from disk (column pruning).
        This is significantly faster than reading all columns and then
        filtering in pandas — the filtering happens at the storage layer.
    """
    return pq.read_table(str(path), columns=columns).to_pandas()


def write_parquet(df: pd.DataFrame,
                  path: str | Path,
                  compression: str = "snappy",
                  row_group_size: int = 200_000) -> None:
    """
    Write a pandas DataFrame to Parquet.

    Parameters
    ----------
    compression : {"snappy", "zstd", "gzip", "none"}
        snappy  — fastest, ~3-4x size reduction (default)
        zstd    — ~20% smaller than snappy, slightly slower reads
        gzip    — smallest, slowest
        none    — no compression, maximum read speed
    row_group_size : int
        Number of rows per internal Parquet row group.  Smaller values
        allow faster filtered reads (row group statistics enable skipping).
        Larger values improve compression ratio.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path),
                   compression=compression,
                   row_group_size=row_group_size)


def read_csv_cached(csv_path: str | Path,
                    cache_path: str | Path | None = None,
                    **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV file, writing a Parquet cache on first load.
    Subsequent loads read the Parquet (much faster).

    Useful for ad-hoc CSV files that aren't fetched via pybaseball.
    """
    csv_path   = Path(csv_path)
    cache_path = Path(cache_path) if cache_path else csv_path.with_suffix(".parquet")

    if cache_path.exists() and cache_path.stat().st_mtime > csv_path.stat().st_mtime:
        return read_parquet(cache_path)

    df = pd.read_csv(csv_path, **read_csv_kwargs)
    write_parquet(df, cache_path)
    return df
