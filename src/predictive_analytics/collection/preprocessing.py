"""Data preprocessing utilities for time-series data.

This module provides functions for transforming raw market data into
feature-rich DataFrames suitable for model training and evaluation.

Key changes from the legacy module:

* **Bug fix**: ``numpy`` is now imported at module level (previously
  only available inside ``__main__``, causing ``NameError`` at
  runtime when ``np.log`` was called in :func:`preprocess_data`).
* **Deprecated API fix**: ``DataFrame.fillna(method="bfill")`` has been
  replaced with ``DataFrame.bfill()``.

Typical usage::

    from predictive_analytics.collection.preprocessing import (
        collect_and_preprocess_data,
    )
    from predictive_analytics.config.settings import get_config

    config = get_config()
    df = collect_and_preprocess_data(config, symbol="AAPL")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictive_analytics.collection.client import AlphaVantageClient
from predictive_analytics.config.logging import setup_logging
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import DataPreprocessingError

__all__: list[str] = [
    "preprocess_data",
    "collect_and_preprocess_data",
]

# Initialise module logger.
_logger = setup_logging()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw time-series data into a feature-enriched DataFrame.

    The following transformations are applied (in order):

    1. Missing-value imputation via linear interpolation followed by
       backward fill.
    2. Date-based calendar features (year, month, day, day-of-week,
       quarter).
    3. Lag features (1, 5, and 10 periods) on the ``close`` column.
    4. Rolling mean and standard-deviation features (windows of 5 and
       10) on the ``close`` column.
    5. Daily simple and log returns derived from ``close``.
    6. Rows containing any remaining ``NaN`` values are dropped.

    Args:
        df: Raw time-series DataFrame with a :class:`~pandas.DatetimeIndex`
            and at least a ``close`` column.

    Returns:
        A new DataFrame with all original and derived columns, with
        ``NaN``-containing rows removed.

    Raises:
        DataPreprocessingError: If the input DataFrame is empty or
            lacks the required ``close`` column.
    """
    if df.empty:
        raise DataPreprocessingError(
            "Cannot preprocess an empty DataFrame."
        )

    if "close" not in df.columns:
        raise DataPreprocessingError(
            "Input DataFrame must contain a 'close' column."
        )

    # Work on a copy to avoid mutating the caller's data.
    data: pd.DataFrame = df.copy()

    # 1. Handle missing values with linear interpolation.
    if data.isna().any().any():
        _logger.info("Handling missing values with linear interpolation")
        data = data.interpolate(method="linear")

    # Handle remaining NaN values (e.g. at the beginning of the series).
    # NOTE: ``fillna(method="bfill")`` is deprecated in pandas >= 2.0;
    # use ``bfill()`` instead.
    data = data.bfill()

    # 2. Add date-based features.
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["day_of_week"] = data.index.dayofweek
    data["quarter"] = data.index.quarter

    # 3. Add lag features.
    _logger.info("Adding lag features")
    data["lag_1"] = data["close"].shift(1)
    data["lag_5"] = data["close"].shift(5)
    data["lag_10"] = data["close"].shift(10)

    # 4. Add rolling statistics.
    _logger.info("Adding rolling statistics")
    data["rolling_mean_5"] = data["close"].rolling(window=5).mean()
    data["rolling_mean_10"] = data["close"].rolling(window=10).mean()
    data["rolling_std_5"] = data["close"].rolling(window=5).std()
    data["rolling_std_10"] = data["close"].rolling(window=10).std()

    # 5. Calculate returns.
    _logger.info("Calculating returns")
    data["daily_return"] = data["close"].pct_change()
    data["log_return"] = data["daily_return"].apply(
        lambda x: 0.0 if x <= -1 else np.log(x + 1)
    )

    # 6. Drop rows with NaN values from lag and rolling features.
    data = data.dropna()

    _logger.info(
        "Preprocessing complete. Data shape: %s", data.shape
    )
    return data


# ---------------------------------------------------------------------------
# End-to-end collection + preprocessing
# ---------------------------------------------------------------------------


def collect_and_preprocess_data(
    config: AppConfig,
    symbol: str = "MSFT",
    save: bool = True,
) -> pd.DataFrame:
    """Collect raw data from Alpha Vantage and preprocess it.

    This convenience function chains :meth:`AlphaVantageClient.get_time_series`
    and :func:`preprocess_data`, optionally persisting both the raw and
    preprocessed DataFrames to CSV.

    Args:
        config: Validated application configuration.
        symbol: Stock ticker symbol to fetch.
        save: Whether to persist raw and preprocessed CSVs under
            ``config.output_dir``.

    Returns:
        The preprocessed :class:`~pandas.DataFrame`.

    Raises:
        DataCollectionError: If the API request fails.
        DataPreprocessingError: If preprocessing encounters an error.
    """
    client = AlphaVantageClient(config)

    # Fetch data.
    df: pd.DataFrame = client.get_time_series(symbol=symbol)

    if save:
        raw_file_path: Path = (
            Path(config.output_dir) / "raw" / f"{symbol}_raw.csv"
        )
        client.save_data(df, raw_file_path)

    # Preprocess data.
    preprocessed_df: pd.DataFrame = preprocess_data(df)

    if save:
        preprocessed_file_path: Path = (
            Path(config.output_dir)
            / "preprocessed"
            / f"{symbol}_preprocessed.csv"
        )
        client.save_data(preprocessed_df, preprocessed_file_path)

    return preprocessed_df
