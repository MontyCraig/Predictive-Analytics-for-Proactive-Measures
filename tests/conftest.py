"""Shared test fixtures for the predictive-analytics test suite."""

from __future__ import annotations

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from predictive_analytics.config.settings import APIConfig, AppConfig, ModelConfig  # noqa: E402


@pytest.fixture()
def sample_time_series() -> pd.DataFrame:
    """Return a realistic time-series DataFrame suitable for preprocessing.

    The DataFrame has a DatetimeIndex and columns: close, open, high, low,
    volume plus derived features lag_1, lag_5, rolling_mean_5, and
    log_return so that tests can assert on both raw and enriched shapes.
    """
    dates = pd.date_range(start="2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)

    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=len(dates)))
    open_ = close + rng.normal(0, 0.5, size=len(dates))
    high = np.maximum(close, open_) + rng.uniform(0, 1, size=len(dates))
    low = np.minimum(close, open_) - rng.uniform(0, 1, size=len(dates))
    volume = rng.integers(1_000_000, 10_000_000, size=len(dates)).astype(float)

    df = pd.DataFrame(
        {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
        },
        index=dates,
    )

    # Add derived features that mirror what preprocessing produces.
    df["lag_1"] = df["close"].shift(1)
    df["lag_5"] = df["close"].shift(5)
    df["rolling_mean_5"] = df["close"].rolling(window=5).mean()
    daily_return = df["close"].pct_change()
    df["log_return"] = daily_return.apply(lambda x: 0.0 if x <= -1 else np.log(x + 1))

    return df


@pytest.fixture()
def mock_config() -> AppConfig:
    """Return an AppConfig with a mocked API key and default settings."""
    api = APIConfig(
        alpha_vantage_api_key=SecretStr("test-api-key-12345"),
        alpha_vantage_base_url="https://www.alphavantage.co/query",
    )
    model = ModelConfig(
        model_type="sarima",
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )
    return AppConfig(api=api, model=model)


@pytest.fixture()
def mock_alpha_vantage_response() -> dict:
    """Return a mock Alpha Vantage TIME_SERIES_DAILY JSON response."""
    return {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": "MSFT",
            "3. Last Refreshed": "2024-01-15",
            "4. Output Size": "Full size",
            "5. Time Zone": "US/Eastern",
        },
        "Time Series (Daily)": {
            "2024-01-15": {
                "1. open": "390.0000",
                "2. high": "395.0000",
                "3. low": "388.0000",
                "4. close": "392.0000",
                "5. volume": "25000000",
            },
            "2024-01-14": {
                "1. open": "385.0000",
                "2. high": "391.0000",
                "3. low": "384.0000",
                "4. close": "389.0000",
                "5. volume": "22000000",
            },
            "2024-01-13": {
                "1. open": "382.0000",
                "2. high": "387.0000",
                "3. low": "381.0000",
                "4. close": "385.0000",
                "5. volume": "20000000",
            },
        },
    }
