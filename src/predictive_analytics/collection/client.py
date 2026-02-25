"""Alpha Vantage API client for time-series data collection.

This module provides the :class:`AlphaVantageClient` for retrieving
market data from the Alpha Vantage REST API, as well as the
:class:`TimeSeriesData` Pydantic model for structured validation of
raw API responses.

Typical usage::

    from predictive_analytics.config.settings import get_config
    from predictive_analytics.collection.client import AlphaVantageClient

    config = get_config()
    client = AlphaVantageClient(config)
    df = client.get_time_series("AAPL")
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field

from predictive_analytics.config.logging import logger, setup_logging
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import APIRateLimitError, DataCollectionError

__all__: list[str] = [
    "TimeSeriesData",
    "AlphaVantageClient",
]

# Initialise module logger.
_logger = setup_logging()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TimeSeriesData(BaseModel):
    """Pydantic model representing a validated time-series payload.

    Attributes:
        symbol: Stock ticker or asset identifier.
        interval: Temporal granularity of the data points.
        time_series: Mapping of ISO-date strings to OHLCV dictionaries.
        last_refreshed: Timestamp of the most recent data refresh.
        output_size: Whether the response is ``"compact"`` or ``"full"``.
        time_zone: IANA time-zone identifier for the timestamps.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    symbol: str = Field(..., description="Stock symbol or identifier")
    interval: str = Field(..., description="Data interval (daily, weekly, monthly)")
    time_series: dict[str, dict[str, str]] = Field(..., description="Time series data points")
    last_refreshed: datetime = Field(..., description="Last data refresh timestamp")
    output_size: str = Field(..., description="Output size (compact or full)")
    time_zone: str = Field(..., description="Time zone of the data")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

# Mapping from API function name to response key.
_FUNCTION_TO_KEY: dict[str, str] = {
    "TIME_SERIES_DAILY": "Time Series (Daily)",
    "TIME_SERIES_WEEKLY": "Weekly Time Series",
    "TIME_SERIES_MONTHLY": "Monthly Time Series",
}


class AlphaVantageClient:
    """HTTP client for the Alpha Vantage financial-data API.

    The client manages a persistent :class:`requests.Session` and
    automatically handles API rate-limit back-off.

    Args:
        config: Validated application configuration containing API
            credentials and model settings.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config: AppConfig = config
        self.base_url: str = config.api.alpha_vantage_base_url
        self.api_key: str = config.api.alpha_vantage_api_key.get_secret_value()
        self.session: requests.Session = requests.Session()

        _logger.info("Alpha Vantage client initialized")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_time_series(
        self,
        symbol: str,
        function: str = "TIME_SERIES_DAILY",
        interval: Optional[str] = None,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """Fetch time-series data from Alpha Vantage.

        Args:
            symbol: Stock ticker symbol (e.g. ``"MSFT"``).
            function: Alpha Vantage API function name.
            interval: Intraday interval (only used when *function* is
                ``"TIME_SERIES_INTRADAY"``).
            outputsize: ``"compact"`` (last 100 points) or ``"full"``.

        Returns:
            A :class:`~pandas.DataFrame` indexed by datetime with
            columns ``open``, ``high``, ``low``, ``close``, ``volume``,
            and ``symbol``.

        Raises:
            APIRateLimitError: If the upstream rate limit is hit.
            DataCollectionError: For any other collection failure
                (network, unexpected response format, etc.).
        """
        params: dict[str, str] = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
        }

        if interval and function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval

        try:
            _logger.info("Fetching %s data for %s", function, symbol)
            response: requests.Response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            data: dict = response.json()

            # Handle API rate-limit note.
            if "Note" in data:
                _logger.warning("API call frequency limit reached. " "Waiting for 60 seconds.")
                raise APIRateLimitError(
                    "Alpha Vantage API rate limit reached. "
                    "Consider upgrading your plan or reducing request "
                    "frequency."
                )

            # Determine the response key for the requested function.
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({interval})"
            elif function in _FUNCTION_TO_KEY:
                time_series_key = _FUNCTION_TO_KEY[function]
            else:
                raise DataCollectionError(f"Unsupported API function: {function}")

            if time_series_key not in data:
                _logger.error("Unexpected API response format: %s", data)
                raise DataCollectionError(
                    f"Time series key '{time_series_key}' not found in " f"response"
                )

            # Convert to DataFrame.
            df: pd.DataFrame = pd.DataFrame.from_dict(data[time_series_key], orient="index")

            # Rename columns to remove numeric prefixes
            # (e.g. "1. open" -> "open").
            df.columns = [col.split(". ")[1] if ". " in col else col for col in df.columns]

            # Convert data types.
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Set index as datetime and sort chronologically.
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Tag rows with the source symbol.
            df["symbol"] = symbol

            _logger.info(
                "Successfully fetched %d records for %s",
                len(df),
                symbol,
            )
            return df

        except requests.RequestException as exc:
            _logger.error("Error fetching data from Alpha Vantage: %s", exc)
            raise DataCollectionError(
                f"Network error while fetching data from Alpha Vantage: " f"{exc}"
            ) from exc
        except (APIRateLimitError, DataCollectionError):
            # Re-raise domain exceptions without wrapping.
            raise
        except (ValueError, KeyError) as exc:
            _logger.error("Error processing Alpha Vantage response: %s", exc)
            raise DataCollectionError(f"Failed to process Alpha Vantage response: {exc}") from exc

    def save_data(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
    ) -> None:
        """Persist a DataFrame to CSV.

        Args:
            df: The data to save.
            file_path: Destination file path.  Parent directories are
                created automatically if they do not exist.

        Raises:
            DataCollectionError: If the file cannot be written.
        """
        try:
            resolved_path: Path = Path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(resolved_path)
            _logger.info("Data saved to %s", resolved_path)
        except IOError as exc:
            _logger.error("Error saving data to %s: %s", file_path, exc)
            raise DataCollectionError(f"Failed to save data to {file_path}: {exc}") from exc
