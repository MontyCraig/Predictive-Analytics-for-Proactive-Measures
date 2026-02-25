"""Tests for the predictive_analytics.collection.client module.

Covers:
- TimeSeriesData Pydantic model validation.
- AlphaVantageClient initialisation.
- get_time_series: success (daily/weekly/monthly/intraday), rate-limit
  handling, HTTP errors, missing time-series key, unsupported function,
  response processing errors.
- save_data: success and IOError paths.
- Column renaming logic (numeric prefix stripping).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from predictive_analytics.collection.client import (
    AlphaVantageClient,
    TimeSeriesData,
    _FUNCTION_TO_KEY,
)
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import (
    APIRateLimitError,
    DataCollectionError,
)


# ---------------------------------------------------------------------------
# TimeSeriesData model
# ---------------------------------------------------------------------------


class TestTimeSeriesData:
    """Tests for the TimeSeriesData Pydantic model."""

    def test_valid_construction(self) -> None:
        ts = TimeSeriesData(
            symbol="AAPL",
            interval="daily",
            time_series={"2024-01-15": {"1. open": "100"}},
            last_refreshed=datetime(2024, 1, 15),
            output_size="full",
            time_zone="US/Eastern",
        )
        assert ts.symbol == "AAPL"
        assert ts.interval == "daily"
        assert "2024-01-15" in ts.time_series
        assert ts.output_size == "full"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(Exception):
            TimeSeriesData(symbol="AAPL")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AlphaVantageClient initialisation
# ---------------------------------------------------------------------------


class TestAlphaVantageClientInit:
    """Test client constructor."""

    def test_init_sets_attributes(self, mock_config: AppConfig) -> None:
        client = AlphaVantageClient(mock_config)
        assert client.config is mock_config
        assert client.base_url == mock_config.api.alpha_vantage_base_url
        assert client.api_key == "test-api-key-12345"
        assert isinstance(client.session, requests.Session)


# ---------------------------------------------------------------------------
# get_time_series - success paths
# ---------------------------------------------------------------------------


class TestGetTimeSeriesSuccess:
    """Happy-path tests for fetching time-series data."""

    def test_daily_fetch(
        self,
        mock_config: AppConfig,
        mock_alpha_vantage_response: dict,
        mocker,
    ) -> None:
        client = AlphaVantageClient(mock_config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_alpha_vantage_response
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        df = client.get_time_series("MSFT")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Column renaming should have stripped "1. " prefixes.
        assert "open" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
        assert "symbol" in df.columns
        assert (df["symbol"] == "MSFT").all()
        # Index should be sorted chronologically.
        assert df.index.is_monotonic_increasing

    def test_weekly_fetch(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        response_data = {
            "Weekly Time Series": {
                "2024-01-12": {
                    "1. open": "100",
                    "2. high": "110",
                    "3. low": "95",
                    "4. close": "105",
                    "5. volume": "5000000",
                },
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        df = client.get_time_series("AAPL", function="TIME_SERIES_WEEKLY")
        assert len(df) == 1
        assert "close" in df.columns

    def test_monthly_fetch(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        response_data = {
            "Monthly Time Series": {
                "2024-01-31": {
                    "1. open": "200",
                    "2. high": "210",
                    "3. low": "195",
                    "4. close": "205",
                    "5. volume": "50000000",
                },
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        df = client.get_time_series("GOOG", function="TIME_SERIES_MONTHLY")
        assert len(df) == 1

    def test_intraday_fetch(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        response_data = {
            "Time Series (5min)": {
                "2024-01-15 16:00:00": {
                    "1. open": "100",
                    "2. high": "101",
                    "3. low": "99",
                    "4. close": "100.5",
                    "5. volume": "1000000",
                },
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        df = client.get_time_series(
            "TSLA",
            function="TIME_SERIES_INTRADAY",
            interval="5min",
        )
        assert len(df) == 1

    def test_columns_without_dot_prefix_unchanged(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        """If the API returns columns without '1. ' prefix, they pass through."""
        response_data = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "open": "100",
                    "high": "110",
                    "low": "95",
                    "close": "105",
                    "volume": "5000000",
                },
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        df = client.get_time_series("TEST")
        assert "open" in df.columns


# ---------------------------------------------------------------------------
# get_time_series - error paths
# ---------------------------------------------------------------------------


class TestGetTimeSeriesErrors:
    """Error-path tests for get_time_series."""

    def test_rate_limit_raises(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        response_data = {"Note": "Thank you for using Alpha Vantage!"}
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        with pytest.raises(APIRateLimitError, match="rate limit"):
            client.get_time_series("MSFT")

    def test_http_error_raises_data_collection_error(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        with pytest.raises(DataCollectionError, match="Network error"):
            client.get_time_series("MSFT")

    def test_connection_error_raises_data_collection_error(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        client = AlphaVantageClient(mock_config)
        mocker.patch.object(
            client.session,
            "get",
            side_effect=requests.ConnectionError("Connection refused"),
        )

        with pytest.raises(DataCollectionError, match="Network error"):
            client.get_time_series("MSFT")

    def test_unsupported_function_raises(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        response_data = {"some_key": "some_value"}
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        with pytest.raises(DataCollectionError, match="Unsupported API function"):
            client.get_time_series("MSFT", function="UNKNOWN_FUNCTION")

    def test_missing_time_series_key_raises(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        """Response is valid JSON but lacks the expected time-series key."""
        response_data = {"Meta Data": {"1. Symbol": "MSFT"}}
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        with pytest.raises(DataCollectionError, match="not found in"):
            client.get_time_series("MSFT")

    def test_value_error_during_processing(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        """Non-numeric data triggers a ValueError during conversion."""
        response_data = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "not_a_number",
                    "2. high": "also_bad",
                    "3. low": "nope",
                    "4. close": "nah",
                    "5. volume": "invalid",
                },
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        # pd.to_numeric with non-numeric strings won't raise by default
        # (it coerces to NaN), so this path is harder to trigger.
        # Instead, force a ValueError via mocking.
        mocker.patch(
            "predictive_analytics.collection.client.pd.DataFrame.from_dict",
            side_effect=ValueError("bad data"),
        )
        with pytest.raises(DataCollectionError, match="Failed to process"):
            client.get_time_series("MSFT")

    def test_key_error_during_processing(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        """A KeyError during processing is caught and re-raised."""
        response_data = {
            "Time Series (Daily)": {
                "2024-01-15": {"1. open": "100"},
            },
        }
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        mocker.patch(
            "predictive_analytics.collection.client.pd.DataFrame.from_dict",
            side_effect=KeyError("missing_col"),
        )
        with pytest.raises(DataCollectionError, match="Failed to process"):
            client.get_time_series("MSFT")

    def test_intraday_without_interval(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        """Intraday with no interval -- should still attempt the call."""
        # The key constructed would be "Time Series (None)" which won't exist.
        response_data = {"some_other_key": "value"}
        client = AlphaVantageClient(mock_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mocker.patch.object(client.session, "get", return_value=mock_resp)

        with pytest.raises(DataCollectionError, match="not found in"):
            client.get_time_series("MSFT", function="TIME_SERIES_INTRADAY")


# ---------------------------------------------------------------------------
# _FUNCTION_TO_KEY mapping
# ---------------------------------------------------------------------------


class TestFunctionToKeyMapping:
    """Verify the internal function-to-response-key mapping."""

    def test_daily_key(self) -> None:
        assert _FUNCTION_TO_KEY["TIME_SERIES_DAILY"] == "Time Series (Daily)"

    def test_weekly_key(self) -> None:
        assert _FUNCTION_TO_KEY["TIME_SERIES_WEEKLY"] == "Weekly Time Series"

    def test_monthly_key(self) -> None:
        assert _FUNCTION_TO_KEY["TIME_SERIES_MONTHLY"] == "Monthly Time Series"


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------


class TestSaveData:
    """Tests for AlphaVantageClient.save_data."""

    def test_save_data_creates_file(
        self,
        mock_config: AppConfig,
        tmp_path: Path,
    ) -> None:
        client = AlphaVantageClient(mock_config)
        df = pd.DataFrame({"close": [100, 200]})
        file_path = tmp_path / "subdir" / "data.csv"
        client.save_data(df, file_path)

        assert file_path.exists()
        loaded = pd.read_csv(file_path)
        assert len(loaded) == 2

    def test_save_data_with_string_path(
        self,
        mock_config: AppConfig,
        tmp_path: Path,
    ) -> None:
        client = AlphaVantageClient(mock_config)
        df = pd.DataFrame({"value": [1, 2, 3]})
        file_path = str(tmp_path / "string_path.csv")
        client.save_data(df, file_path)
        assert Path(file_path).exists()

    def test_save_data_io_error_raises(
        self,
        mock_config: AppConfig,
        mocker,
    ) -> None:
        client = AlphaVantageClient(mock_config)
        df = pd.DataFrame({"close": [100]})

        mocker.patch.object(
            pd.DataFrame,
            "to_csv",
            side_effect=IOError("Permission denied"),
        )

        with pytest.raises(DataCollectionError, match="Failed to save"):
            client.save_data(df, "/some/invalid/path.csv")


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestClientModuleExports:
    """Verify that __all__ contains the expected names."""

    def test_all_contains_expected_names(self) -> None:
        from predictive_analytics.collection import client

        expected = {"TimeSeriesData", "AlphaVantageClient"}
        assert set(client.__all__) == expected
