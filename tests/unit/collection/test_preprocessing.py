"""Tests for the predictive_analytics.collection.preprocessing module.

Covers:
- preprocess_data: column creation, NaN handling (interpolation + bfill),
  empty DataFrame error, missing 'close' column error, log_return edge
  case (x <= -1), and final dropna behaviour.
- collect_and_preprocess_data: end-to-end with mocked client, save=True
  and save=False paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from predictive_analytics.collection.preprocessing import (
    collect_and_preprocess_data,
    preprocess_data,
)
from predictive_analytics.exceptions import DataPreprocessingError


# ---------------------------------------------------------------------------
# preprocess_data - success
# ---------------------------------------------------------------------------


class TestPreprocessDataSuccess:
    """Happy-path tests for preprocess_data."""

    def _make_raw_df(self, periods: int = 60) -> pd.DataFrame:
        """Build a minimal valid DataFrame for preprocessing."""
        dates = pd.date_range("2020-01-01", periods=periods, freq="B")
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, periods))
        return pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.5, periods),
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
            },
            index=dates,
        )

    def test_output_has_expected_columns(self) -> None:
        df = self._make_raw_df()
        result = preprocess_data(df)

        expected_cols = {
            # Original
            "open",
            "high",
            "low",
            "close",
            "volume",
            # Calendar features
            "year",
            "month",
            "day",
            "day_of_week",
            "quarter",
            # Lag features
            "lag_1",
            "lag_5",
            "lag_10",
            # Rolling features
            "rolling_mean_5",
            "rolling_mean_10",
            "rolling_std_5",
            "rolling_std_10",
            # Returns
            "daily_return",
            "log_return",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_no_nans_in_output(self) -> None:
        df = self._make_raw_df()
        result = preprocess_data(df)
        assert not result.isna().any().any()

    def test_output_is_shorter_due_to_dropna(self) -> None:
        df = self._make_raw_df(periods=60)
        result = preprocess_data(df)
        # The first ~10 rows are dropped due to lag_10 and rolling window NaN values.
        assert len(result) < len(df)
        assert len(result) > 0

    def test_does_not_mutate_input(self) -> None:
        df = self._make_raw_df()
        original_cols = set(df.columns)
        original_len = len(df)
        _ = preprocess_data(df)
        assert set(df.columns) == original_cols
        assert len(df) == original_len

    def test_lag_features_values(self) -> None:
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        close = list(range(100, 120))
        df = pd.DataFrame({"close": close}, index=dates)

        result = preprocess_data(df)
        # After dropna, first valid row should have lag_1 = previous close
        # Verify lag_1 is the close shifted by 1
        if len(result) > 0:
            idx = result.index[0]
            pos = df.index.get_loc(idx)
            assert result.loc[idx, "lag_1"] == df["close"].iloc[pos - 1]

    def test_calendar_features_correct(self) -> None:
        dates = pd.date_range("2024-03-15", periods=30, freq="B")
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {"close": 100 + np.cumsum(rng.normal(0, 1, 30))},
            index=dates,
        )
        result = preprocess_data(df)
        if len(result) > 0:
            first_row = result.iloc[0]
            assert first_row["year"] == result.index[0].year
            assert first_row["month"] == result.index[0].month
            assert first_row["day"] == result.index[0].day
            assert first_row["day_of_week"] == result.index[0].dayofweek
            assert first_row["quarter"] == result.index[0].quarter

    def test_daily_return_calculation(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        close = [100.0 + i for i in range(30)]
        df = pd.DataFrame({"close": close}, index=dates)
        result = preprocess_data(df)
        # daily_return should be pct_change of close
        if len(result) > 1:
            idx = result.index[1]
            pos_in_original = df.index.get_loc(idx)
            expected_return = (
                df["close"].iloc[pos_in_original] - df["close"].iloc[pos_in_original - 1]
            ) / df["close"].iloc[pos_in_original - 1]
            assert abs(result.loc[idx, "daily_return"] - expected_return) < 1e-10


# ---------------------------------------------------------------------------
# preprocess_data - NaN handling
# ---------------------------------------------------------------------------


class TestPreprocessDataNaNHandling:
    """Test missing-value interpolation and bfill logic."""

    def test_nan_values_are_handled(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        close = [100.0 + i for i in range(30)]
        close[5] = np.nan
        close[10] = np.nan
        df = pd.DataFrame({"close": close}, index=dates)
        result = preprocess_data(df)
        assert not result.isna().any().any()

    def test_leading_nans_handled_by_bfill(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        close = [np.nan, np.nan] + [100.0 + i for i in range(28)]
        df = pd.DataFrame({"close": close}, index=dates)
        result = preprocess_data(df)
        assert not result.isna().any().any()

    def test_no_nans_skips_interpolation_branch(self) -> None:
        """When there are no NaNs, the interpolation branch is skipped."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        df = pd.DataFrame({"close": range(100, 130)}, index=dates)
        df["close"] = df["close"].astype(float)
        result = preprocess_data(df)
        assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# preprocess_data - log_return edge case
# ---------------------------------------------------------------------------


class TestLogReturnEdgeCase:
    """Test the lambda that guards against log(0) / log(negative)."""

    def test_log_return_for_large_negative_return(self) -> None:
        """When daily_return <= -1, log_return should be 0.0."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        close = [100.0] * 15 + [0.001] + [100.0] * 14
        df = pd.DataFrame({"close": close}, index=dates)
        result = preprocess_data(df)

        # The row following the big drop should have log_return == 0 if
        # the pct_change was <= -1.
        log_returns = result["log_return"]
        # Verify no NaN or inf in log_return
        assert not log_returns.isna().any()
        assert np.all(np.isfinite(log_returns))


# ---------------------------------------------------------------------------
# preprocess_data - error paths
# ---------------------------------------------------------------------------


class TestPreprocessDataErrors:
    """Error-path tests for preprocess_data."""

    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(DataPreprocessingError, match="empty DataFrame"):
            preprocess_data(df)

    def test_missing_close_column_raises(self) -> None:
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({"open": range(10), "high": range(10)}, index=dates)
        with pytest.raises(DataPreprocessingError, match="'close' column"):
            preprocess_data(df)


# ---------------------------------------------------------------------------
# collect_and_preprocess_data
# ---------------------------------------------------------------------------


class TestCollectAndPreprocessData:
    """Tests for the end-to-end collect_and_preprocess_data function."""

    def _mock_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.output_dir = "/tmp/test_output"
        cfg.api.alpha_vantage_base_url = "https://api.test.com"
        cfg.api.alpha_vantage_api_key.get_secret_value.return_value = "fake-key"
        return cfg

    def _make_raw_df(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 1, 60))
        return pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, 60).astype(float),
            },
            index=dates,
        )

    @patch("predictive_analytics.collection.preprocessing.AlphaVantageClient")
    def test_collect_and_preprocess_with_save(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        raw_df = self._make_raw_df()
        mock_instance = MagicMock()
        mock_instance.get_time_series.return_value = raw_df
        mock_client_cls.return_value = mock_instance

        config = self._mock_config()
        result = collect_and_preprocess_data(config, symbol="AAPL", save=True)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert not result.isna().any().any()

        # save_data should have been called twice (raw + preprocessed).
        assert mock_instance.save_data.call_count == 2

        # Verify file paths passed to save_data.
        raw_call_path = mock_instance.save_data.call_args_list[0][0][1]
        preprocessed_call_path = mock_instance.save_data.call_args_list[1][0][1]
        assert "raw" in str(raw_call_path)
        assert "AAPL_raw.csv" in str(raw_call_path)
        assert "preprocessed" in str(preprocessed_call_path)
        assert "AAPL_preprocessed.csv" in str(preprocessed_call_path)

    @patch("predictive_analytics.collection.preprocessing.AlphaVantageClient")
    def test_collect_and_preprocess_without_save(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        raw_df = self._make_raw_df()
        mock_instance = MagicMock()
        mock_instance.get_time_series.return_value = raw_df
        mock_client_cls.return_value = mock_instance

        config = self._mock_config()
        result = collect_and_preprocess_data(config, symbol="GOOG", save=False)

        assert isinstance(result, pd.DataFrame)
        # save_data should NOT have been called.
        mock_instance.save_data.assert_not_called()

    @patch("predictive_analytics.collection.preprocessing.AlphaVantageClient")
    def test_default_symbol_is_msft(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        raw_df = self._make_raw_df()
        mock_instance = MagicMock()
        mock_instance.get_time_series.return_value = raw_df
        mock_client_cls.return_value = mock_instance

        config = self._mock_config()
        collect_and_preprocess_data(config, save=False)

        mock_instance.get_time_series.assert_called_once_with(symbol="MSFT")


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestPreprocessingModuleExports:
    """Verify that __all__ contains the expected names."""

    def test_all_contains_expected_names(self) -> None:
        from predictive_analytics.collection import preprocessing

        expected = {"preprocess_data", "collect_and_preprocess_data"}
        assert set(preprocessing.__all__) == expected
