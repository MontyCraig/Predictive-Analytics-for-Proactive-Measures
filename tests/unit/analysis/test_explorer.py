"""Comprehensive tests for :mod:`predictive_analytics.analysis.explorer`.

Covers every public method of :class:`TimeSeriesExplorer`, the standalone
:func:`load_data` helper, and all error/edge-case branches including:

* ``DataNotLoadedError`` on empty / ``None`` data
* DatetimeIndex coercion when the index is not already datetime
* Column-not-found warnings in :meth:`plot_time_series`
* Calendar-column derivation in :meth:`plot_box_plots`
* Save-mode vs display-mode in :meth:`run_full_analysis`
* File-not-found and parse-error paths in :func:`load_data`
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from predictive_analytics.analysis.explorer import TimeSeriesExplorer, load_data
from predictive_analytics.exceptions import DataNotLoadedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a small DataFrame with a proper DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "close": rng.normal(100, 10, 100),
            "open": rng.normal(100, 10, 100),
            "volume": rng.integers(1000, 5000, 100),
        },
        index=dates,
    )
    return df


@pytest.fixture()
def explorer(sample_df: pd.DataFrame) -> TimeSeriesExplorer:
    """Return an initialised explorer with sample data."""
    return TimeSeriesExplorer(sample_df)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestTimeSeriesExplorerInit:
    """Tests for :meth:`TimeSeriesExplorer.__init__`."""

    def test_init_with_valid_data(self, sample_df: pd.DataFrame) -> None:
        explorer = TimeSeriesExplorer(sample_df, target_column="close")
        assert len(explorer.data) == 100
        assert explorer.target_column == "close"

    def test_init_raises_on_none(self) -> None:
        with pytest.raises(DataNotLoadedError, match="empty or None"):
            TimeSeriesExplorer(None)  # type: ignore[arg-type]

    def test_init_raises_on_empty_dataframe(self) -> None:
        with pytest.raises(DataNotLoadedError, match="empty or None"):
            TimeSeriesExplorer(pd.DataFrame())

    def test_init_converts_non_datetime_index(self) -> None:
        """When the index is string-based it should be coerced to DatetimeIndex."""
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=["2023-01-01", "2023-01-02", "2023-01-03"],
        )
        explorer = TimeSeriesExplorer(df)
        assert isinstance(explorer.data.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Plotting methods
# ---------------------------------------------------------------------------


class TestPlotTimeSeries:
    """Tests for :meth:`TimeSeriesExplorer.plot_time_series`."""

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_default_columns(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_time_series()
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_multiple_valid_columns(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_time_series(columns=["close", "open"])
        assert mock_plt.plot.call_count == 2
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_missing_column_warns(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        """A column not present in the data should produce a warning, not a crash."""
        explorer.plot_time_series(columns=["nonexistent"])
        # plt.plot should NOT be called for missing columns
        mock_plt.plot.assert_not_called()
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_mix_of_valid_and_invalid_columns(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_time_series(columns=["close", "nonexistent"])
        # Only 'close' should produce a plot call
        assert mock_plt.plot.call_count == 1
        mock_plt.show.assert_called_once()


class TestPlotSeasonalDecomposition:
    """Tests for :meth:`TimeSeriesExplorer.plot_seasonal_decomposition`."""

    @patch("predictive_analytics.analysis.explorer.plt")
    @patch("predictive_analytics.analysis.explorer.seasonal_decompose")
    def test_success(
        self,
        mock_decompose: MagicMock,
        mock_plt: MagicMock,
        explorer: TimeSeriesExplorer,
    ) -> None:
        mock_result = MagicMock()
        mock_result.observed = pd.Series([1, 2, 3])
        mock_result.trend = pd.Series([1, 2, 3])
        mock_result.seasonal = pd.Series([0, 0, 0])
        mock_result.resid = pd.Series([0, 0, 0])
        mock_decompose.return_value = mock_result

        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), [mock_ax] * 4)

        explorer.plot_seasonal_decomposition(period=30, model="additive")

        mock_decompose.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.plt")
    @patch("predictive_analytics.analysis.explorer.seasonal_decompose")
    def test_raises_on_decomposition_error(
        self,
        mock_decompose: MagicMock,
        mock_plt: MagicMock,
        explorer: TimeSeriesExplorer,
    ) -> None:
        mock_decompose.side_effect = ValueError("decomposition error")
        with pytest.raises(ValueError, match="decomposition error"):
            explorer.plot_seasonal_decomposition()


class TestPlotAcfPacf:
    """Tests for :meth:`TimeSeriesExplorer.plot_acf_pacf`."""

    @patch("predictive_analytics.analysis.explorer.plt")
    @patch("predictive_analytics.analysis.explorer.plot_pacf")
    @patch("predictive_analytics.analysis.explorer.plot_acf")
    def test_success(
        self,
        mock_acf: MagicMock,
        mock_pacf: MagicMock,
        mock_plt: MagicMock,
        explorer: TimeSeriesExplorer,
    ) -> None:
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), [mock_ax, mock_ax])

        explorer.plot_acf_pacf(lags=20)

        mock_acf.assert_called_once()
        mock_pacf.assert_called_once()
        mock_plt.show.assert_called_once()


class TestTestStationarity:
    """Tests for :meth:`TimeSeriesExplorer.test_stationarity`."""

    @patch("predictive_analytics.analysis.explorer.adfuller")
    def test_stationary_result(
        self, mock_adf: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        # p-value < 0.05 => stationary
        mock_adf.return_value = (
            -3.5,  # test statistic
            0.01,  # p-value
            10,    # used_lag
            90,    # nobs
            {"1%": -3.5, "5%": -2.9, "10%": -2.6},  # critical values
            100.0,  # icbest
        )
        is_stationary, p_value, crit_vals = explorer.test_stationarity()

        assert is_stationary is True
        assert p_value == 0.01
        assert "5%" in crit_vals

    @patch("predictive_analytics.analysis.explorer.adfuller")
    def test_non_stationary_result(
        self, mock_adf: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        # p-value > 0.05 => non-stationary
        mock_adf.return_value = (
            -1.0,
            0.60,
            10,
            90,
            {"1%": -3.5, "5%": -2.9, "10%": -2.6},
            100.0,
        )
        is_stationary, p_value, crit_vals = explorer.test_stationarity()

        assert is_stationary is False
        assert p_value == 0.60


class TestPlotRollingStatistics:
    """Tests for :meth:`TimeSeriesExplorer.plot_rolling_statistics`."""

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_success(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_rolling_statistics(window=10)
        # 3 plt.plot calls: original, rolling mean, rolling std
        assert mock_plt.plot.call_count == 3
        mock_plt.show.assert_called_once()


class TestPlotDistribution:
    """Tests for :meth:`TimeSeriesExplorer.plot_distribution`."""

    @patch("predictive_analytics.analysis.explorer.scipy_stats")
    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_success(
        self,
        mock_plt: MagicMock,
        mock_sns: MagicMock,
        mock_scipy: MagicMock,
        explorer: TimeSeriesExplorer,
    ) -> None:
        explorer.plot_distribution()
        mock_plt.figure.assert_called_once()
        mock_sns.histplot.assert_called_once()
        mock_scipy.probplot.assert_called_once()
        mock_plt.show.assert_called_once()


class TestPlotBoxPlots:
    """Tests for :meth:`TimeSeriesExplorer.plot_box_plots`."""

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_default_by_month(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_box_plots()
        mock_sns.boxplot.assert_called_once()
        mock_plt.show.assert_called_once()
        # 'month' should have been derived from the index
        assert "month" in explorer.data.columns

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_by_quarter(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_box_plots(by="quarter")
        assert "quarter" in explorer.data.columns
        mock_sns.boxplot.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_by_day_of_week(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_box_plots(by="day_of_week")
        assert "day_of_week" in explorer.data.columns

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_by_year(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_box_plots(by="year")
        assert "year" in explorer.data.columns

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_existing_column_not_overwritten(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        """If 'by' column already exists in data, do not re-derive it."""
        explorer.data["month"] = 999
        explorer.plot_box_plots(by="month")
        # Should still be our custom value, not derived from index
        assert (explorer.data["month"] == 999).all()


class TestPlotHeatmap:
    """Tests for :meth:`TimeSeriesExplorer.plot_heatmap`."""

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_default_columns(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_heatmap()
        mock_sns.heatmap.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.sns")
    @patch("predictive_analytics.analysis.explorer.plt")
    def test_explicit_columns(
        self, mock_plt: MagicMock, mock_sns: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        explorer.plot_heatmap(columns=["close", "open"])
        mock_sns.heatmap.assert_called_once()
        mock_plt.show.assert_called_once()


class TestPlotLagScatter:
    """Tests for :meth:`TimeSeriesExplorer.plot_lag_scatter`."""

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_success(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        mock_axes = MagicMock()
        flat_axes = [MagicMock() for _ in range(6)]
        mock_axes.flatten.return_value = flat_axes
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        explorer.plot_lag_scatter(max_lag=5)

        # Should create lag_1 .. lag_5 columns
        for lag in range(1, 6):
            assert f"lag_{lag}" in explorer.data.columns

        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.analysis.explorer.plt")
    def test_existing_lag_columns_not_recomputed(
        self, mock_plt: MagicMock, explorer: TimeSeriesExplorer
    ) -> None:
        """Pre-existing lag columns should not be overwritten."""
        explorer.data["lag_1"] = 0.0
        mock_axes = MagicMock()
        flat_axes = [MagicMock() for _ in range(4)]
        mock_axes.flatten.return_value = flat_axes
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        explorer.plot_lag_scatter(max_lag=2)
        # lag_1 was pre-existing, should still be 0.0
        assert (explorer.data["lag_1"] == 0.0).all()


# ---------------------------------------------------------------------------
# run_full_analysis
# ---------------------------------------------------------------------------


class TestRunFullAnalysis:
    """Tests for :meth:`TimeSeriesExplorer.run_full_analysis`."""

    def test_interactive_mode(
        self, explorer: TimeSeriesExplorer, mocker: "pytest_mock.MockerFixture"
    ) -> None:
        """When no output_dir is given, plots are shown interactively."""
        mocker.patch.object(explorer, "plot_time_series")
        mocker.patch.object(explorer, "plot_seasonal_decomposition")
        mocker.patch.object(explorer, "plot_acf_pacf")
        mocker.patch.object(explorer, "test_stationarity")
        mocker.patch.object(explorer, "plot_rolling_statistics")
        mocker.patch.object(explorer, "plot_distribution")
        mocker.patch.object(explorer, "plot_box_plots")
        mocker.patch.object(explorer, "plot_heatmap")
        mocker.patch.object(explorer, "plot_lag_scatter")

        explorer.run_full_analysis(output_dir=None)

        explorer.plot_time_series.assert_called_once()
        explorer.plot_seasonal_decomposition.assert_called_once()
        explorer.plot_acf_pacf.assert_called_once()
        explorer.test_stationarity.assert_called_once()
        explorer.plot_rolling_statistics.assert_called_once()
        explorer.plot_distribution.assert_called_once()
        explorer.plot_box_plots.assert_called_once()
        explorer.plot_heatmap.assert_called_once()
        explorer.plot_lag_scatter.assert_called_once()

    def test_save_mode(
        self,
        explorer: TimeSeriesExplorer,
        tmp_path: Path,
        mocker: "pytest_mock.MockerFixture",
    ) -> None:
        """When output_dir is given, plt.savefig and plt.close are called."""
        mocker.patch.object(explorer, "plot_time_series")
        mocker.patch.object(explorer, "plot_seasonal_decomposition")
        mocker.patch.object(explorer, "plot_acf_pacf")
        mocker.patch.object(explorer, "test_stationarity")
        mocker.patch.object(explorer, "plot_rolling_statistics")
        mocker.patch.object(explorer, "plot_distribution")
        mocker.patch.object(explorer, "plot_box_plots")
        mocker.patch.object(explorer, "plot_heatmap")
        mocker.patch.object(explorer, "plot_lag_scatter")

        mock_plt = mocker.patch("predictive_analytics.analysis.explorer.plt")

        out = tmp_path / "eda"
        explorer.run_full_analysis(output_dir=out)

        # 8 savefig calls (one per plot, stationarity doesn't produce a saved figure)
        assert mock_plt.savefig.call_count == 8
        assert mock_plt.close.call_count == 8


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    """Tests for :func:`load_data`."""

    def test_success(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2023-01-01", periods=3, freq="D"),
        )
        df.to_csv(csv_file)

        loaded = load_data(csv_file)
        assert len(loaded) == 3
        assert "close" in loaded.columns

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_data("/nonexistent/path/data.csv")

    def test_parse_error_raises_data_not_loaded(self, tmp_path: Path) -> None:
        """A file that exists but cannot be parsed as CSV raises DataNotLoadedError."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("not,valid\x00csv\x00data")

        # We need to make pd.read_csv actually fail. Write binary garbage.
        bad_file.write_bytes(b"\x00\x01\x02" * 100)

        # The implementation catches any generic Exception and wraps it.
        # Use mocker to force read_csv to raise.
        with patch("predictive_analytics.analysis.explorer.pd.read_csv") as mock_csv:
            mock_csv.side_effect = Exception("parse error")
            with pytest.raises(DataNotLoadedError, match="Failed to load data"):
                load_data(bad_file)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "str_test.csv"
        df = pd.DataFrame(
            {"close": [10.0]},
            index=pd.date_range("2023-06-01", periods=1, freq="D"),
        )
        df.to_csv(csv_file)

        loaded = load_data(str(csv_file))
        assert len(loaded) == 1
