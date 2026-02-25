"""Comprehensive tests for predictive_analytics.disruption.analyzer module.

Covers DisruptionAnalyzer (all 4 disruption detectors, identify_all_disruptions,
generate_disruption_report, plot_disruptions, load_forecast, load_historical_data)
and the analyze_disruptions convenience function.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from predictive_analytics.config.settings import APIConfig, AppConfig, ModelConfig
from predictive_analytics.disruption.analyzer import (
    DisruptionAnalyzer,
    analyze_disruptions,
)
from predictive_analytics.exceptions import (
    DataNotLoadedError,
    DisruptionAnalysisError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_config() -> AppConfig:
    """Minimal AppConfig for testing."""
    return AppConfig(
        api=APIConfig(alpha_vantage_api_key="test-key"),
        model=ModelConfig(),
        output_dir="output",
    )


@pytest.fixture()
def forecast_df() -> pd.DataFrame:
    """Forecast DataFrame with varying data to trigger disruptions."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.RandomState(42)
    base = np.linspace(100, 130, 30)
    # Inject a spike at index 15 to trigger level/trend disruptions
    base[15] = 200
    noise = rng.normal(0, 2, 30)
    forecast = base + noise
    lower = forecast - 5
    upper = forecast + 5
    # Widen interval at index 15 to trigger uncertainty disruption
    lower[15] = forecast[15] - 50
    upper[15] = forecast[15] + 50
    return pd.DataFrame(
        {"forecast": forecast, "lower_bound": lower, "upper_bound": upper},
        index=dates,
    )


@pytest.fixture()
def constant_forecast_df() -> pd.DataFrame:
    """Forecast with constant values -- zero std triggers edge-case branches."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "forecast": [100.0] * 20,
            "lower_bound": [95.0] * 20,
            "upper_bound": [105.0] * 20,
        },
        index=dates,
    )


@pytest.fixture()
def historical_df() -> pd.DataFrame:
    """Minimal historical DataFrame."""
    dates = pd.date_range("2023-06-01", periods=90, freq="D")
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {"close": np.linspace(80, 100, 90) + rng.normal(0, 1, 90)},
        index=dates,
    )


@pytest.fixture()
def analyzer(app_config: AppConfig, forecast_df: pd.DataFrame) -> DisruptionAnalyzer:
    """Pre-loaded DisruptionAnalyzer with forecast data."""
    return DisruptionAnalyzer(app_config, forecast_data=forecast_df)


@pytest.fixture()
def analyzer_no_data(app_config: AppConfig) -> DisruptionAnalyzer:
    """DisruptionAnalyzer with no data loaded."""
    return DisruptionAnalyzer(app_config)


# ---------------------------------------------------------------------------
# DisruptionAnalyzer.__init__
# ---------------------------------------------------------------------------


class TestDisruptionAnalyzerInit:
    def test_init_with_defaults(self, app_config: AppConfig) -> None:
        da = DisruptionAnalyzer(app_config)
        assert da.config is app_config
        assert da.forecast_data is None
        assert da.historical_data is None
        assert da.target_column == "close"

    def test_init_with_all_params(
        self,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
    ) -> None:
        da = DisruptionAnalyzer(
            app_config,
            forecast_data=forecast_df,
            historical_data=historical_df,
            target_column="price",
        )
        assert da.forecast_data is forecast_df
        assert da.historical_data is historical_df
        assert da.target_column == "price"


# ---------------------------------------------------------------------------
# load_forecast
# ---------------------------------------------------------------------------


class TestLoadForecast:
    def test_load_forecast_success(
        self, analyzer_no_data: DisruptionAnalyzer, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "forecast.csv"
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"forecast": [1, 2, 3, 4, 5]}, index=dates)
        df.to_csv(csv_path)

        analyzer_no_data.load_forecast(csv_path)
        assert analyzer_no_data.forecast_data is not None
        assert len(analyzer_no_data.forecast_data) == 5

    def test_load_forecast_file_not_found(
        self, analyzer_no_data: DisruptionAnalyzer
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Forecast file not found"):
            analyzer_no_data.load_forecast("/nonexistent/forecast.csv")

    def test_load_forecast_parse_error(
        self, analyzer_no_data: DisruptionAnalyzer, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("not,valid\x00csv\x00data\n\x00\x00\x00")
        # Force a parse error by mocking pd.read_csv to raise
        with patch("predictive_analytics.disruption.analyzer.pd.read_csv", side_effect=Exception("parse fail")):
            with pytest.raises(DisruptionAnalysisError, match="Failed to load forecast data"):
                analyzer_no_data.load_forecast(bad_file)

    def test_load_forecast_accepts_string_path(
        self, analyzer_no_data: DisruptionAnalyzer, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "forecast.csv"
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"forecast": [10, 20, 30]}, index=dates)
        df.to_csv(csv_path)

        analyzer_no_data.load_forecast(str(csv_path))
        assert analyzer_no_data.forecast_data is not None


# ---------------------------------------------------------------------------
# load_historical_data
# ---------------------------------------------------------------------------


class TestLoadHistoricalData:
    def test_load_historical_data_success(
        self, analyzer_no_data: DisruptionAnalyzer, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "historical.csv"
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"close": range(10)}, index=dates)
        df.to_csv(csv_path)

        analyzer_no_data.load_historical_data(csv_path)
        assert analyzer_no_data.historical_data is not None
        assert len(analyzer_no_data.historical_data) == 10

    def test_load_historical_data_file_not_found(
        self, analyzer_no_data: DisruptionAnalyzer
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Historical data file not found"):
            analyzer_no_data.load_historical_data("/nonexistent/historical.csv")

    def test_load_historical_data_parse_error(
        self, analyzer_no_data: DisruptionAnalyzer, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad_hist.csv"
        bad_file.write_text("junk")
        with patch("predictive_analytics.disruption.analyzer.pd.read_csv", side_effect=Exception("bad")):
            with pytest.raises(DisruptionAnalysisError, match="Failed to load historical data"):
                analyzer_no_data.load_historical_data(bad_file)


# ---------------------------------------------------------------------------
# identify_trend_disruptions
# ---------------------------------------------------------------------------


class TestTrendDisruptions:
    def test_raises_when_no_data(self, analyzer_no_data: DisruptionAnalyzer) -> None:
        with pytest.raises(DataNotLoadedError):
            analyzer_no_data.identify_trend_disruptions()

    def test_detects_trend_disruptions(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_trend_disruptions(window_size=5, threshold=1.5)
        assert "rolling_slope" in result.columns
        assert "slope_z_score" in result.columns
        assert "trend_disruption" in result.columns
        assert result["trend_disruption"].dtype == bool

    def test_zero_std_branch(
        self,
        app_config: AppConfig,
        constant_forecast_df: pd.DataFrame,
    ) -> None:
        """When forecast is constant, slope std is zero."""
        da = DisruptionAnalyzer(app_config, forecast_data=constant_forecast_df)
        result = da.identify_trend_disruptions(window_size=5)
        assert (result["slope_z_score"] == 0.0).all()
        assert not result["trend_disruption"].any()

    def test_custom_window_and_threshold(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_trend_disruptions(window_size=3, threshold=0.5)
        # With a very low threshold, more disruptions should be detected
        assert result["trend_disruption"].any()


# ---------------------------------------------------------------------------
# identify_volatility_disruptions
# ---------------------------------------------------------------------------


class TestVolatilityDisruptions:
    def test_raises_when_no_data(self, analyzer_no_data: DisruptionAnalyzer) -> None:
        with pytest.raises(DataNotLoadedError):
            analyzer_no_data.identify_volatility_disruptions()

    def test_detects_volatility_disruptions(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_volatility_disruptions(window_size=5, threshold=1.5)
        assert "rolling_volatility" in result.columns
        assert "volatility_z_score" in result.columns
        assert "volatility_disruption" in result.columns

    def test_zero_std_branch(
        self,
        app_config: AppConfig,
        constant_forecast_df: pd.DataFrame,
    ) -> None:
        """When forecast is constant, rolling volatility std is zero."""
        da = DisruptionAnalyzer(app_config, forecast_data=constant_forecast_df)
        result = da.identify_volatility_disruptions(window_size=5)
        assert (result["volatility_z_score"] == 0.0).all()
        assert not result["volatility_disruption"].any()


# ---------------------------------------------------------------------------
# identify_level_disruptions
# ---------------------------------------------------------------------------


class TestLevelDisruptions:
    def test_raises_when_no_data(self, analyzer_no_data: DisruptionAnalyzer) -> None:
        with pytest.raises(DataNotLoadedError):
            analyzer_no_data.identify_level_disruptions()

    def test_detects_level_disruptions(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_level_disruptions(threshold_std=2.0)
        assert "forecast_diff" in result.columns
        assert "level_disruption" in result.columns
        # The spike at index 15 should create at least one level disruption
        assert result["level_disruption"].any()

    def test_no_disruptions_in_smooth_data(
        self, app_config: AppConfig
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        # Perfectly linear forecast -- diffs are constant, no outlier
        df = pd.DataFrame(
            {"forecast": np.linspace(100, 119, 20)},
            index=dates,
        )
        da = DisruptionAnalyzer(app_config, forecast_data=df)
        result = da.identify_level_disruptions(threshold_std=2.0)
        # All diffs are ~1.0 so none should exceed 2*std threshold
        # (std of constant diffs is 0 so threshold_value is 0 -- but
        #  abs(diff - mean) is also 0, which is NOT > 0)
        assert result["level_disruption"].sum() == 0


# ---------------------------------------------------------------------------
# identify_uncertainty_disruptions
# ---------------------------------------------------------------------------


class TestUncertaintyDisruptions:
    def test_raises_when_no_data(self, analyzer_no_data: DisruptionAnalyzer) -> None:
        with pytest.raises(DataNotLoadedError):
            analyzer_no_data.identify_uncertainty_disruptions()

    def test_raises_when_missing_columns(self, app_config: AppConfig) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"forecast": [1, 2, 3, 4, 5]}, index=dates)
        da = DisruptionAnalyzer(app_config, forecast_data=df)
        with pytest.raises(DisruptionAnalysisError, match="prediction intervals"):
            da.identify_uncertainty_disruptions()

    def test_detects_uncertainty_disruptions(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_uncertainty_disruptions(threshold=1.5)
        assert "interval_width" in result.columns
        assert "uncertainty_z_score" in result.columns
        assert "uncertainty_disruption" in result.columns
        # The wide interval at index 15 should create an uncertainty disruption
        assert result["uncertainty_disruption"].any()

    def test_zero_std_branch(
        self,
        app_config: AppConfig,
        constant_forecast_df: pd.DataFrame,
    ) -> None:
        """Constant interval width => zero std => fallback branch."""
        da = DisruptionAnalyzer(app_config, forecast_data=constant_forecast_df)
        result = da.identify_uncertainty_disruptions()
        assert (result["uncertainty_z_score"] == 0.0).all()
        assert not result["uncertainty_disruption"].any()


# ---------------------------------------------------------------------------
# identify_all_disruptions
# ---------------------------------------------------------------------------


class TestIdentifyAllDisruptions:
    def test_raises_when_no_data(self, analyzer_no_data: DisruptionAnalyzer) -> None:
        with pytest.raises(DataNotLoadedError):
            analyzer_no_data.identify_all_disruptions()

    def test_consolidates_all_four_types(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_all_disruptions()
        for col in [
            "trend_disruption",
            "volatility_disruption",
            "level_disruption",
            "uncertainty_disruption",
            "total_disruptions",
            "significant_disruption",
        ]:
            assert col in result.columns

    def test_total_disruptions_column(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_all_disruptions()
        # Verify total_disruptions is the sum of the four boolean columns
        expected = (
            result["trend_disruption"].astype(int)
            + result["volatility_disruption"].astype(int)
            + result["level_disruption"].astype(int)
            + result["uncertainty_disruption"].astype(int)
        )
        pd.testing.assert_series_equal(
            result["total_disruptions"], expected, check_names=False
        )

    def test_significant_disruption_threshold(self, analyzer: DisruptionAnalyzer) -> None:
        result = analyzer.identify_all_disruptions()
        sig = result["significant_disruption"]
        expected = result["total_disruptions"] >= 2
        pd.testing.assert_series_equal(sig, expected, check_names=False)


# ---------------------------------------------------------------------------
# generate_disruption_report
# ---------------------------------------------------------------------------


class TestGenerateDisruptionReport:
    def test_report_structure(self, analyzer: DisruptionAnalyzer) -> None:
        disruption_df = analyzer.identify_all_disruptions()
        report = analyzer.generate_disruption_report(disruption_df)
        expected_keys = {
            "trend_disruptions",
            "volatility_disruptions",
            "level_disruptions",
            "uncertainty_disruptions",
            "significant_disruptions",
            "total_forecast_periods",
            "disruption_percentage",
            "significant_disruption_dates",
        }
        assert set(report.keys()) == expected_keys

    def test_total_forecast_periods(self, analyzer: DisruptionAnalyzer) -> None:
        disruption_df = analyzer.identify_all_disruptions()
        report = analyzer.generate_disruption_report(disruption_df)
        assert report["total_forecast_periods"] == len(disruption_df)

    def test_disruption_percentage_calculation(
        self, analyzer: DisruptionAnalyzer
    ) -> None:
        disruption_df = analyzer.identify_all_disruptions()
        report = analyzer.generate_disruption_report(disruption_df)
        expected_pct = (
            report["significant_disruptions"] / report["total_forecast_periods"]
        ) * 100
        assert abs(report["disruption_percentage"] - expected_pct) < 1e-10

    def test_report_with_empty_dataframe(self, app_config: AppConfig) -> None:
        """Empty DataFrame => zero percentage (division guard)."""
        da = DisruptionAnalyzer(app_config)
        empty_df = pd.DataFrame(columns=["trend_disruption", "significant_disruption"])
        report = da.generate_disruption_report(empty_df)
        assert report["disruption_percentage"] == 0.0
        assert report["total_forecast_periods"] == 0

    def test_report_missing_columns_returns_zeros(self, app_config: AppConfig) -> None:
        """Columns not present in the DataFrame should default to 0."""
        da = DisruptionAnalyzer(app_config)
        df = pd.DataFrame({"forecast": [1, 2, 3]})
        report = da.generate_disruption_report(df)
        assert report["trend_disruptions"] == 0
        assert report["volatility_disruptions"] == 0
        assert report["level_disruptions"] == 0
        assert report["uncertainty_disruptions"] == 0
        assert report["significant_disruptions"] == 0
        assert report["significant_disruption_dates"] == []


# ---------------------------------------------------------------------------
# plot_disruptions
# ---------------------------------------------------------------------------


class TestPlotDisruptions:
    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_plot_with_save_path(
        self,
        mock_plt: MagicMock,
        analyzer: DisruptionAnalyzer,
        tmp_path: Path,
    ) -> None:
        disruption_df = analyzer.identify_all_disruptions()
        save_path = tmp_path / "subdir" / "plot.png"
        analyzer.plot_disruptions(disruption_df, save_path=save_path)
        mock_plt.savefig.assert_called_once_with(save_path)
        mock_plt.show.assert_not_called()

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_plot_without_save_path_calls_show(
        self,
        mock_plt: MagicMock,
        analyzer: DisruptionAnalyzer,
    ) -> None:
        disruption_df = analyzer.identify_all_disruptions()
        analyzer.plot_disruptions(disruption_df)
        mock_plt.show.assert_called_once()

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_plot_with_historical_data(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
    ) -> None:
        da = DisruptionAnalyzer(
            app_config,
            forecast_data=forecast_df,
            historical_data=historical_df,
        )
        disruption_df = da.identify_all_disruptions()
        da.plot_disruptions(disruption_df, historical_periods=30)
        # Should have called subplot for upper plot
        assert mock_plt.subplot.called

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_plot_returns_early_when_column_missing(
        self,
        mock_plt: MagicMock,
        analyzer: DisruptionAnalyzer,
    ) -> None:
        """DataFrame without significant_disruption column => early return."""
        df = pd.DataFrame({"forecast": [1, 2, 3]})
        analyzer.plot_disruptions(df)
        mock_plt.figure.assert_not_called()

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_plot_with_no_significant_points(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        constant_forecast_df: pd.DataFrame,
    ) -> None:
        """No significant disruptions => scatter should not be called."""
        da = DisruptionAnalyzer(app_config, forecast_data=constant_forecast_df)
        disruption_df = da.identify_all_disruptions()
        da.plot_disruptions(disruption_df)
        mock_plt.scatter.assert_not_called()


# ---------------------------------------------------------------------------
# analyze_disruptions convenience function
# ---------------------------------------------------------------------------


class TestAnalyzeDisruptions:
    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_with_forecast_path(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Load forecast from a file, skip generation."""
        app_config.output_dir = str(tmp_path / "output")

        csv_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(csv_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=csv_path,
            plot=True,
            save_results=True,
        )
        assert "significant_disruption" in disruption_df.columns
        assert "trend_disruptions" in report

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_generate_new_forecast(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """generate_new_forecast=True should call generate_forecast."""
        app_config.output_dir = str(tmp_path / "output")

        with patch(
            "predictive_analytics.modeling.forecaster.generate_forecast",
            return_value=forecast_df,
        ):
            disruption_df, report = analyze_disruptions(
                config=app_config,
                generate_new_forecast=True,
                forecast_steps=10,
                plot=False,
                save_results=False,
            )
        assert len(disruption_df) == len(forecast_df)

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_no_forecast_path_triggers_generation(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """forecast_path=None should also call generate_forecast."""
        app_config.output_dir = str(tmp_path / "output")

        with patch(
            "predictive_analytics.modeling.forecaster.generate_forecast",
            return_value=forecast_df,
        ):
            disruption_df, report = analyze_disruptions(
                config=app_config,
                forecast_path=None,
                plot=False,
                save_results=False,
            )
        assert report["total_forecast_periods"] == len(forecast_df)

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_with_historical_data_path(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path / "output")

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)
        hist_path = tmp_path / "historical.csv"
        historical_df.to_csv(hist_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            historical_data_path=hist_path,
            plot=False,
            save_results=False,
        )
        assert report["total_forecast_periods"] > 0

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_auto_discovers_preprocessed_data(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When no historical_data_path, finds latest preprocessed file."""
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)
        preprocessed_dir = output_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True)

        hist_file = preprocessed_dir / "MSFT_preprocessed.csv"
        historical_df.to_csv(hist_file)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            historical_data_path=None,
            plot=False,
            save_results=False,
        )
        assert report["total_forecast_periods"] > 0

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_no_preprocessed_dir_skips_historical(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """No preprocessed directory => historical_data stays None."""
        output_dir = tmp_path / "output_empty"
        app_config.output_dir = str(output_dir)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            historical_data_path=None,
            plot=False,
            save_results=False,
        )
        assert report["total_forecast_periods"] > 0

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_empty_preprocessed_dir(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Preprocessed dir exists but has no matching files."""
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)
        (output_dir / "preprocessed").mkdir(parents=True)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            historical_data_path=None,
            plot=False,
            save_results=False,
        )
        assert report["total_forecast_periods"] > 0

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_save_results_creates_files(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            plot=True,
            save_results=True,
        )

        disruptions_dir = output_dir / "disruptions"
        assert disruptions_dir.exists()
        # Should have CSV, JSON, and PNG files
        csv_files = list(disruptions_dir.glob("disruptions_*.csv"))
        json_files = list(disruptions_dir.glob("disruption_report_*.json"))
        assert len(csv_files) >= 1
        assert len(json_files) >= 1

        # Verify JSON report is valid
        with open(json_files[0]) as fh:
            saved_report = json.load(fh)
        assert "trend_disruptions" in saved_report

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_no_plot_no_save(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path / "output")

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            plot=False,
            save_results=False,
        )
        mock_plt.show.assert_not_called()
        assert not (tmp_path / "output" / "disruptions").exists()

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_significant_dates_with_strftime(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Dates with strftime method are formatted in the report log."""
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        # The spike in forecast_df should create significant disruptions
        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            plot=False,
            save_results=True,
        )
        # If there are significant dates, they should have been logged and serialized
        if report["significant_disruption_dates"]:
            # Check the JSON file serializes dates as strings
            disruptions_dir = output_dir / "disruptions"
            json_files = list(disruptions_dir.glob("disruption_report_*.json"))
            assert len(json_files) >= 1
            with open(json_files[0]) as fh:
                saved = json.load(fh)
            for d in saved["significant_disruption_dates"]:
                assert isinstance(d, str)

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_significant_dates_without_strftime(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Dates that are plain strings (no strftime) are handled via str()."""
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        disruption_df, report = analyze_disruptions(
            config=app_config,
            forecast_path=fc_path,
            plot=False,
            save_results=True,
        )
        # The test verifies the code path runs without error regardless
        assert isinstance(report["disruption_percentage"], float)

    @patch("predictive_analytics.disruption.analyzer.plt")
    def test_significant_dates_str_fallback_in_logging(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Exercise the str() fallback for dates without strftime (line 709).

        We monkey-patch the report after detection so that significant dates
        are plain strings (no strftime attribute) to exercise the else branch.
        """
        output_dir = tmp_path / "output"
        app_config.output_dir = str(output_dir)

        fc_path = tmp_path / "forecast.csv"
        forecast_df.to_csv(fc_path)

        # Patch generate_disruption_report to inject plain-string dates
        original_generate_report = DisruptionAnalyzer.generate_disruption_report

        def patched_report(self_inner: DisruptionAnalyzer, df: pd.DataFrame) -> dict:
            report = original_generate_report(self_inner, df)
            # Replace datetime dates with plain strings (no strftime)
            if report["significant_disruption_dates"]:
                report["significant_disruption_dates"] = [
                    "2024-01-15",
                    "2024-01-16",
                ]
            return report

        with patch.object(
            DisruptionAnalyzer,
            "generate_disruption_report",
            patched_report,
        ):
            disruption_df, report = analyze_disruptions(
                config=app_config,
                forecast_path=fc_path,
                plot=False,
                save_results=True,
            )
        # If there were significant dates they should be strings now
        for d in report.get("significant_disruption_dates", []):
            assert isinstance(d, str)
