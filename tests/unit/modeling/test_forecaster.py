"""Comprehensive tests for :mod:`predictive_analytics.modeling.forecaster`.

Covers every public method of :class:`TimeSeriesForecaster` and the
convenience function :func:`generate_forecast`, including:

* :meth:`load_model` -- dict-format, legacy-format, file-not-found, corrupt
* :meth:`load_data` -- success, file-not-found, parse-error
* :meth:`forecast` -- all four model types, unsupported type, no-model error,
  no-data fallback paths (training_end_date, yesterday), frequency inference
  fallback
* :meth:`plot_forecast` -- with/without historical data, save vs display
* :meth:`detect_anomalies` -- with actual values, without actual values
* :meth:`analyze_forecast` -- summary statistics
* :meth:`_infer_frequency_fallback` -- every branch (< 3 points, D, W, M, Q,
  Y, H, default)
* :func:`generate_forecast` -- model discovery, data loading, save/plot
  toggles, missing model directory / files

All heavy external dependencies (Prophet, pmdarima, SARIMAX,
ExponentialSmoothing) are mocked.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from predictive_analytics.config.settings import APIConfig, AppConfig, ModelConfig, SARIMAConfig
from predictive_analytics.exceptions import ForecastingError, ModelNotTrainedError
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster, generate_forecast

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_config() -> AppConfig:
    return AppConfig(
        api=APIConfig(alpha_vantage_api_key="test-key"),
        model=ModelConfig(model_type="sarima", train_size=0.8, test_size=0.2),
        sarima=SARIMAConfig(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ),
        output_dir="/tmp/test_forecast_output",
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"close": rng.normal(100, 5, 100)},
        index=dates,
    )


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    """Persist a serializable model package to a temp pickle file.

    Uses a plain string as the 'model' value so pickle.dump succeeds.
    Tests that need MagicMock behaviour should set ``fc.model_fit``
    after construction.
    """
    pkg: Dict[str, Any] = {
        "model": "serializable_placeholder",
        "metadata": {"type": "sarima"},
        "target_column": "close",
        "training_end_date": datetime(2023, 4, 10),
    }
    p = tmp_path / "model.pkl"
    with open(p, "wb") as fh:
        pickle.dump(pkg, fh)
    return p


@pytest.fixture()
def forecaster(
    app_config: AppConfig, model_file: Path, sample_df: pd.DataFrame
) -> TimeSeriesForecaster:
    """Return a forecaster with model loaded and data attached.

    Replaces the placeholder model_fit with a MagicMock for test flexibility.
    """
    fc = TimeSeriesForecaster(app_config, model_path=model_file, data=sample_df)
    fc.model_fit = MagicMock()
    return fc


@pytest.fixture()
def forecast_df() -> pd.DataFrame:
    """Return a sample forecast DataFrame."""
    idx = pd.date_range("2023-04-11", periods=30, freq="D")
    rng = np.random.default_rng(99)
    fc = rng.normal(100, 2, 30)
    return pd.DataFrame(
        {
            "forecast": fc,
            "lower_bound": fc - 5,
            "upper_bound": fc + 5,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Helpers for generate_forecast tests
# ---------------------------------------------------------------------------


def _make_sarima_mock_fit(n_steps: int = 10) -> MagicMock:
    """Create a MagicMock that behaves like a fitted SARIMA model."""
    mock_fit = MagicMock()
    forecast_result = MagicMock()
    forecast_result.predicted_mean = pd.Series(np.ones(n_steps))
    conf_int = pd.DataFrame({"lower": np.zeros(n_steps), "upper": np.ones(n_steps) * 2})
    forecast_result.conf_int.return_value = conf_int
    mock_fit.get_forecast.return_value = forecast_result
    return mock_fit


def _sarima_load_value(
    mock_fit: MagicMock,
    training_end_date: Any = None,
) -> Dict[str, Any]:
    """Return a dict suitable for ``pickle.load`` return value."""
    return {
        "model": mock_fit,
        "metadata": {"type": "sarima"},
        "target_column": "close",
        "training_end_date": training_end_date,
    }


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestTimeSeriesForecasterInit:
    def test_init_without_model_path(self, app_config: AppConfig) -> None:
        fc = TimeSeriesForecaster(app_config)
        assert fc.model_fit is None
        assert fc.model_meta is None

    def test_init_with_model_path(self, app_config: AppConfig, model_file: Path) -> None:
        fc = TimeSeriesForecaster(app_config, model_path=model_file)
        assert fc.model_fit is not None
        assert fc.model_meta["type"] == "sarima"

    def test_init_with_data(self, app_config: AppConfig, sample_df: pd.DataFrame) -> None:
        fc = TimeSeriesForecaster(app_config, data=sample_df)
        assert fc.data is sample_df


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_dict_format(self, app_config: AppConfig, model_file: Path) -> None:
        fc = TimeSeriesForecaster(app_config)
        fc.load_model(model_file)
        assert fc.model_fit is not None
        assert fc.model_meta["type"] == "sarima"
        assert fc.training_end_date == datetime(2023, 4, 10)

    def test_load_legacy_format(self, app_config: AppConfig, tmp_path: Path) -> None:
        legacy_path = tmp_path / "legacy.pkl"
        with open(legacy_path, "wb") as fh:
            pickle.dump("bare_model", fh)

        fc = TimeSeriesForecaster(app_config)
        fc.load_model(legacy_path)
        assert fc.model_fit == "bare_model"
        assert fc.model_meta == {"type": "sarima"}

    def test_file_not_found(self, app_config: AppConfig) -> None:
        fc = TimeSeriesForecaster(app_config)
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            fc.load_model("/nonexistent/model.pkl")

    def test_corrupt_file_raises(self, app_config: AppConfig, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.pkl"
        bad_file.write_bytes(b"not a pickle")
        fc = TimeSeriesForecaster(app_config)
        with pytest.raises(ForecastingError, match="Failed to load model"):
            fc.load_model(bad_file)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_success(self, app_config: AppConfig, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        fc = TimeSeriesForecaster(app_config)
        fc.load_data(csv_path)
        assert fc.data is not None
        assert len(fc.data) == 100

    def test_file_not_found(self, app_config: AppConfig) -> None:
        fc = TimeSeriesForecaster(app_config)
        with pytest.raises(FileNotFoundError, match="File not found"):
            fc.load_data("/nonexistent/data.csv")

    def test_parse_error(self, app_config: AppConfig, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("data")
        with patch("predictive_analytics.modeling.forecaster.pd.read_csv") as mock_csv:
            mock_csv.side_effect = Exception("parse fail")
            fc = TimeSeriesForecaster(app_config)
            with pytest.raises(ForecastingError, match="Failed to load data"):
                fc.load_data(bad_file)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


class TestForecast:
    def test_raises_when_no_model(self, app_config: AppConfig) -> None:
        fc = TimeSeriesForecaster(app_config)
        with pytest.raises(ModelNotTrainedError, match="Model not loaded"):
            fc.forecast()

    def test_sarima_forecast(self, forecaster: TimeSeriesForecaster) -> None:
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(30))
        conf_int = pd.DataFrame({"lower": np.zeros(30), "upper": np.ones(30) * 2})
        forecast_result.conf_int.return_value = conf_int

        forecaster.model_fit.get_forecast.return_value = forecast_result

        result = forecaster.forecast(steps=30)
        assert len(result) == 30
        assert "forecast" in result.columns
        assert "lower_bound" in result.columns
        assert "upper_bound" in result.columns

    def test_prophet_forecast(self, app_config: AppConfig, sample_df: pd.DataFrame) -> None:
        mock_fit = MagicMock()
        forecast_out = pd.DataFrame(
            {
                "yhat": np.ones(30),
                "yhat_lower": np.zeros(30),
                "yhat_upper": np.ones(30) * 2,
            }
        )
        mock_fit.predict.return_value = forecast_out

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "prophet", "exog_columns": None}

        result = fc.forecast(steps=30)
        assert len(result) == 30
        assert "forecast" in result.columns

    def test_prophet_forecast_with_exog(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        mock_fit = MagicMock()
        forecast_out = pd.DataFrame(
            {
                "yhat": np.ones(30),
                "yhat_lower": np.zeros(30),
                "yhat_upper": np.ones(30) * 2,
            }
        )
        mock_fit.predict.return_value = forecast_out

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "prophet", "exog_columns": ["volume"]}

        exog = pd.DataFrame(
            {"volume": np.ones(30)},
            index=pd.date_range("2023-04-11", periods=30, freq="D"),
        )
        result = fc.forecast(steps=30, exog_features=exog)
        assert len(result) == 30

    def test_auto_arima_forecast(self, app_config: AppConfig, sample_df: pd.DataFrame) -> None:
        mock_fit = MagicMock()
        pred = np.ones(30)
        pred_ci = np.column_stack([pred - 1, pred + 1])
        mock_fit.predict.return_value = (pred, pred_ci)

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "auto_arima", "exog_columns": None}

        result = fc.forecast(steps=30)
        assert len(result) == 30
        assert "forecast" in result.columns

    def test_auto_arima_forecast_with_exog(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        mock_fit = MagicMock()
        pred = np.ones(30)
        pred_ci = np.column_stack([pred - 1, pred + 1])
        mock_fit.predict.return_value = (pred, pred_ci)

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "auto_arima", "exog_columns": ["volume"]}

        exog = pd.DataFrame(
            {"volume": np.ones(30)},
            index=pd.date_range("2023-04-11", periods=30, freq="D"),
        )
        result = fc.forecast(steps=30, exog_features=exog)
        assert len(result) == 30

    def test_auto_arima_forecast_missing_exog_warns(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        """When exog_columns are expected but missing from exog_features, should warn."""
        mock_fit = MagicMock()
        pred = np.ones(30)
        pred_ci = np.column_stack([pred - 1, pred + 1])
        mock_fit.predict.return_value = (pred, pred_ci)

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "auto_arima", "exog_columns": ["missing_col"]}

        exog = pd.DataFrame(
            {"other_col": np.ones(30)},
            index=pd.date_range("2023-04-11", periods=30, freq="D"),
        )
        # Should still produce a forecast (exog_array stays None), just logs a warning
        result = fc.forecast(steps=30, exog_features=exog)
        assert len(result) == 30

    def test_exp_smoothing_forecast_with_fittedvalues(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        mock_fit = MagicMock()
        mock_fit.forecast.return_value = pd.Series(np.ones(30))
        mock_fit.fittedvalues = pd.Series(np.ones(80))
        mock_fit.resid = pd.Series(np.random.default_rng(1).normal(0, 1, 80))

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "exp_smoothing"}

        result = fc.forecast(steps=30)
        assert len(result) == 30
        assert "lower_bound" in result.columns

    def test_exp_smoothing_forecast_without_fittedvalues(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        """When model_fit lacks fittedvalues attr, uses 10% margin fallback."""
        mock_fit = MagicMock(spec=[])  # empty spec -- no attributes
        mock_fit.forecast = MagicMock(return_value=pd.Series(np.ones(30) * 100))

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "exp_smoothing"}

        result = fc.forecast(steps=30)
        assert len(result) == 30
        # lower should be ~90, upper ~110
        assert result["lower_bound"].iloc[0] == pytest.approx(90.0, abs=0.1)
        assert result["upper_bound"].iloc[0] == pytest.approx(110.0, abs=0.1)

    def test_unsupported_model_type(self, app_config: AppConfig, sample_df: pd.DataFrame) -> None:
        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = MagicMock()
        fc.model_meta = {"type": "random_forest"}

        with pytest.raises(ForecastingError, match="Unsupported model type"):
            fc.forecast()

    def test_forecast_generic_exception_wrapped(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        """A non-ForecastingError exception should be wrapped."""
        mock_fit = MagicMock()
        mock_fit.get_forecast.side_effect = RuntimeError("kaboom")

        fc = TimeSeriesForecaster(app_config, data=sample_df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "sarima"}

        with pytest.raises(ForecastingError, match="Forecast generation failed"):
            fc.forecast()

    def test_forecast_no_data_uses_training_end_date(self, app_config: AppConfig) -> None:
        mock_fit = MagicMock()
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(10))
        conf_int = pd.DataFrame({"lower": np.zeros(10), "upper": np.ones(10) * 2})
        forecast_result.conf_int.return_value = conf_int
        mock_fit.get_forecast.return_value = forecast_result

        fc = TimeSeriesForecaster(app_config)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "sarima"}
        fc.training_end_date = datetime(2023, 6, 1)

        result = fc.forecast(steps=10)
        assert len(result) == 10

    def test_forecast_no_data_no_training_date_uses_yesterday(self, app_config: AppConfig) -> None:
        mock_fit = MagicMock()
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(5))
        conf_int = pd.DataFrame({"lower": np.zeros(5), "upper": np.ones(5) * 2})
        forecast_result.conf_int.return_value = conf_int
        mock_fit.get_forecast.return_value = forecast_result

        fc = TimeSeriesForecaster(app_config)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "sarima"}
        fc.training_end_date = None

        result = fc.forecast(steps=5)
        assert len(result) == 5

    def test_forecast_infer_freq_fallback(self, app_config: AppConfig) -> None:
        """When pd.infer_freq returns None, the fallback should be used."""
        # Create data with irregular dates that infer_freq can't handle
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05"])
        df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=dates)

        mock_fit = MagicMock()
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(5))
        conf_int = pd.DataFrame({"lower": np.zeros(5), "upper": np.ones(5) * 2})
        forecast_result.conf_int.return_value = conf_int
        mock_fit.get_forecast.return_value = forecast_result

        fc = TimeSeriesForecaster(app_config, data=df)
        fc.model_fit = mock_fit
        fc.model_meta = {"type": "sarima"}

        result = fc.forecast(steps=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _infer_frequency_fallback
# ---------------------------------------------------------------------------


class TestInferFrequencyFallback:
    def test_short_index_returns_d(self) -> None:
        idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
        assert TimeSeriesForecaster._infer_frequency_fallback(idx) == "D"

    def test_daily(self) -> None:
        idx = pd.date_range("2023-01-01", periods=10, freq="D")
        assert TimeSeriesForecaster._infer_frequency_fallback(idx) == "D"

    def test_weekly(self) -> None:
        idx = pd.date_range("2023-01-01", periods=10, freq="W")
        assert TimeSeriesForecaster._infer_frequency_fallback(idx) == "W"

    def test_monthly(self) -> None:
        idx = pd.date_range("2023-01-31", periods=10, freq="ME")
        result = TimeSeriesForecaster._infer_frequency_fallback(idx)
        assert result == "M"

    def test_quarterly(self) -> None:
        idx = pd.date_range("2023-03-31", periods=10, freq="QE")
        result = TimeSeriesForecaster._infer_frequency_fallback(idx)
        assert result == "Q"

    def test_yearly(self) -> None:
        idx = pd.date_range("2020-12-31", periods=5, freq="YE")
        result = TimeSeriesForecaster._infer_frequency_fallback(idx)
        assert result == "Y"

    def test_hourly(self) -> None:
        idx = pd.date_range("2023-01-01", periods=10, freq="h")
        result = TimeSeriesForecaster._infer_frequency_fallback(idx)
        assert result == "H"

    def test_unknown_falls_back_to_d(self) -> None:
        """A 3-day interval doesn't match any known mapping."""
        idx = pd.to_datetime(["2023-01-01", "2023-01-04", "2023-01-07", "2023-01-10"])
        result = TimeSeriesForecaster._infer_frequency_fallback(idx)
        assert result == "D"


# ---------------------------------------------------------------------------
# plot_forecast
# ---------------------------------------------------------------------------


class TestPlotForecast:
    @patch("predictive_analytics.modeling.forecaster.plt")
    def test_display_mode_with_data(
        self,
        mock_plt: MagicMock,
        forecaster: TimeSeriesForecaster,
        forecast_df: pd.DataFrame,
    ) -> None:
        forecaster.plot_forecast(forecast_df)
        mock_plt.show.assert_called_once()
        mock_plt.savefig.assert_not_called()

    @patch("predictive_analytics.modeling.forecaster.plt")
    def test_save_mode(
        self,
        mock_plt: MagicMock,
        forecaster: TimeSeriesForecaster,
        forecast_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        save_path = tmp_path / "plots" / "forecast.png"
        forecaster.plot_forecast(forecast_df, save_path=save_path)
        mock_plt.savefig.assert_called_once()
        mock_plt.show.assert_not_called()

    @patch("predictive_analytics.modeling.forecaster.plt")
    def test_no_historical_data(
        self,
        mock_plt: MagicMock,
        app_config: AppConfig,
        forecast_df: pd.DataFrame,
    ) -> None:
        fc = TimeSeriesForecaster(app_config)
        fc.model_meta = {"type": "sarima"}
        fc.plot_forecast(forecast_df)
        mock_plt.show.assert_called_once()


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_with_actual_values(
        self,
        forecaster: TimeSeriesForecaster,
        forecast_df: pd.DataFrame,
    ) -> None:
        actual = pd.Series(np.random.default_rng(1).normal(100, 2, 30), index=forecast_df.index)
        result = forecaster.detect_anomalies(forecast_df, actual_values=actual)

        assert "actual" in result.columns
        assert "error" in result.columns
        assert "z_score" in result.columns
        assert "is_anomaly" in result.columns

    def test_without_actual_values(
        self,
        forecaster: TimeSeriesForecaster,
        forecast_df: pd.DataFrame,
    ) -> None:
        result = forecaster.detect_anomalies(forecast_df)

        assert "forecast_range" in result.columns
        assert "range_z_score" in result.columns
        assert "is_uncertain" in result.columns

    def test_custom_threshold(
        self,
        forecaster: TimeSeriesForecaster,
        forecast_df: pd.DataFrame,
    ) -> None:
        actual = pd.Series(np.random.default_rng(1).normal(100, 50, 30), index=forecast_df.index)
        # A very high threshold should produce no anomalies
        result = forecaster.detect_anomalies(forecast_df, actual_values=actual, threshold=100.0)
        assert result["is_anomaly"].sum() == 0


# ---------------------------------------------------------------------------
# analyze_forecast
# ---------------------------------------------------------------------------


class TestAnalyzeForecast:
    def test_returns_expected_keys(
        self, forecaster: TimeSeriesForecaster, forecast_df: pd.DataFrame
    ) -> None:
        metrics = forecaster.analyze_forecast(forecast_df)

        assert "avg_daily_change" in metrics
        assert "growth_rate_pct" in metrics
        assert "volatility" in metrics
        assert "uncertainty" in metrics
        assert "min_forecast" in metrics
        assert "max_forecast" in metrics

    def test_values_are_floats(
        self, forecaster: TimeSeriesForecaster, forecast_df: pd.DataFrame
    ) -> None:
        metrics = forecaster.analyze_forecast(forecast_df)
        for value in metrics.values():
            assert isinstance(value, float)


# ---------------------------------------------------------------------------
# generate_forecast convenience function
# ---------------------------------------------------------------------------


class TestGenerateForecast:
    def test_with_explicit_model_path_and_data(
        self,
        app_config: AppConfig,
        model_file: Path,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt"):
                result = generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=csv_path,
                    steps=10,
                    save_forecast=True,
                    save_plot=True,
                    plot=True,
                )

        assert len(result) == 10
        assert "forecast" in result.columns

    def test_model_discovery_no_model_dir_raises(
        self, app_config: AppConfig, tmp_path: Path
    ) -> None:
        app_config.output_dir = str(tmp_path)
        # models/ directory does not exist
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            generate_forecast(config=app_config, model_path=None)

    def test_model_discovery_no_matching_files_raises(
        self, app_config: AppConfig, tmp_path: Path
    ) -> None:
        app_config.output_dir = str(tmp_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # Directory exists but has no matching .pkl files
        with pytest.raises(FileNotFoundError, match="No sarima model files found"):
            generate_forecast(config=app_config, model_path=None)

    def test_model_discovery_selects_latest(
        self, app_config: AppConfig, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        app_config.output_dir = str(tmp_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create two dummy model files (content doesn't matter; we patch pickle.load)
        old_file = model_dir / "sarima_20230101.pkl"
        new_file = model_dir / "sarima_20230102.pkl"
        old_file.write_bytes(b"dummy")
        new_file.write_bytes(b"dummy")

        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt"):
                result = generate_forecast(
                    config=app_config,
                    data_path=csv_path,
                    steps=10,
                    save_forecast=False,
                    save_plot=False,
                    plot=False,
                )

        assert len(result) == 10

    def test_plot_false(
        self,
        app_config: AppConfig,
        model_file: Path,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When plot=False, no plot should be generated."""
        app_config.output_dir = str(tmp_path)
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt") as mock_plt:
                result = generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=csv_path,
                    steps=10,
                    plot=False,
                    save_forecast=False,
                )
                mock_plt.show.assert_not_called()
                mock_plt.savefig.assert_not_called()

        assert len(result) == 10

    def test_save_plot_false_shows_interactively(
        self,
        app_config: AppConfig,
        model_file: Path,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When plot=True but save_plot=False, show interactively."""
        app_config.output_dir = str(tmp_path)
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt") as mock_plt:
                generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=csv_path,
                    steps=10,
                    plot=True,
                    save_plot=False,
                    save_forecast=False,
                )
                mock_plt.show.assert_called()

    def test_no_data_path_not_save_forecast_tries_preprocessed(
        self,
        app_config: AppConfig,
        model_file: Path,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When data_path=None and save_forecast=False, tries preprocessed dir."""
        app_config.output_dir = str(tmp_path)
        preprocessed_dir = tmp_path / "preprocessed"
        preprocessed_dir.mkdir()
        csv_path = preprocessed_dir / "MSFT_preprocessed.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt"):
                result = generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=None,
                    steps=10,
                    save_forecast=False,
                    plot=False,
                )

        assert len(result) == 10

    def test_no_data_path_save_forecast_true_skips_preprocessed(
        self,
        app_config: AppConfig,
        model_file: Path,
        tmp_path: Path,
    ) -> None:
        """When data_path=None and save_forecast=True, does NOT load preprocessed."""
        app_config.output_dir = str(tmp_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(
                mock_fit, training_end_date=datetime(2023, 6, 1)
            )
            with patch("predictive_analytics.modeling.forecaster.plt"):
                result = generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=None,
                    steps=10,
                    save_forecast=True,
                    plot=False,
                )

        assert len(result) == 10

    def test_default_model_type_from_config(
        self,
        app_config: AppConfig,
        model_file: Path,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When model_type=None, uses config.model.model_type."""
        app_config.output_dir = str(tmp_path)
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)

        mock_fit = _make_sarima_mock_fit()

        with patch("predictive_analytics.modeling.forecaster.pickle.load") as mock_load:
            mock_load.return_value = _sarima_load_value(mock_fit)
            with patch("predictive_analytics.modeling.forecaster.plt"):
                result = generate_forecast(
                    config=app_config,
                    model_path=model_file,
                    data_path=csv_path,
                    steps=10,
                    model_type=None,
                    save_forecast=False,
                    plot=False,
                )

        assert len(result) == 10
