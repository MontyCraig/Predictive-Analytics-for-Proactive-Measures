"""Comprehensive tests for :mod:`predictive_analytics.modeling.trainer`.

Covers every public method of :class:`ModelTrainer` and the convenience
function :func:`train_and_evaluate_model`, including:

* All four training methods (SARIMA, Prophet, Auto ARIMA, Exp Smoothing)
* The unified :meth:`train_model` dispatcher (including unsupported type)
* :meth:`split_data` with default and explicit ``test_size``
* :meth:`evaluate_model` for every model type branch plus error paths
* :meth:`save_model` / :meth:`load_model` round-trip and error paths
* :meth:`plot_results` (save-path vs display, and not-trained error)
* :func:`train_and_evaluate_model` with data / file_path / default path
  branches, save_model, save_plot toggles, and the unsupported-type branch

All heavy external dependencies (SARIMAX, Prophet, pmdarima, ExponentialSmoothing)
are mocked to avoid expensive computation and optional C-extension requirements.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from predictive_analytics.config.settings import APIConfig, AppConfig, ModelConfig, SARIMAConfig
from predictive_analytics.exceptions import (
    DataNotLoadedError,
    ModelNotTrainedError,
    ModelTrainingError,
)
from predictive_analytics.modeling.trainer import ModelTrainer, train_and_evaluate_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_config() -> AppConfig:
    """Return a minimal AppConfig for SARIMA."""
    return AppConfig(
        api=APIConfig(alpha_vantage_api_key="test-key"),
        model=ModelConfig(model_type="sarima", train_size=0.8, test_size=0.2),
        sarima=SARIMAConfig(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ),
        output_dir="/tmp/test_output",
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a 100-row DataFrame with DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "close": rng.normal(100, 5, 100),
            "open": rng.normal(100, 5, 100),
            "volume": rng.integers(1000, 5000, 100).astype(float),
        },
        index=dates,
    )


@pytest.fixture()
def trainer(app_config: AppConfig, sample_df: pd.DataFrame) -> ModelTrainer:
    """Return a ModelTrainer pre-loaded with sample data."""
    return ModelTrainer(app_config, sample_df)


@pytest.fixture()
def split_trainer(trainer: ModelTrainer) -> ModelTrainer:
    """Return a ModelTrainer that already has its data split."""
    trainer.split_data()
    return trainer


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestModelTrainerInit:
    def test_init_stores_attributes(self, app_config: AppConfig, sample_df: pd.DataFrame) -> None:
        t = ModelTrainer(app_config, sample_df, target_column="open")
        assert t.data is sample_df
        assert t.target_column == "open"
        assert t.model_type == "sarima"

    def test_init_without_data(self, app_config: AppConfig) -> None:
        t = ModelTrainer(app_config)
        assert t.data is None


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_load_data_success(
        self, app_config: AppConfig, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path)
        t = ModelTrainer(app_config)
        t.load_data(csv_path)
        assert t.data is not None
        assert len(t.data) == 100

    def test_load_data_file_not_found(self, app_config: AppConfig) -> None:
        t = ModelTrainer(app_config)
        with pytest.raises(FileNotFoundError, match="File not found"):
            t.load_data("/nonexistent/file.csv")

    def test_load_data_parse_error(self, app_config: AppConfig, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("data")
        with patch("predictive_analytics.modeling.trainer.pd.read_csv") as mock_csv:
            mock_csv.side_effect = Exception("parse fail")
            t = ModelTrainer(app_config)
            with pytest.raises(ModelTrainingError, match="Failed to load data"):
                t.load_data(bad_file)


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_split_default(self, trainer: ModelTrainer) -> None:
        train, test = trainer.split_data()
        assert len(train) == 80
        assert len(test) == 20

    def test_split_custom_size(self, trainer: ModelTrainer) -> None:
        train, test = trainer.split_data(test_size=0.3)
        assert len(train) == 70
        assert len(test) == 30

    def test_split_raises_without_data(self, app_config: AppConfig) -> None:
        t = ModelTrainer(app_config)
        with pytest.raises(DataNotLoadedError, match="Data not loaded"):
            t.split_data()


# ---------------------------------------------------------------------------
# Training methods
# ---------------------------------------------------------------------------


class TestTrainSarimaModel:
    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_success(self, mock_sarimax_cls: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        split_trainer.train_sarima_model()

        mock_sarimax_cls.assert_called_once()
        mock_model.fit.assert_called_once_with(disp=False)
        assert split_trainer.model_fit is mock_fit
        assert split_trainer.model_meta["type"] == "sarima"

    def test_raises_without_split(self, trainer: ModelTrainer) -> None:
        with pytest.raises(DataNotLoadedError, match="Data not split"):
            trainer.train_sarima_model()

    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_with_exog_columns(
        self, mock_sarimax_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_model = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = MagicMock()

        split_trainer.train_sarima_model(exog_columns=["volume"])
        call_kwargs = mock_sarimax_cls.call_args
        assert call_kwargs.kwargs.get("exog") is not None or call_kwargs[1].get("exog") is not None

    def test_missing_exog_raises(self, split_trainer: ModelTrainer) -> None:
        with pytest.raises(ModelTrainingError, match="Exogenous columns not found"):
            split_trainer.train_sarima_model(exog_columns=["nonexistent_col"])

    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_fit_failure_raises(
        self, mock_sarimax_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_sarimax_cls.side_effect = RuntimeError("convergence failed")
        with pytest.raises(ModelTrainingError, match="SARIMA model training failed"):
            split_trainer.train_sarima_model()

    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_custom_order(self, mock_sarimax_cls: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_model = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = MagicMock()

        split_trainer.train_sarima_model(order=(2, 1, 0), seasonal_order=(0, 1, 1, 7))
        assert split_trainer.model_meta["order"] == (2, 1, 0)
        assert split_trainer.model_meta["seasonal_order"] == (0, 1, 1, 7)


class TestTrainProphetModel:
    @patch("predictive_analytics.modeling.trainer.Prophet")
    def test_success(self, mock_prophet_cls: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance
        mock_instance.fit.return_value = mock_instance

        split_trainer.train_prophet_model()

        mock_prophet_cls.assert_called_once()
        mock_instance.fit.assert_called_once()
        assert split_trainer.model_meta["type"] == "prophet"

    def test_raises_without_split(self, trainer: ModelTrainer) -> None:
        with pytest.raises(DataNotLoadedError, match="Data not split"):
            trainer.train_prophet_model()

    @patch("predictive_analytics.modeling.trainer.Prophet")
    def test_with_exog_columns(
        self, mock_prophet_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance
        mock_instance.fit.return_value = mock_instance

        split_trainer.train_prophet_model(exog_columns=["volume"])
        mock_instance.add_regressor.assert_called_once_with("volume")

    @patch("predictive_analytics.modeling.trainer.Prophet")
    def test_missing_exog_raises(
        self, mock_prophet_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_prophet_cls.return_value = MagicMock()
        with pytest.raises(ModelTrainingError, match="Regressor column not found"):
            split_trainer.train_prophet_model(exog_columns=["nonexistent"])

    @patch("predictive_analytics.modeling.trainer.Prophet")
    def test_fit_failure_raises(
        self, mock_prophet_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance
        mock_instance.fit.side_effect = RuntimeError("prophet fail")

        with pytest.raises(ModelTrainingError, match="Prophet model training failed"):
            split_trainer.train_prophet_model()


class TestTrainAutoArimaModel:
    @patch("predictive_analytics.modeling.trainer.pm")
    def test_success(self, mock_pm: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_model = MagicMock()
        mock_model.order = (1, 0, 1)
        mock_model.seasonal_order = (0, 1, 1, 7)
        mock_pm.auto_arima.return_value = mock_model

        split_trainer.train_auto_arima_model()

        mock_pm.auto_arima.assert_called_once()
        assert split_trainer.model_fit is mock_model
        assert split_trainer.model_meta["type"] == "auto_arima"

    def test_raises_without_split(self, trainer: ModelTrainer) -> None:
        with pytest.raises(DataNotLoadedError, match="Data not split"):
            trainer.train_auto_arima_model()

    @patch("predictive_analytics.modeling.trainer.pm")
    def test_with_exog_columns(self, mock_pm: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_model = MagicMock()
        mock_model.order = (1, 0, 0)
        mock_model.seasonal_order = (0, 0, 0, 7)
        mock_pm.auto_arima.return_value = mock_model

        split_trainer.train_auto_arima_model(exog_columns=["volume"])
        call_kwargs = mock_pm.auto_arima.call_args
        assert call_kwargs.kwargs.get("exogenous") is not None

    def test_missing_exog_raises(self, split_trainer: ModelTrainer) -> None:
        with pytest.raises(ModelTrainingError, match="Exogenous columns not found"):
            split_trainer.train_auto_arima_model(exog_columns=["nonexistent_col"])

    @patch("predictive_analytics.modeling.trainer.pm")
    def test_fit_failure_raises(self, mock_pm: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_pm.auto_arima.side_effect = RuntimeError("auto arima fail")
        with pytest.raises(ModelTrainingError, match="Auto ARIMA model training failed"):
            split_trainer.train_auto_arima_model()


class TestTrainExponentialSmoothingModel:
    @patch("predictive_analytics.modeling.trainer.ExponentialSmoothing")
    def test_success_no_seasonal(
        self, mock_es_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_model = MagicMock()
        mock_es_cls.return_value = mock_model
        mock_model.fit.return_value = MagicMock()

        split_trainer.train_exponential_smoothing_model(trend="add")

        mock_es_cls.assert_called_once()
        assert split_trainer.model_meta["type"] == "exponential_smoothing"

    def test_raises_without_split(self, trainer: ModelTrainer) -> None:
        with pytest.raises(DataNotLoadedError, match="Data not split"):
            trainer.train_exponential_smoothing_model()

    @patch("predictive_analytics.modeling.trainer.ExponentialSmoothing")
    def test_seasonal_defaults_period_to_7(
        self, mock_es_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_model = MagicMock()
        mock_es_cls.return_value = mock_model
        mock_model.fit.return_value = MagicMock()

        split_trainer.train_exponential_smoothing_model(seasonal="add", seasonal_periods=None)
        # Should default seasonal_periods to 7
        assert split_trainer.model_meta["seasonal_periods"] == 7

    @patch("predictive_analytics.modeling.trainer.ExponentialSmoothing")
    def test_explicit_seasonal_periods(
        self, mock_es_cls: MagicMock, split_trainer: ModelTrainer
    ) -> None:
        mock_model = MagicMock()
        mock_es_cls.return_value = mock_model
        mock_model.fit.return_value = MagicMock()

        split_trainer.train_exponential_smoothing_model(seasonal="add", seasonal_periods=12)
        assert split_trainer.model_meta["seasonal_periods"] == 12

    @patch("predictive_analytics.modeling.trainer.ExponentialSmoothing")
    def test_fit_failure_raises(self, mock_es_cls: MagicMock, split_trainer: ModelTrainer) -> None:
        mock_es_cls.side_effect = RuntimeError("es fail")
        with pytest.raises(
            ModelTrainingError, match="Exponential Smoothing model training failed"
        ):
            split_trainer.train_exponential_smoothing_model()


# ---------------------------------------------------------------------------
# train_model dispatcher
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_dispatches_to_sarima(self, split_trainer: ModelTrainer, mocker: Any) -> None:
        mock_method = mocker.patch.object(split_trainer, "train_sarima_model")
        split_trainer.train_model("sarima", order=(1, 0, 0))
        mock_method.assert_called_once_with(order=(1, 0, 0))

    def test_dispatches_to_prophet(self, split_trainer: ModelTrainer, mocker: Any) -> None:
        mock_method = mocker.patch.object(split_trainer, "train_prophet_model")
        split_trainer.train_model("prophet")
        mock_method.assert_called_once_with()

    def test_dispatches_to_auto_arima(self, split_trainer: ModelTrainer, mocker: Any) -> None:
        mock_method = mocker.patch.object(split_trainer, "train_auto_arima_model")
        split_trainer.train_model("auto_arima")
        mock_method.assert_called_once_with()

    def test_dispatches_to_exp_smoothing(self, split_trainer: ModelTrainer, mocker: Any) -> None:
        mock_method = mocker.patch.object(split_trainer, "train_exponential_smoothing_model")
        split_trainer.train_model("exp_smoothing")
        mock_method.assert_called_once_with()

    def test_default_model_type(self, split_trainer: ModelTrainer, mocker: Any) -> None:
        mock_method = mocker.patch.object(split_trainer, "train_sarima_model")
        split_trainer.train_model()  # Should use self.model_type = "sarima"
        mock_method.assert_called_once()

    def test_unsupported_type_raises(self, split_trainer: ModelTrainer) -> None:
        with pytest.raises(ModelTrainingError, match="Unsupported model type"):
            split_trainer.train_model("random_forest")


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_raises_when_not_trained(self, split_trainer: ModelTrainer) -> None:
        with pytest.raises(ModelNotTrainedError, match="Model not trained"):
            split_trainer.evaluate_model()

    def test_sarima_evaluation(self, split_trainer: ModelTrainer) -> None:
        # Set up mock model_fit for SARIMA
        mock_fit = MagicMock()
        forecast_result = MagicMock()
        predicted = pd.Series(
            np.ones(len(split_trainer.test_data)),
            index=split_trainer.test_data.index,
        )
        forecast_result.predicted_mean = predicted
        mock_fit.get_forecast.return_value = forecast_result

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "sarima"}

        metrics = split_trainer.evaluate_model()
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert split_trainer.predictions is not None

    def test_sarima_evaluation_with_exog(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        forecast_result = MagicMock()
        predicted = pd.Series(
            np.ones(len(split_trainer.test_data)),
            index=split_trainer.test_data.index,
        )
        forecast_result.predicted_mean = predicted
        mock_fit.get_forecast.return_value = forecast_result

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "sarima"}

        metrics = split_trainer.evaluate_model(exog_columns=["volume"])
        assert "mae" in metrics

    def test_sarima_missing_exog_raises(self, split_trainer: ModelTrainer) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "sarima"}

        with pytest.raises(ModelTrainingError, match="Exogenous columns not found"):
            split_trainer.evaluate_model(exog_columns=["nonexistent"])

    def test_prophet_evaluation(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        n_total = len(split_trainer.data)

        future_df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n_total + n_test)})
        mock_fit.make_future_dataframe.return_value = future_df

        forecast_df = pd.DataFrame(
            {"yhat": np.ones(n_total + n_test)},
            index=range(n_total + n_test),
        )
        mock_fit.predict.return_value = forecast_df

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "prophet"}

        metrics = split_trainer.evaluate_model()
        assert "mae" in metrics

    def test_prophet_evaluation_with_exog(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        n_total = len(split_trainer.data)

        future_df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n_total + n_test)})
        mock_fit.make_future_dataframe.return_value = future_df

        forecast_df = pd.DataFrame(
            {"yhat": np.ones(n_total + n_test)},
            index=range(n_total + n_test),
        )
        mock_fit.predict.return_value = forecast_df

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "prophet"}

        metrics = split_trainer.evaluate_model(exog_columns=["volume"])
        assert "mae" in metrics

    def test_auto_arima_evaluation(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        pred_values = np.ones(n_test)
        conf_int = np.column_stack([pred_values - 1, pred_values + 1])
        mock_fit.predict.return_value = (pred_values, conf_int)

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "auto_arima"}

        metrics = split_trainer.evaluate_model()
        assert "mae" in metrics

    def test_auto_arima_evaluation_with_exog(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        pred_values = np.ones(n_test)
        conf_int = np.column_stack([pred_values - 1, pred_values + 1])
        mock_fit.predict.return_value = (pred_values, conf_int)

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "auto_arima"}

        metrics = split_trainer.evaluate_model(exog_columns=["volume"])
        assert "mae" in metrics

    def test_auto_arima_missing_exog_raises(self, split_trainer: ModelTrainer) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "auto_arima"}

        with pytest.raises(ModelTrainingError, match="Exogenous columns not found"):
            split_trainer.evaluate_model(exog_columns=["nonexistent"])

    def test_exp_smoothing_evaluation(self, split_trainer: ModelTrainer) -> None:
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        predictions = pd.Series(np.ones(n_test), index=split_trainer.test_data.index)
        mock_fit.forecast.return_value = predictions

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "exponential_smoothing"}

        metrics = split_trainer.evaluate_model()
        assert "mae" in metrics

    def test_mape_with_zero_actuals(self, split_trainer: ModelTrainer) -> None:
        """When all actuals are zero, MAPE should be NaN instead of raising."""
        mock_fit = MagicMock()
        n_test = len(split_trainer.test_data)
        predictions = pd.Series(np.ones(n_test), index=split_trainer.test_data.index)
        mock_fit.forecast.return_value = predictions

        split_trainer.model_fit = mock_fit
        split_trainer.model_meta = {"type": "exponential_smoothing"}

        # Set all test actuals to zero to trigger the MAPE guard.
        split_trainer.test_data[split_trainer.target_column] = 0.0

        metrics = split_trainer.evaluate_model()
        assert np.isnan(metrics["mape"])

    def test_unsupported_type_raises(self, split_trainer: ModelTrainer) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "random_forest"}

        with pytest.raises(ModelTrainingError, match="Unsupported model type"):
            split_trainer.evaluate_model()


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


class TestSaveLoadModel:
    def test_save_raises_when_not_trained(self, trainer: ModelTrainer) -> None:
        with pytest.raises(ModelNotTrainedError, match="Model not trained"):
            trainer.save_model("/tmp/model.pkl")

    def test_save_and_load_roundtrip(self, split_trainer: ModelTrainer, tmp_path: Path) -> None:
        # Simulate a trained model
        split_trainer.model_fit = {"fake": "model"}
        split_trainer.model_meta = {"type": "sarima", "order": (1, 1, 1)}

        model_path = tmp_path / "models" / "test_model.pkl"
        split_trainer.save_model(model_path)
        assert model_path.exists()

        # Load into a new trainer
        new_trainer = ModelTrainer(split_trainer.config)
        new_trainer.load_model(model_path)
        assert new_trainer.model_fit == {"fake": "model"}
        assert new_trainer.model_meta["type"] == "sarima"

    def test_load_legacy_format(self, trainer: ModelTrainer, tmp_path: Path) -> None:
        """Loading a bare (non-dict) pickle should work as legacy format."""
        model_path = tmp_path / "legacy.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump("bare_model_object", fh)

        trainer.load_model(model_path)
        assert trainer.model_fit == "bare_model_object"
        assert trainer.model_meta == {"type": "unknown"}

    def test_load_file_not_found(self, trainer: ModelTrainer) -> None:
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            trainer.load_model("/nonexistent/model.pkl")

    def test_load_corrupt_file_raises(self, trainer: ModelTrainer, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.pkl"
        bad_file.write_bytes(b"not a pickle")
        with pytest.raises(ModelTrainingError, match="Failed to load model"):
            trainer.load_model(bad_file)

    def test_save_serialization_failure(
        self, split_trainer: ModelTrainer, tmp_path: Path, mocker: Any
    ) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "sarima"}

        mocker.patch("builtins.open", side_effect=OSError("disk full"))
        with pytest.raises(ModelTrainingError, match="Failed to save model"):
            split_trainer.save_model(tmp_path / "fail.pkl")


# ---------------------------------------------------------------------------
# plot_results
# ---------------------------------------------------------------------------


class TestPlotResults:
    def test_raises_when_not_evaluated(self, trainer: ModelTrainer) -> None:
        with pytest.raises(ModelNotTrainedError, match="Model not evaluated"):
            trainer.plot_results()

    def test_raises_when_model_fit_but_no_predictions(self, split_trainer: ModelTrainer) -> None:
        split_trainer.model_fit = MagicMock()
        # predictions is still None
        with pytest.raises(ModelNotTrainedError, match="Model not evaluated"):
            split_trainer.plot_results()

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    def test_display_mode(
        self,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        split_trainer: ModelTrainer,
    ) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "sarima"}
        split_trainer.predictions = pd.Series(
            np.ones(len(split_trainer.test_data)),
            index=split_trainer.test_data.index,
        )

        split_trainer.plot_results()
        mock_plt.show.assert_called_once()
        mock_plt.savefig.assert_not_called()

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    def test_save_mode(
        self,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        split_trainer: ModelTrainer,
        tmp_path: Path,
    ) -> None:
        split_trainer.model_fit = MagicMock()
        split_trainer.model_meta = {"type": "sarima"}
        split_trainer.predictions = pd.Series(
            np.ones(len(split_trainer.test_data)),
            index=split_trainer.test_data.index,
        )

        save_path = tmp_path / "plots" / "results.png"
        split_trainer.plot_results(save_path=save_path)
        mock_plt.savefig.assert_called_once()
        mock_plt.show.assert_not_called()


# ---------------------------------------------------------------------------
# train_and_evaluate_model convenience function
# ---------------------------------------------------------------------------


class TestTrainAndEvaluateModel:
    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.pickle.dump")
    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_with_data_provided(
        self,
        mock_sarimax_cls: MagicMock,
        mock_pickle_dump: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)

        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        # evaluate_model needs get_forecast
        n_test = int(len(sample_df) * 0.2)
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(n_test))
        mock_fit.get_forecast.return_value = forecast_result

        metrics = train_and_evaluate_model(
            config=app_config,
            data=sample_df,
            save_model=True,
            save_plot=True,
        )

        assert "mae" in metrics
        assert "rmse" in metrics
        mock_pickle_dump.assert_called_once()

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_with_file_path(
        self,
        mock_sarimax_cls: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)
        csv_path = tmp_path / "input.csv"
        sample_df.to_csv(csv_path)

        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        n_test = int(len(sample_df) * 0.2)
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(n_test))
        mock_fit.get_forecast.return_value = forecast_result

        metrics = train_and_evaluate_model(
            config=app_config,
            data=None,
            file_path=csv_path,
            save_model=False,
            save_plot=False,
        )

        assert "mae" in metrics

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_default_file_path(
        self,
        mock_sarimax_cls: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """When data=None and file_path=None, tries the default preprocessed path."""
        app_config.output_dir = str(tmp_path)
        preprocessed_dir = tmp_path / "preprocessed"
        preprocessed_dir.mkdir()
        csv_path = preprocessed_dir / "MSFT_preprocessed.csv"
        sample_df.to_csv(csv_path)

        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        n_test = int(len(sample_df) * 0.2)
        forecast_result = MagicMock()
        forecast_result.predicted_mean = pd.Series(np.ones(n_test))
        mock_fit.get_forecast.return_value = forecast_result

        metrics = train_and_evaluate_model(
            config=app_config,
            save_model=False,
            save_plot=True,
        )

        assert "mae" in metrics

    def test_unsupported_model_type_raises(
        self, app_config: AppConfig, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ModelTrainingError, match="Unsupported model type"):
            train_and_evaluate_model(
                config=app_config,
                data=sample_df,
                model_type="random_forest",
            )

    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_generic_exception_wrapped(
        self,
        mock_sarimax_cls: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """A non-ModelTrainingError exception should be wrapped."""
        app_config.output_dir = str(tmp_path)

        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_sarimax_cls.return_value = mock_model
        mock_model.fit.return_value = mock_fit

        # Make evaluate_model raise a generic exception
        mock_fit.get_forecast.side_effect = RuntimeError("unexpected")

        with pytest.raises(ModelTrainingError, match="Training and evaluation pipeline failed"):
            train_and_evaluate_model(config=app_config, data=sample_df)

    @patch("predictive_analytics.modeling.trainer.SARIMAX")
    def test_model_training_error_not_double_wrapped(
        self,
        mock_sarimax_cls: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """A ModelTrainingError should be re-raised directly, not wrapped."""
        app_config.output_dir = str(tmp_path)

        mock_sarimax_cls.side_effect = ModelTrainingError("original error")

        # The except ModelTrainingError: raise path should propagate it
        with pytest.raises(ModelTrainingError, match="SARIMA model training failed"):
            train_and_evaluate_model(config=app_config, data=sample_df)

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.Prophet")
    def test_prophet_dispatch(
        self,
        mock_prophet_cls: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)
        app_config.model.model_type = "prophet"

        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance
        mock_instance.fit.return_value = mock_instance

        n_test = int(len(sample_df) * 0.2)
        n_total = len(sample_df)
        future_df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n_total + n_test)})
        mock_instance.make_future_dataframe.return_value = future_df
        forecast_df = pd.DataFrame({"yhat": np.ones(n_total + n_test)})
        mock_instance.predict.return_value = forecast_df

        metrics = train_and_evaluate_model(
            config=app_config,
            data=sample_df,
            model_type="prophet",
            save_model=False,
            save_plot=False,
        )

        assert "mae" in metrics

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.pm")
    def test_auto_arima_dispatch(
        self,
        mock_pm: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)

        mock_model = MagicMock()
        mock_model.order = (1, 0, 1)
        mock_model.seasonal_order = (0, 1, 1, 7)
        mock_pm.auto_arima.return_value = mock_model

        n_test = int(len(sample_df) * 0.2)
        pred_values = np.ones(n_test)
        conf_int = np.column_stack([pred_values - 1, pred_values + 1])
        mock_model.predict.return_value = (pred_values, conf_int)

        metrics = train_and_evaluate_model(
            config=app_config,
            data=sample_df,
            model_type="auto_arima",
            save_model=False,
            save_plot=False,
        )

        assert "mae" in metrics

    @patch("matplotlib.pyplot")
    @patch("seaborn.set_style")
    @patch("predictive_analytics.modeling.trainer.ExponentialSmoothing")
    def test_exp_smoothing_dispatch(
        self,
        mock_es_cls: MagicMock,
        mock_set_style: MagicMock,
        mock_plt: MagicMock,
        app_config: AppConfig,
        sample_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        app_config.output_dir = str(tmp_path)

        mock_model = MagicMock()
        mock_es_cls.return_value = mock_model
        mock_fit = MagicMock()
        mock_model.fit.return_value = mock_fit

        n_test = int(len(sample_df) * 0.2)
        predictions = pd.Series(np.ones(n_test))
        mock_fit.forecast.return_value = predictions

        metrics = train_and_evaluate_model(
            config=app_config,
            data=sample_df,
            model_type="exp_smoothing",
            save_model=False,
            save_plot=False,
        )

        assert "mae" in metrics
