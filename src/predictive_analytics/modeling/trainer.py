"""Model selection and training for time series prediction.

Provides the ``ModelTrainer`` class which supports four model families --
SARIMA, Prophet, Auto ARIMA, and Exponential Smoothing -- together with
evaluation, serialization, and visualization utilities.

A convenience function :func:`train_and_evaluate_model` wraps the full
train-evaluate-save workflow for common use-cases.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from predictive_analytics.config.logging import setup_logging
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import (
    DataNotLoadedError,
    ModelNotTrainedError,
    ModelTrainingError,
)
from predictive_analytics.types import MetricsDict, SARIMAOrder, SARIMASeasonalOrder

__all__: list[str] = ["ModelTrainer", "train_and_evaluate_model"]

logger: logging.Logger = setup_logging()


class ModelTrainer:
    """Trainer for time series forecasting models.

    Supports SARIMA, Facebook Prophet, Auto ARIMA (``pmdarima``), and
    Holt-Winters Exponential Smoothing.  The typical lifecycle is::

        trainer = ModelTrainer(config, data)
        trainer.split_data()
        trainer.train_model()
        metrics = trainer.evaluate_model()
        trainer.save_model("model.pkl")
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: AppConfig,
        data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ) -> None:
        """Initialise the model trainer.

        Args:
            config: Application configuration.
            data: Time series data.  When *None* data must be loaded via
                :meth:`load_data` before training.
            target_column: Name of the column to predict.
        """
        self.config: AppConfig = config
        self.data: Optional[pd.DataFrame] = data
        self.target_column: str = target_column
        self.model: Any = None
        self.model_fit: Any = None
        self.model_meta: Dict[str, Any] = {}
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.Series] = None
        self.model_type: str = config.model.model_type.lower()

        logger.info("ModelTrainer initialized with model type: %s", self.model_type)

    # ------------------------------------------------------------------
    # Data loading / splitting
    # ------------------------------------------------------------------

    def load_data(self, file_path: Union[str, Path]) -> None:
        """Load time-series data from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ModelTrainingError: If the file cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Loading data from %s", file_path)

        try:
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info("Loaded %d records from %s", len(self.data), file_path)
        except Exception as exc:
            logger.error("Error loading data from %s: %s", file_path, exc)
            raise ModelTrainingError(f"Failed to load data from {file_path}") from exc

    def split_data(self, test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets chronologically.

        Args:
            test_size: Proportion of data reserved for testing.  Falls back
                to the value in :attr:`config` when *None*.

        Returns:
            A ``(train_data, test_data)`` tuple.

        Raises:
            DataNotLoadedError: If data has not been loaded yet.
        """
        if self.data is None:
            logger.error("Data not loaded. Please load data first.")
            raise DataNotLoadedError("Data not loaded. Please load data first.")

        if test_size is None:
            test_size = self.config.model.test_size

        split_idx: int = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()

        logger.info(
            "Data split: train_size=%d, test_size=%d",
            len(self.train_data),
            len(self.test_data),
        )

        return self.train_data, self.test_data

    # ------------------------------------------------------------------
    # Model-specific training methods
    # ------------------------------------------------------------------

    def train_sarima_model(
        self,
        order: Optional[SARIMAOrder] = None,
        seasonal_order: Optional[SARIMASeasonalOrder] = None,
        exog_columns: Optional[List[str]] = None,
    ) -> None:
        """Train a SARIMA model.

        Args:
            order: ``(p, d, q)`` SARIMA order.  Defaults to config value.
            seasonal_order: ``(P, D, Q, s)`` seasonal order.  Defaults to
                config value.
            exog_columns: Exogenous variable column names.

        Raises:
            DataNotLoadedError: If training data is unavailable.
            ModelTrainingError: If the model fails to fit.
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise DataNotLoadedError("Data not split. Please split data first.")

        if order is None:
            order = self.config.sarima.order

        if seasonal_order is None:
            seasonal_order = self.config.sarima.seasonal_order

        # Prepare exogenous variables
        train_exog: Optional[pd.DataFrame] = None
        if exog_columns:
            missing_cols: List[str] = [
                col for col in exog_columns if col not in self.train_data.columns
            ]
            if missing_cols:
                logger.error("Exogenous columns not found in data: %s", missing_cols)
                raise ModelTrainingError(f"Exogenous columns not found in data: {missing_cols}")
            train_exog = self.train_data[exog_columns]

        logger.info(
            "Training SARIMA model with order=%s, seasonal_order=%s",
            order,
            seasonal_order,
        )

        try:
            self.model = SARIMAX(
                self.train_data[self.target_column],
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=self.config.sarima.enforce_stationarity,
                enforce_invertibility=self.config.sarima.enforce_invertibility,
            )

            self.model_fit = self.model.fit(disp=False)
        except Exception as exc:
            logger.error("SARIMA model training failed: %s", exc)
            raise ModelTrainingError("SARIMA model training failed") from exc

        logger.info("SARIMA model training completed successfully")
        logger.debug("Model summary:\n%s", self.model_fit.summary())

        self.model_meta = {
            "type": "sarima",
            "order": order,
            "seasonal_order": seasonal_order,
            "exog_columns": exog_columns,
        }

    def train_prophet_model(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        exog_columns: Optional[List[str]] = None,
    ) -> None:
        """Train a Facebook Prophet model.

        Args:
            yearly_seasonality: Include yearly seasonality.
            weekly_seasonality: Include weekly seasonality.
            daily_seasonality: Include daily seasonality.
            exog_columns: Additional regressor column names.

        Raises:
            DataNotLoadedError: If training data is unavailable.
            ModelTrainingError: If a specified regressor column is missing.
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise DataNotLoadedError("Data not split. Please split data first.")

        logger.info(
            "Training Prophet model with yearly_seasonality=%s, "
            "weekly_seasonality=%s, daily_seasonality=%s",
            yearly_seasonality,
            weekly_seasonality,
            daily_seasonality,
        )

        # Prophet requires columns named 'ds' and 'y'
        prophet_data: pd.DataFrame = self.train_data.copy().reset_index()
        prophet_data.rename(
            columns={prophet_data.columns[0]: "ds", self.target_column: "y"},
            inplace=True,
        )

        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
        )

        if exog_columns:
            for col in exog_columns:
                if col not in self.train_data.columns:
                    logger.error("Regressor column not found in data: %s", col)
                    raise ModelTrainingError(f"Regressor column not found in data: {col}")
                self.model.add_regressor(col)
                logger.info("Added regressor: %s", col)

        try:
            self.model_fit = self.model.fit(prophet_data)
        except Exception as exc:
            logger.error("Prophet model training failed: %s", exc)
            raise ModelTrainingError("Prophet model training failed") from exc

        logger.info("Prophet model training completed successfully")

        self.model_meta = {
            "type": "prophet",
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "exog_columns": exog_columns,
        }

    def train_auto_arima_model(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        m: int = 7,
        exog_columns: Optional[List[str]] = None,
    ) -> None:
        """Train an Auto ARIMA model using ``pmdarima``.

        Args:
            max_p: Maximum ``p`` order.
            max_d: Maximum ``d`` order.
            max_q: Maximum ``q`` order.
            max_P: Maximum seasonal ``P`` order.
            max_D: Maximum seasonal ``D`` order.
            max_Q: Maximum seasonal ``Q`` order.
            m: Seasonal periodicity.
            exog_columns: Exogenous variable column names.

        Raises:
            DataNotLoadedError: If training data is unavailable.
            ModelTrainingError: If exogenous columns are missing or fitting
                fails.
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise DataNotLoadedError("Data not split. Please split data first.")

        logger.info(
            "Training Auto ARIMA model with max_p=%d, max_d=%d, max_q=%d, "
            "max_P=%d, max_D=%d, max_Q=%d, m=%d",
            max_p,
            max_d,
            max_q,
            max_P,
            max_D,
            max_Q,
            m,
        )

        # Prepare exogenous variables
        train_exog: Optional[np.ndarray] = None
        if exog_columns:
            missing_cols: List[str] = [
                col for col in exog_columns if col not in self.train_data.columns
            ]
            if missing_cols:
                logger.error("Exogenous columns not found in data: %s", missing_cols)
                raise ModelTrainingError(f"Exogenous columns not found in data: {missing_cols}")
            train_exog = self.train_data[exog_columns].values

        try:
            self.model = pm.auto_arima(
                self.train_data[self.target_column],
                exogenous=train_exog,
                start_p=1,
                start_q=1,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                start_P=0,
                start_Q=0,
                max_P=max_P,
                max_D=max_D,
                max_Q=max_Q,
                m=m,
                seasonal=True,
                d=None,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
        except Exception as exc:
            logger.error("Auto ARIMA model training failed: %s", exc)
            raise ModelTrainingError("Auto ARIMA model training failed") from exc

        self.model_fit = self.model

        logger.info("Auto ARIMA model training completed successfully")
        logger.info("Best model: %s", self.model)

        best_order: SARIMAOrder = self.model.order
        best_seasonal_order: SARIMASeasonalOrder = self.model.seasonal_order

        self.model_meta = {
            "type": "auto_arima",
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "exog_columns": exog_columns,
        }

    def train_exponential_smoothing_model(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ) -> None:
        """Train a Holt-Winters Exponential Smoothing model.

        Args:
            trend: Trend component type (``'add'``, ``'mul'``, or *None*).
            seasonal: Seasonal component type (``'add'``, ``'mul'``, or
                *None*).
            seasonal_periods: Number of periods in a complete season.

        Raises:
            DataNotLoadedError: If training data is unavailable.
            ModelTrainingError: If the model fails to fit.
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise DataNotLoadedError("Data not split. Please split data first.")

        logger.info(
            "Training Exponential Smoothing model with trend=%s, "
            "seasonal=%s, seasonal_periods=%s",
            trend,
            seasonal,
            seasonal_periods,
        )

        if seasonal is not None and seasonal_periods is None:
            seasonal_periods = 7
            logger.info("No seasonal_periods specified, defaulting to %d", seasonal_periods)

        try:
            self.model = ExponentialSmoothing(
                self.train_data[self.target_column],
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )

            self.model_fit = self.model.fit()
        except Exception as exc:
            logger.error("Exponential Smoothing model training failed: %s", exc)
            raise ModelTrainingError("Exponential Smoothing model training failed") from exc

        logger.info("Exponential Smoothing model training completed successfully")

        self.model_meta = {
            "type": "exponential_smoothing",
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
        }

    # ------------------------------------------------------------------
    # Unified training dispatcher
    # ------------------------------------------------------------------

    def train_model(self, model_type: Optional[str] = None, **kwargs: Any) -> None:
        """Train a model selected by *model_type*.

        Args:
            model_type: One of ``'sarima'``, ``'prophet'``,
                ``'auto_arima'``, ``'exp_smoothing'``.  Falls back to
                :attr:`model_type` when *None*.
            **kwargs: Forwarded to the model-specific training method.

        Raises:
            ModelTrainingError: If the model type is unsupported.
        """
        if model_type is None:
            model_type = self.model_type

        logger.info("Training model of type: %s", model_type)

        dispatch: Dict[str, Any] = {
            "sarima": self.train_sarima_model,
            "prophet": self.train_prophet_model,
            "auto_arima": self.train_auto_arima_model,
            "exp_smoothing": self.train_exponential_smoothing_model,
        }

        trainer_fn = dispatch.get(model_type)
        if trainer_fn is None:
            logger.error("Unsupported model type: %s", model_type)
            raise ModelTrainingError(f"Unsupported model type: {model_type}")

        trainer_fn(**kwargs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, exog_columns: Optional[List[str]] = None) -> MetricsDict:
        """Evaluate the trained model on the held-out test set.

        Args:
            exog_columns: Exogenous variable column names required by the
                model during prediction.

        Returns:
            Dictionary of evaluation metrics (MAE, MSE, RMSE, R-squared,
            MAPE).

        Raises:
            ModelNotTrainedError: If no model has been trained.
            ModelTrainingError: If the model type is unsupported or
                exogenous columns are missing.
        """
        if self.model_fit is None:
            logger.error("Model not trained. Please train a model first.")
            raise ModelNotTrainedError("Model not trained. Please train a model first.")

        model_type_str: str = self.model_meta["type"]
        logger.info("Evaluating %s model", model_type_str)

        predictions: pd.Series

        if model_type_str == "sarima":
            test_exog: Optional[pd.DataFrame] = None
            if exog_columns:
                missing_cols = [
                    col
                    for col in exog_columns
                    if col not in self.test_data.columns  # type: ignore[union-attr]
                ]
                if missing_cols:
                    logger.error("Exogenous columns not found in data: %s", missing_cols)
                    raise ModelTrainingError(
                        f"Exogenous columns not found in data: {missing_cols}"
                    )
                test_exog = self.test_data[exog_columns]  # type: ignore[index]

            logger.info("Generating predictions for test data")
            forecast_result = self.model_fit.get_forecast(
                steps=len(self.test_data),  # type: ignore[arg-type]
                exog=test_exog,
            )
            predictions = forecast_result.predicted_mean
            predictions.index = self.test_data.index  # type: ignore[union-attr]

        elif model_type_str == "prophet":
            future: pd.DataFrame = self.model_fit.make_future_dataframe(
                periods=len(self.test_data),  # type: ignore[arg-type]
                freq=pd.infer_freq(self.data.index),  # type: ignore[union-attr]
            )

            if exog_columns:
                test_data_reset: pd.DataFrame = (
                    self.test_data.copy().reset_index()  # type: ignore[union-attr]
                )
                for col in exog_columns:
                    future.loc[
                        len(future) - len(self.test_data) :, col  # type: ignore[arg-type]
                    ] = test_data_reset[col].values

            forecast: pd.DataFrame = self.model_fit.predict(future)
            predictions = forecast.iloc[-len(self.test_data) :]["yhat"]  # type: ignore[arg-type]
            predictions.index = self.test_data.index  # type: ignore[union-attr]

        elif model_type_str == "auto_arima":
            test_exog_arr: Optional[np.ndarray] = None
            if exog_columns:
                missing_cols = [
                    col
                    for col in exog_columns
                    if col not in self.test_data.columns  # type: ignore[union-attr]
                ]
                if missing_cols:
                    logger.error("Exogenous columns not found in data: %s", missing_cols)
                    raise ModelTrainingError(
                        f"Exogenous columns not found in data: {missing_cols}"
                    )
                test_exog_arr = self.test_data[exog_columns].values  # type: ignore[index]

            pred_values, _conf_int = self.model_fit.predict(
                n_periods=len(self.test_data),  # type: ignore[arg-type]
                exogenous=test_exog_arr,
                return_conf_int=True,
            )
            predictions = pd.Series(
                pred_values,
                index=self.test_data.index,  # type: ignore[union-attr]
            )

        elif model_type_str == "exponential_smoothing":
            predictions = self.model_fit.forecast(len(self.test_data))  # type: ignore[arg-type]
            predictions.index = self.test_data.index  # type: ignore[union-attr]

        else:
            logger.error("Unsupported model type for evaluation: %s", model_type_str)
            raise ModelTrainingError(f"Unsupported model type for evaluation: {model_type_str}")

        # ---- Compute metrics -------------------------------------------------
        actuals: pd.Series = self.test_data[self.target_column]  # type: ignore[index]

        mae: float = float(mean_absolute_error(actuals, predictions))
        mse: float = float(mean_squared_error(actuals, predictions))
        rmse: float = float(np.sqrt(mse))
        r2: float = float(r2_score(actuals, predictions))
        mape: float = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

        metrics: MetricsDict = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }

        logger.info("Evaluation metrics: %s", metrics)

        self.predictions = predictions

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Persist the trained model to disk.

        Args:
            file_path: Destination path (typically ``*.pkl``).

        Raises:
            ModelNotTrainedError: If no model has been trained.
            ModelTrainingError: If serialisation fails.
        """
        if self.model_fit is None:
            logger.error("Model not trained. Please train a model first.")
            raise ModelNotTrainedError("Model not trained. Please train a model first.")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model_package: Dict[str, Any] = {
                "model": self.model_fit,
                "metadata": self.model_meta,
                "target_column": self.target_column,
                "training_end_date": (
                    self.train_data.index[-1] if self.train_data is not None else None
                ),
            }

            with open(file_path, "wb") as fh:
                pickle.dump(model_package, fh)

            logger.info("Model saved to %s", file_path)
        except Exception as exc:
            logger.error("Error saving model to %s: %s", file_path, exc)
            raise ModelTrainingError(f"Failed to save model to {file_path}") from exc

    def load_model(self, file_path: Union[str, Path]) -> None:
        """Load a previously saved model from disk.

        Args:
            file_path: Path to the serialised model file.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ModelTrainingError: If deserialisation fails.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("Model file not found: %s", file_path)
            raise FileNotFoundError(f"Model file not found: {file_path}")

        try:
            with open(file_path, "rb") as fh:
                model_package: Any = pickle.load(fh)  # nosec B301

            if isinstance(model_package, dict) and "model" in model_package:
                self.model_fit = model_package["model"]
                self.model_meta = model_package.get("metadata", {"type": "unknown"})
                self.target_column = model_package.get("target_column", self.target_column)
            else:
                # Legacy format: bare model object
                self.model_fit = model_package
                self.model_meta = {"type": "unknown"}

            logger.info("Model loaded from %s", file_path)
            logger.info("Model type: %s", self.model_meta.get("type", "unknown"))
        except Exception as exc:
            logger.error("Error loading model from %s: %s", file_path, exc)
            raise ModelTrainingError(f"Failed to load model from {file_path}") from exc

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_results(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot actual versus predicted values.

        Args:
            save_path: When provided the figure is saved to this path rather
                than displayed interactively.

        Raises:
            ModelNotTrainedError: If the model has not been evaluated yet.
        """
        if self.model_fit is None or self.predictions is None:
            logger.error("Model not evaluated. Please evaluate the model first.")
            raise ModelNotTrainedError("Model not evaluated. Please evaluate the model first.")

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))

        # Plot training data
        plt.plot(
            self.train_data.index,  # type: ignore[union-attr]
            self.train_data[self.target_column],  # type: ignore[index]
            label="Train Data",
            color="blue",
        )

        # Plot test data
        plt.plot(
            self.test_data.index,  # type: ignore[union-attr]
            self.test_data[self.target_column],  # type: ignore[index]
            label="Test Data",
            color="green",
        )

        # Plot predictions
        plt.plot(
            self.predictions.index,
            self.predictions,
            label="Predictions",
            color="red",
            linestyle="--",
        )

        model_type_label: str = self.model_meta.get("type", "Unknown")

        plt.title(f"{model_type_label.upper()} Model: Actual vs Predicted Values")
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info("Plot saved to %s", save_path)
        else:
            plt.show()


# =====================================================================
# Convenience function
# =====================================================================


def train_and_evaluate_model(
    config: AppConfig,
    data: Optional[pd.DataFrame] = None,
    file_path: Optional[Union[str, Path]] = None,
    target_column: str = "close",
    exog_columns: Optional[List[str]] = None,
    save_model: bool = True,
    save_plot: bool = True,
    model_type: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> MetricsDict:
    """Train and evaluate a time series model end-to-end.

    This is a high-level convenience wrapper around :class:`ModelTrainer`
    that handles data loading, splitting, training, evaluation, and
    optional persistence in a single call.

    Args:
        config: Application configuration.
        data: Pre-loaded time series ``DataFrame``.  When *None* the data
            is loaded from *file_path* or the default preprocessed
            directory.
        file_path: Path to a CSV data file.  Used when *data* is *None*.
        target_column: Column name to predict.
        exog_columns: Exogenous variable column names.
        save_model: Persist the trained model to disk.
        save_plot: Save the evaluation plot to disk.
        model_type: Override the model type specified in *config*.
        model_params: Additional keyword arguments forwarded to the
            model-specific training method.

    Returns:
        Dictionary of evaluation metrics.

    Raises:
        ModelTrainingError: If any stage of the pipeline fails.
    """
    if model_type is None:
        model_type = config.model.model_type.lower()

    if model_params is None:
        model_params = {}

    trainer = ModelTrainer(config, data, target_column)

    # Load data when not provided directly
    if data is None and file_path:
        trainer.load_data(file_path)
    elif data is None:
        preprocessed_dir = Path(config.output_dir) / "preprocessed"
        symbol: str = "MSFT"
        default_path: Path = preprocessed_dir / f"{symbol}_preprocessed.csv"
        trainer.load_data(default_path)

    trainer.split_data()

    logger.info("Training %s model with params: %s", model_type, model_params)

    try:
        dispatch: Dict[str, Any] = {
            "sarima": lambda: trainer.train_sarima_model(
                exog_columns=exog_columns, **model_params
            ),
            "prophet": lambda: trainer.train_prophet_model(
                exog_columns=exog_columns, **model_params
            ),
            "auto_arima": lambda: trainer.train_auto_arima_model(
                exog_columns=exog_columns, **model_params
            ),
            "exp_smoothing": lambda: trainer.train_exponential_smoothing_model(**model_params),
        }

        trainer_fn = dispatch.get(model_type)
        if trainer_fn is None:
            logger.error("Unsupported model type: %s", model_type)
            raise ModelTrainingError(f"Unsupported model type: {model_type}")

        trainer_fn()

        # Evaluate
        metrics: MetricsDict = trainer.evaluate_model(exog_columns=exog_columns)

        # Save model
        if save_model:
            model_dir: Path = Path(config.output_dir) / "models"
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path: Path = model_dir / f"{model_type}_{timestamp}.pkl"
            trainer.save_model(model_path)

        # Plot results
        if save_plot:
            plots_dir: Path = Path(config.output_dir) / "plots"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path: Path = plots_dir / f"{model_type}_results_{timestamp}.png"
            trainer.plot_results(plot_path)
        else:
            trainer.plot_results()

        return metrics

    except ModelTrainingError:
        raise
    except Exception as exc:
        logger.error("Error training and evaluating model: %s", exc)
        raise ModelTrainingError(
            f"Training and evaluation pipeline failed for {model_type}"
        ) from exc
