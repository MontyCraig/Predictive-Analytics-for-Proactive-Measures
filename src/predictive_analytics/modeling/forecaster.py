"""Forecasting and prediction module for time series data.

Provides the ``TimeSeriesForecaster`` class which loads a trained model and
generates multi-step-ahead forecasts with prediction intervals for SARIMA,
Prophet, Auto ARIMA, and Exponential Smoothing model families.

A convenience function :func:`generate_forecast` wraps the full
forecast-analyse-save workflow for common use-cases.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from predictive_analytics.config.logging import setup_logging
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import ForecastingError, ModelNotTrainedError

__all__: list[str] = ["TimeSeriesForecaster", "generate_forecast"]

logger: logging.Logger = setup_logging()


class TimeSeriesForecaster:
    """Generate forecasts from pre-trained time series models.

    Supports SARIMA, Facebook Prophet, Auto ARIMA (``pmdarima``), and
    Holt-Winters Exponential Smoothing.  The typical lifecycle is::

        forecaster = TimeSeriesForecaster(config, model_path="model.pkl")
        forecaster.load_data("data.csv")
        forecast_df = forecaster.forecast(steps=30)
        forecaster.plot_forecast(forecast_df)
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: AppConfig,
        model_path: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ) -> None:
        """Initialise the forecaster.

        Args:
            config: Application configuration.
            model_path: Path to a serialised model file.  When provided the
                model is loaded immediately.
            data: Historical time series data.
            target_column: Name of the target column.
        """
        self.config: AppConfig = config
        self.data: Optional[pd.DataFrame] = data
        self.target_column: str = target_column
        self.model_fit: Any = None
        self.model_meta: Optional[Dict[str, Any]] = None
        self.training_end_date: Optional[datetime] = None

        if model_path is not None:
            self.load_model(model_path)

        logger.info("TimeSeriesForecaster initialized")

    # ------------------------------------------------------------------
    # Data / model loading
    # ------------------------------------------------------------------

    def load_model(self, file_path: Union[str, Path]) -> None:
        """Load a trained model from a pickle file.

        Args:
            file_path: Path to the serialised model.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ForecastingError: If deserialisation fails.
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
                self.training_end_date = model_package.get("training_end_date")
            else:
                # Legacy format: bare model object
                self.model_fit = model_package
                self.model_meta = {"type": "sarima"}

            logger.info("Model loaded from %s", file_path)
            logger.info("Model type: %s", self.model_meta.get("type", "unknown"))
        except Exception as exc:
            logger.error("Error loading model from %s: %s", file_path, exc)
            raise ForecastingError(f"Failed to load model from {file_path}") from exc

    def load_data(self, file_path: Union[str, Path]) -> None:
        """Load historical data from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ForecastingError: If the file cannot be parsed.
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
            raise ForecastingError(f"Failed to load data from {file_path}") from exc

    # ------------------------------------------------------------------
    # Main forecast entry-point
    # ------------------------------------------------------------------

    def forecast(
        self,
        steps: int = 30,
        exog_features: Optional[pd.DataFrame] = None,
        confidence_interval: float = 0.95,
    ) -> pd.DataFrame:
        """Generate multi-step-ahead forecasts.

        Args:
            steps: Number of future periods to forecast.
            exog_features: Exogenous features for the forecast horizon.
            confidence_interval: Width of the prediction interval
                (e.g. ``0.95`` for 95 %).

        Returns:
            ``DataFrame`` with columns ``forecast``, ``lower_bound``, and
            ``upper_bound``.

        Raises:
            ModelNotTrainedError: If no model has been loaded.
            ForecastingError: If the model type is unsupported or
                forecasting fails.
        """
        if self.model_fit is None:
            logger.error("Model not loaded. Please load a model first.")
            raise ModelNotTrainedError("Model not loaded. Please load a model first.")

        model_type: str = (self.model_meta or {}).get("type", "unknown").lower()
        logger.info(
            "Generating forecast for %d steps ahead using %s model",
            steps,
            model_type,
        )

        # Determine the last known date
        last_date: datetime
        if self.data is not None:
            last_date = self.data.index[-1]
        elif self.training_end_date is not None:
            last_date = self.training_end_date
        else:
            last_date = datetime.now() - timedelta(days=1)
            logger.warning(
                "No data available, using yesterday as the last date: %s",
                last_date,
            )

        # Infer frequency
        freq: str = "D"
        if self.data is not None and len(self.data) > 1:
            inferred: Optional[str] = pd.infer_freq(self.data.index)
            if inferred is not None:
                freq = inferred
            else:
                freq = self._infer_frequency_fallback(self.data.index)

        forecast_index: pd.DatetimeIndex = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq,
        )

        try:
            dispatch: Dict[str, Any] = {
                "sarima": lambda: self._forecast_sarima(
                    steps, exog_features, confidence_interval, forecast_index
                ),
                "prophet": lambda: self._forecast_prophet(
                    steps, exog_features, confidence_interval, forecast_index
                ),
                "auto_arima": lambda: self._forecast_auto_arima(
                    steps, exog_features, confidence_interval, forecast_index
                ),
                "exp_smoothing": lambda: self._forecast_exp_smoothing(
                    steps, confidence_interval, forecast_index
                ),
            }

            forecast_fn = dispatch.get(model_type)
            if forecast_fn is None:
                logger.error("Unsupported model type for forecasting: %s", model_type)
                raise ForecastingError(f"Unsupported model type for forecasting: {model_type}")

            return forecast_fn()

        except ForecastingError:
            raise
        except Exception as exc:
            logger.error("Error generating forecast: %s", exc)
            raise ForecastingError("Forecast generation failed") from exc

    # ------------------------------------------------------------------
    # Model-specific forecasting
    # ------------------------------------------------------------------

    def _forecast_sarima(
        self,
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate SARIMA forecasts.

        Args:
            steps: Number of forecast steps.
            exog_features: Exogenous features for the forecast horizon.
            confidence_interval: Prediction interval width.
            forecast_index: DatetimeIndex for the forecast period.

        Returns:
            DataFrame with forecast, lower_bound, upper_bound.
        """
        forecast_result = self.model_fit.get_forecast(steps=steps, exog=exog_features)
        forecast_mean: pd.Series = forecast_result.predicted_mean

        alpha: float = 1 - confidence_interval
        forecast_ci: pd.DataFrame = forecast_result.conf_int(alpha=alpha)

        forecast_df = pd.DataFrame(
            {
                "forecast": forecast_mean,
                "lower_bound": forecast_ci.iloc[:, 0],
                "upper_bound": forecast_ci.iloc[:, 1],
            }
        )
        forecast_df.index = forecast_index

        logger.info("SARIMA forecast generated successfully for %d steps", steps)
        return forecast_df

    def _forecast_prophet(
        self,
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate Prophet forecasts.

        Args:
            steps: Number of forecast steps.
            exog_features: Exogenous features for the forecast horizon.
            confidence_interval: Prediction interval width.
            forecast_index: DatetimeIndex for the forecast period.

        Returns:
            DataFrame with forecast, lower_bound, upper_bound.
        """
        future = pd.DataFrame({"ds": forecast_index})

        meta: Dict[str, Any] = self.model_meta or {}
        exog_cols: Optional[List[str]] = meta.get("exog_columns")
        if exog_features is not None and exog_cols is not None:
            for col in exog_cols:
                if col in exog_features:
                    future[col] = exog_features[col].values

        forecast: pd.DataFrame = self.model_fit.predict(future)

        forecast_df = pd.DataFrame(
            {
                "forecast": forecast["yhat"],
                "lower_bound": forecast["yhat_lower"],
                "upper_bound": forecast["yhat_upper"],
            }
        )
        forecast_df.index = forecast_index

        logger.info("Prophet forecast generated successfully for %d steps", steps)
        return forecast_df

    def _forecast_auto_arima(
        self,
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate Auto ARIMA forecasts.

        Args:
            steps: Number of forecast steps.
            exog_features: Exogenous features for the forecast horizon.
            confidence_interval: Prediction interval width.
            forecast_index: DatetimeIndex for the forecast period.

        Returns:
            DataFrame with forecast, lower_bound, upper_bound.
        """
        exog_array: Optional[np.ndarray] = None
        meta: Dict[str, Any] = self.model_meta or {}
        exog_cols: Optional[List[str]] = meta.get("exog_columns")

        if exog_features is not None and exog_cols is not None:
            if all(col in exog_features for col in exog_cols):
                exog_array = exog_features[exog_cols].values
            else:
                missing: List[str] = [col for col in exog_cols if col not in exog_features]
                logger.warning("Missing exogenous columns for forecast: %s", missing)

        pred: np.ndarray
        pred_ci: np.ndarray
        pred, pred_ci = self.model_fit.predict(
            n_periods=steps,
            exogenous=exog_array,
            return_conf_int=True,
            alpha=(1 - confidence_interval),
        )

        forecast_df = pd.DataFrame(
            {
                "forecast": pred,
                "lower_bound": pred_ci[:, 0],
                "upper_bound": pred_ci[:, 1],
            }
        )
        forecast_df.index = forecast_index

        logger.info("Auto ARIMA forecast generated successfully for %d steps", steps)
        return forecast_df

    def _forecast_exp_smoothing(
        self,
        steps: int,
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate Exponential Smoothing forecasts.

        Because ``statsmodels`` Exponential Smoothing does not natively
        provide prediction intervals, an approximation based on residual
        standard error is used.

        Args:
            steps: Number of forecast steps.
            confidence_interval: Prediction interval width.
            forecast_index: DatetimeIndex for the forecast period.

        Returns:
            DataFrame with forecast, lower_bound, upper_bound.
        """
        forecast: pd.Series = self.model_fit.forecast(steps)

        if hasattr(self.model_fit, "fittedvalues"):
            resid: pd.Series = self.model_fit.resid
            sigma: float = float(np.sqrt(np.sum(resid**2) / (len(resid) - 1)))

            z: float = float(stats.norm.ppf(0.5 + confidence_interval / 2))

            widths: List[float] = [sigma * z * np.sqrt(h + 1) for h in range(steps)]

            lower_bound: Union[pd.Series, np.ndarray] = forecast - widths
            upper_bound: Union[pd.Series, np.ndarray] = forecast + widths
        else:
            margin: float = 0.1
            lower_bound = forecast * (1 - margin)
            upper_bound = forecast * (1 + margin)

        forecast_df = pd.DataFrame(
            {
                "forecast": forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )
        forecast_df.index = forecast_index

        logger.info(
            "Exponential Smoothing forecast generated successfully for %d steps",
            steps,
        )
        return forecast_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        historical_periods: int = 60,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot forecasts with historical context and prediction intervals.

        Args:
            forecast_df: Forecast ``DataFrame`` returned by :meth:`forecast`.
            historical_periods: Number of trailing historical periods to
                include in the plot.
            save_path: When provided the figure is saved rather than shown.
        """
        import seaborn as sns

        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 7))

        model_type: str = (self.model_meta or {}).get("type", "Unknown").upper()

        if self.data is None:
            logger.warning("Historical data not loaded. Plotting only forecast.")
        else:
            historical_data: pd.DataFrame = self.data.tail(historical_periods)
            plt.plot(
                historical_data.index,
                historical_data[self.target_column],
                label="Historical Data",
                color="blue",
            )

        plt.plot(
            forecast_df.index,
            forecast_df["forecast"],
            label="Forecast",
            color="red",
            linestyle="--",
        )

        plt.fill_between(
            forecast_df.index,
            forecast_df["lower_bound"],
            forecast_df["upper_bound"],
            color="pink",
            alpha=0.3,
            label="95% Confidence Interval",
        )

        plt.title(f"{model_type} Model: Time Series Forecast ({self.target_column})")
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info("Forecast plot saved to %s", save_path)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Anomaly detection & analysis
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        forecast_df: pd.DataFrame,
        actual_values: Optional[pd.Series] = None,
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Detect anomalies in forecasted or actual values.

        When *actual_values* are supplied the method compares them against
        the forecast using z-scores on the residuals.  Otherwise it flags
        periods with unusually wide prediction intervals.

        Args:
            forecast_df: Forecast ``DataFrame``.
            actual_values: Observed values for the forecast horizon.
            threshold: Z-score threshold for flagging.

        Returns:
            Enhanced ``DataFrame`` with anomaly / uncertainty flags.
        """
        logger.info("Detecting anomalies with threshold %s", threshold)

        result_df: pd.DataFrame = forecast_df.copy()

        if actual_values is not None:
            result_df["actual"] = actual_values
            result_df["error"] = result_df["actual"] - result_df["forecast"]

            result_df["z_score"] = np.abs(stats.zscore(result_df["error"]))
            result_df["is_anomaly"] = result_df["z_score"] > threshold

            n_anomalies: int = int(result_df["is_anomaly"].sum())
            logger.info("Detected %d anomalies in actual values", n_anomalies)
        else:
            result_df["forecast_range"] = result_df["upper_bound"] - result_df["lower_bound"]

            mean_range: float = float(result_df["forecast_range"].mean())
            std_range: float = float(result_df["forecast_range"].std())

            result_df["range_z_score"] = np.abs(
                (result_df["forecast_range"] - mean_range) / std_range
            )
            result_df["is_uncertain"] = result_df["range_z_score"] > threshold

            n_uncertain: int = int(result_df["is_uncertain"].sum())
            logger.info("Detected %d periods with high uncertainty", n_uncertain)

        return result_df

    def analyze_forecast(self, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """Compute summary statistics for a forecast.

        Args:
            forecast_df: Forecast ``DataFrame``.

        Returns:
            Dictionary of analysis metrics including trend, growth rate,
            volatility, and uncertainty.
        """
        logger.info("Analyzing forecast for key indicators")

        daily_changes: pd.Series = forecast_df["forecast"].diff().dropna()
        avg_daily_change: float = float(daily_changes.mean())

        volatility: float = float(forecast_df["forecast"].std())

        uncertainty: float = float(
            (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()
        )

        min_forecast: float = float(forecast_df["forecast"].min())
        max_forecast: float = float(forecast_df["forecast"].max())

        start_value: float = float(forecast_df["forecast"].iloc[0])
        end_value: float = float(forecast_df["forecast"].iloc[-1])
        growth_rate: float = ((end_value / start_value) - 1) * 100

        metrics: Dict[str, float] = {
            "avg_daily_change": avg_daily_change,
            "growth_rate_pct": growth_rate,
            "volatility": volatility,
            "uncertainty": uncertainty,
            "min_forecast": min_forecast,
            "max_forecast": max_forecast,
        }

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_frequency_fallback(time_index: pd.DatetimeIndex) -> str:
        """Best-effort frequency inference when ``pd.infer_freq`` fails.

        Args:
            time_index: The datetime index to analyse.

        Returns:
            A pandas-compatible frequency string, or ``'D'`` as a
            last-resort default.
        """
        if len(time_index) < 3:
            return "D"

        diff: pd.Series = pd.Series(time_index[1:]) - pd.Series(time_index[:-1])
        most_common_diff: pd.Timedelta = diff.mode()[0]
        days: int = most_common_diff.days

        _freq_map: Dict[int, str] = {1: "D", 7: "W"}
        if days in _freq_map:
            return _freq_map[days]
        if 28 <= days <= 31:
            return "M"
        if 90 <= days <= 92:
            return "Q"
        if 365 <= days <= 366:
            return "Y"

        hours: int = most_common_diff.seconds // 3600
        if hours == 1:
            return "H"

        return "D"


# =====================================================================
# Convenience function
# =====================================================================


def generate_forecast(
    config: AppConfig,
    model_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    steps: int = 30,
    target_column: str = "close",
    plot: bool = True,
    save_forecast: bool = True,
    save_plot: bool = True,
    model_type: Optional[str] = None,
    exog_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate, analyse, and optionally persist a forecast.

    This is a high-level convenience wrapper around
    :class:`TimeSeriesForecaster` that handles model discovery, data
    loading, forecasting, analysis, and optional visualisation / saving
    in a single call.

    Args:
        config: Application configuration.
        model_path: Path to a serialised model.  When *None* the latest
            model matching *model_type* is selected from the output
            directory.
        data_path: Path to historical data CSV.
        steps: Number of future periods to forecast.
        target_column: Column name to predict.
        plot: Whether to generate a plot.
        save_forecast: Whether to persist the forecast as CSV.
        save_plot: Whether to save the plot to disk (only when *plot*
            is also *True*).
        model_type: Override model type from configuration.
        exog_features: Exogenous features for the forecast horizon.

    Returns:
        ``DataFrame`` containing the forecast.

    Raises:
        FileNotFoundError: If no model can be located.
        ForecastingError: If forecasting fails.
    """
    if model_type is None:
        model_type = config.model.model_type.lower()

    # Discover latest model on disk when no explicit path given
    if model_path is None:
        model_dir: Path = Path(config.output_dir) / "models"
        if not model_dir.exists():
            logger.error("Model directory not found: %s", model_dir)
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_files: List[Path] = list(model_dir.glob(f"{model_type}_*.pkl"))
        if not model_files:
            logger.error("No %s model files found in %s", model_type, model_dir)
            raise FileNotFoundError(f"No {model_type} model files found in {model_dir}")

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info("Using latest model: %s", model_path)

    forecaster = TimeSeriesForecaster(config, model_path, target_column=target_column)

    # Load historical data
    if data_path:
        forecaster.load_data(data_path)
    elif data_path is None and not save_forecast:
        preprocessed_dir: Path = Path(config.output_dir) / "preprocessed"
        data_files: List[Path] = list(preprocessed_dir.glob("*_preprocessed.csv"))
        if data_files:
            latest_data: Path = max(data_files, key=lambda p: p.stat().st_mtime)
            forecaster.load_data(latest_data)

    forecast_df: pd.DataFrame = forecaster.forecast(steps=steps, exog_features=exog_features)

    # Analyse
    metrics: Dict[str, float] = forecaster.analyze_forecast(forecast_df)

    logger.info("Forecast analysis:")
    for metric_name, value in metrics.items():
        logger.info("  %s: %.4f", metric_name, value)

    # Plot
    if plot:
        if save_plot:
            plots_dir: Path = Path(config.output_dir) / "forecasts" / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path: Path = plots_dir / f"forecast_{model_type}_{steps}_steps_{timestamp}.png"
            forecaster.plot_forecast(forecast_df, save_path=plot_path)
        else:
            forecaster.plot_forecast(forecast_df)

    # Save forecast CSV
    if save_forecast:
        forecasts_dir: Path = Path(config.output_dir) / "forecasts" / "data"
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_path: Path = (
            forecasts_dir / f"forecast_{model_type}_{steps}_steps_{timestamp}.csv"
        )
        forecast_df.to_csv(forecast_path)
        logger.info("Forecast saved to %s", forecast_path)

    return forecast_df
