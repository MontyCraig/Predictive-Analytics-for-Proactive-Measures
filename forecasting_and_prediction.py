"""Forecasting and prediction module for time series data."""
import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging
from utils.helpers import ensure_directory, infer_frequency

# Initialize logger
logger = setup_logging()


class TimeSeriesForecaster:
    """Forecaster for time series data using trained models."""

    def __init__(
        self,
        config: Config,
        model_path: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ):
        """Initialize the forecaster.
        
        Args:
            config: Application configuration
            model_path: Path to trained model file
            data: Time series data (if None, it will be loaded)
            target_column: Target column to predict
        """
        self.config = config
        self.data = data
        self.target_column = target_column
        self.model_fit = None
        self.model_meta = None
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
            
        logger.info("TimeSeriesForecaster initialized")

    def load_model(self, file_path: Union[str, Path]) -> None:
        """Load a trained model from a file.
        
        Args:
            file_path: Path to the model file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")
            
        try:
            with open(file_path, "rb") as f:
                model_package = pickle.load(f)
            
            # Handle different package formats (for backward compatibility)
            if isinstance(model_package, dict) and "model" in model_package:
                self.model_fit = model_package["model"]
                self.model_meta = model_package.get("metadata", {"type": "unknown"})
                self.target_column = model_package.get("target_column", self.target_column)
                self.training_end_date = model_package.get("training_end_date")
            else:
                # Legacy format where only the model was saved
                self.model_fit = model_package
                self.model_meta = {"type": "sarima"}  # Assume SARIMA for legacy models
                
            logger.info(f"Model loaded from {file_path}")
            logger.info(f"Model type: {self.model_meta.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {str(e)}")
            raise

    def load_data(self, file_path: Union[str, Path]) -> None:
        """Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(self.data)} records from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def forecast(
        self, 
        steps: int = 30,
        exog_features: Optional[pd.DataFrame] = None,
        confidence_interval: float = 0.95,
    ) -> pd.DataFrame:
        """Generate forecasts for future time periods.
        
        Args:
            steps: Number of steps to forecast
            exog_features: Exogenous features for the forecast period
            confidence_interval: Confidence interval for prediction intervals
            
        Returns:
            DataFrame with forecasts and prediction intervals
        """
        if self.model_fit is None:
            logger.error("Model not loaded. Please load a model first.")
            raise ValueError("Model not loaded. Please load a model first.")
        
        model_type = self.model_meta.get("type", "unknown").lower()
        logger.info(f"Generating forecast for {steps} steps ahead using {model_type} model")
        
        # Get the date range for forecasting
        if self.data is not None:
            last_date = self.data.index[-1]
        elif hasattr(self, 'training_end_date') and self.training_end_date is not None:
            last_date = self.training_end_date
        else:
            last_date = datetime.now() - timedelta(days=1)
            logger.warning(f"No data available, using yesterday as the last date: {last_date}")
        
        # Infer frequency from data or use daily as default
        if self.data is not None and len(self.data) > 1:
            freq = pd.infer_freq(self.data.index)
            if freq is None:
                freq = infer_frequency(self.data.index)
                if freq == 'unknown':
                    freq = 'D'  # Default to daily if cannot infer
        else:
            freq = 'D'  # Default to daily
        
        # Create date index for forecast
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )
        
        # Generate forecast based on model type
        try:
            if model_type == "sarima":
                return self._forecast_sarima(steps, exog_features, confidence_interval, forecast_index)
            elif model_type == "prophet":
                return self._forecast_prophet(steps, exog_features, confidence_interval, forecast_index)
            elif model_type == "auto_arima":
                return self._forecast_auto_arima(steps, exog_features, confidence_interval, forecast_index)
            elif model_type == "exp_smoothing":
                return self._forecast_exp_smoothing(steps, confidence_interval, forecast_index)
            else:
                logger.error(f"Unsupported model type for forecasting: {model_type}")
                raise ValueError(f"Unsupported model type for forecasting: {model_type}")
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def _forecast_sarima(
        self, 
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate forecasts using a SARIMA model.
        
        Args:
            steps: Number of steps to forecast
            exog_features: Exogenous features for the forecast period
            confidence_interval: Confidence interval for prediction intervals
            forecast_index: DatetimeIndex for the forecast period
            
        Returns:
            DataFrame with forecasts and prediction intervals
        """
        # Generate forecast
        forecast_result = self.model_fit.get_forecast(steps=steps, exog=exog_features)
        forecast_mean = forecast_result.predicted_mean
        
        # Get prediction intervals
        alpha = 1 - confidence_interval
        forecast_ci = forecast_result.conf_int(alpha=alpha)
        
        # Create a DataFrame with forecasted values and intervals
        forecast_df = pd.DataFrame({
            "forecast": forecast_mean,
            "lower_bound": forecast_ci.iloc[:, 0],
            "upper_bound": forecast_ci.iloc[:, 1],
        })
        
        # Set forecast index
        forecast_df.index = forecast_index
        
        logger.info(f"SARIMA forecast generated successfully for {steps} steps")
        return forecast_df

    def _forecast_prophet(
        self, 
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate forecasts using a Prophet model.
        
        Args:
            steps: Number of steps to forecast
            exog_features: Exogenous features for the forecast period
            confidence_interval: Confidence interval for prediction intervals
            forecast_index: DatetimeIndex for the forecast period
            
        Returns:
            DataFrame with forecasts and prediction intervals
        """
        # Create future DataFrame for prediction
        future = pd.DataFrame({'ds': forecast_index})
        
        # Add regressors if provided and supported by model
        if exog_features is not None and self.model_meta.get("exog_columns") is not None:
            for col in self.model_meta.get("exog_columns"):
                if col in exog_features:
                    future[col] = exog_features[col].values
        
        # Generate forecast
        forecast = self.model_fit.predict(future)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "forecast": forecast['yhat'],
            "lower_bound": forecast[f'yhat_lower'],
            "upper_bound": forecast[f'yhat_upper'],
        })
        
        # Set index
        forecast_df.index = forecast_index
        
        logger.info(f"Prophet forecast generated successfully for {steps} steps")
        return forecast_df

    def _forecast_auto_arima(
        self, 
        steps: int,
        exog_features: Optional[pd.DataFrame],
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate forecasts using an Auto ARIMA model.
        
        Args:
            steps: Number of steps to forecast
            exog_features: Exogenous features for the forecast period
            confidence_interval: Confidence interval for prediction intervals
            forecast_index: DatetimeIndex for the forecast period
            
        Returns:
            DataFrame with forecasts and prediction intervals
        """
        # Prepare exogenous variables if specified
        exog_array = None
        if exog_features is not None and self.model_meta.get("exog_columns") is not None:
            exog_cols = self.model_meta.get("exog_columns")
            if all(col in exog_features for col in exog_cols):
                exog_array = exog_features[exog_cols].values
            else:
                missing_cols = [col for col in exog_cols if col not in exog_features]
                logger.warning(f"Missing exogenous columns for forecast: {missing_cols}")
        
        # Generate prediction with confidence intervals
        pred, pred_ci = self.model_fit.predict(
            n_periods=steps, 
            exogenous=exog_array,
            return_conf_int=True,
            alpha=(1 - confidence_interval)
        )
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "forecast": pred,
            "lower_bound": pred_ci[:, 0],
            "upper_bound": pred_ci[:, 1],
        })
        
        # Set index
        forecast_df.index = forecast_index
        
        logger.info(f"Auto ARIMA forecast generated successfully for {steps} steps")
        return forecast_df

    def _forecast_exp_smoothing(
        self, 
        steps: int,
        confidence_interval: float,
        forecast_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate forecasts using an Exponential Smoothing model.
        
        Args:
            steps: Number of steps to forecast
            confidence_interval: Confidence interval for prediction intervals
            forecast_index: DatetimeIndex for the forecast period
            
        Returns:
            DataFrame with forecasts and prediction intervals
        """
        # Generate forecast
        forecast = self.model_fit.forecast(steps)
        
        # Create prediction intervals (using model's properties if available)
        # ExponentialSmoothing in statsmodels doesn't provide confidence intervals directly
        # So we'll approximate using forecast standard errors if available
        if hasattr(self.model_fit, 'fittedvalues'):
            # Residual standard error
            resid = self.model_fit.resid
            sigma = np.sqrt(np.sum(resid**2) / (len(resid) - 1))
            
            # Z value for the confidence interval
            from scipy import stats
            z = stats.norm.ppf(0.5 + confidence_interval / 2)
            
            # Confidence interval width increases with the forecast horizon
            # Simple approximation: width grows as sqrt(h)
            widths = [sigma * z * np.sqrt(h + 1) for h in range(steps)]
            
            lower_bound = forecast - widths
            upper_bound = forecast + widths
        else:
            # Fallback: use a percentage of the forecast value
            margin = 0.1  # 10% margin
            lower_bound = forecast * (1 - margin)
            upper_bound = forecast * (1 + margin)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "forecast": forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        })
        
        # Set index
        forecast_df.index = forecast_index
        
        logger.info(f"Exponential Smoothing forecast generated successfully for {steps} steps")
        return forecast_df

    def plot_forecast(
        self, 
        forecast_df: pd.DataFrame,
        historical_periods: int = 60,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot forecasts with historical data and prediction intervals.
        
        Args:
            forecast_df: DataFrame with forecasts from the forecast method
            historical_periods: Number of historical periods to show
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 7))
        
        # Get model type for plot title
        model_type = self.model_meta.get('type', 'Unknown').upper()
        
        # Plot historical data if available
        if self.data is None:
            logger.warning("Historical data not loaded. Plotting only forecast.")
            historical_data = None
        else:
            historical_data = self.data.tail(historical_periods)
            plt.plot(
                historical_data.index,
                historical_data[self.target_column],
                label="Historical Data",
                color="blue",
            )
            
        # Plot forecast
        plt.plot(
            forecast_df.index,
            forecast_df["forecast"],
            label="Forecast",
            color="red",
            linestyle="--",
        )
        
        # Plot prediction intervals
        plt.fill_between(
            forecast_df.index,
            forecast_df["lower_bound"],
            forecast_df["upper_bound"],
            color="pink",
            alpha=0.3,
            label=f"{int(0.95*100)}% Confidence Interval",
        )
        
        # Add labels and legend
        plt.title(f"{model_type} Model: Time Series Forecast ({self.target_column})")
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            save_path = Path(save_path)
            ensure_directory(save_path.parent)
            plt.savefig(save_path)
            logger.info(f"Forecast plot saved to {save_path}")
        else:
            plt.show()

    def detect_anomalies(
        self, 
        forecast_df: pd.DataFrame,
        actual_values: Optional[pd.Series] = None,
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Detect anomalies in the forecast or actual values.
        
        Args:
            forecast_df: DataFrame with forecasts
            actual_values: Actual values for the forecast period (if available)
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomalies flagged
        """
        logger.info(f"Detecting anomalies with threshold {threshold}")
        
        result_df = forecast_df.copy()
        
        if actual_values is not None:
            # Compare actual values with forecasts
            result_df["actual"] = actual_values
            result_df["error"] = result_df["actual"] - result_df["forecast"]
            
            # Z-score of errors
            result_df["z_score"] = np.abs(stats.zscore(result_df["error"]))
            result_df["is_anomaly"] = result_df["z_score"] > threshold
            
            n_anomalies = result_df["is_anomaly"].sum()
            logger.info(f"Detected {n_anomalies} anomalies in actual values")
            
        else:
            # For future forecasts, use prediction intervals
            result_df["forecast_range"] = result_df["upper_bound"] - result_df["lower_bound"]
            
            # Calculate range z-scores
            mean_range = result_df["forecast_range"].mean()
            std_range = result_df["forecast_range"].std()
            
            result_df["range_z_score"] = np.abs((result_df["forecast_range"] - mean_range) / std_range)
            result_df["is_uncertain"] = result_df["range_z_score"] > threshold
            
            n_uncertain = result_df["is_uncertain"].sum()
            logger.info(f"Detected {n_uncertain} periods with high uncertainty")
            
        return result_df

    def analyze_forecast(self, forecast_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze the forecast for key indicators.
        
        Args:
            forecast_df: DataFrame with forecasts
            
        Returns:
            Dictionary with forecast analysis metrics
        """
        logger.info("Analyzing forecast for key indicators")
        
        # Calculate trend (average daily change)
        daily_changes = forecast_df["forecast"].diff().dropna()
        avg_daily_change = daily_changes.mean()
        
        # Calculate volatility (standard deviation of forecast)
        volatility = forecast_df["forecast"].std()
        
        # Calculate uncertainty (average width of prediction interval)
        uncertainty = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()
        
        # Calculate min and max forecasted values
        min_forecast = forecast_df["forecast"].min()
        max_forecast = forecast_df["forecast"].max()
        
        # Calculate growth rate
        start_value = forecast_df["forecast"].iloc[0]
        end_value = forecast_df["forecast"].iloc[-1]
        growth_rate = ((end_value / start_value) - 1) * 100  # Percentage
        
        # Combine metrics
        metrics = {
            "avg_daily_change": avg_daily_change,
            "growth_rate_pct": growth_rate,
            "volatility": volatility,
            "uncertainty": uncertainty,
            "min_forecast": min_forecast,
            "max_forecast": max_forecast,
        }
        
        return metrics


def generate_forecast(
    config: Config,
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
    """Generate and analyze forecasts.
    
    Args:
        config: Application configuration
        model_path: Path to trained model file (if None, latest model will be used)
        data_path: Path to data file (if None, latest data will be used)
        steps: Number of steps to forecast
        target_column: Target column to predict
        plot: Whether to plot the forecast
        save_forecast: Whether to save the forecast to a CSV file
        save_plot: Whether to save the plot
        model_type: Type of model to use (overrides config if specified)
        exog_features: Exogenous features for the forecast period
        
    Returns:
        DataFrame with forecasts and analysis
    """
    # Determine model type to look for
    if model_type is None:
        model_type = config.model.model_type.lower()
    
    # Find latest model if not provided
    if model_path is None:
        model_dir = Path(config.output_dir) / "models"
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        model_files = list(model_dir.glob(f"{model_type}_*.pkl"))
        if not model_files:
            logger.error(f"No {model_type} model files found in {model_dir}")
            raise FileNotFoundError(f"No {model_type} model files found in {model_dir}")
            
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model: {model_path}")
        
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(config, model_path, target_column=target_column)
    
    # Load data if provided
    if data_path:
        forecaster.load_data(data_path)
    elif data_path is None and not save_forecast:
        # Find latest data if needed for plotting only
        preprocessed_dir = Path(config.output_dir) / "preprocessed"
        data_files = list(preprocessed_dir.glob("*_preprocessed.csv"))
        if data_files:
            data_path = max(data_files, key=lambda p: p.stat().st_mtime)
            forecaster.load_data(data_path)
        
    # Generate forecast
    forecast_df = forecaster.forecast(steps=steps, exog_features=exog_features)
    
    # Analyze forecast
    metrics = forecaster.analyze_forecast(forecast_df)
    
    # Print analysis
    print("\nForecast Analysis:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # Plot forecast
    if plot:
        if save_plot:
            plots_dir = Path(config.output_dir) / "forecasts" / "plots"
            ensure_directory(plots_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plots_dir / f"forecast_{model_type}_{steps}_steps_{timestamp}.png"
            forecaster.plot_forecast(forecast_df, save_path=plot_path)
        else:
            forecaster.plot_forecast(forecast_df)
    
    # Save forecast
    if save_forecast:
        forecasts_dir = Path(config.output_dir) / "forecasts" / "data"
        ensure_directory(forecasts_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forecast_path = forecasts_dir / f"forecast_{model_type}_{steps}_steps_{timestamp}.csv"
        forecast_df.to_csv(forecast_path)
        logger.info(f"Forecast saved to {forecast_path}")
        
    return forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate time series forecasts")
    parser.add_argument("--model", help="Path to trained model file")
    parser.add_argument("--data", help="Path to preprocessed data file")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to forecast")
    parser.add_argument("--target", default="close", help="Target column to predict")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    parser.add_argument("--no-plot", action="store_true", help="Do not display plots")
    parser.add_argument("--model-type", choices=["sarima", "prophet", "auto_arima", "exp_smoothing"],
                     help="Model type to use (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.env_file)
    
    # Generate forecast
    forecast_df = generate_forecast(
        config,
        model_path=args.model,
        data_path=args.data,
        steps=args.steps,
        target_column=args.target,
        plot=not args.no_plot,
        model_type=args.model_type,
    )
