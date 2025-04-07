"""Model selection and training for time series prediction."""
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging
from utils.helpers import ensure_directory

# Initialize logger
logger = setup_logging()


class ModelTrainer:
    """Trainer for time series forecasting models."""

    def __init__(
        self, 
        config: Config,
        data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ):
        """Initialize the model trainer.
        
        Args:
            config: Application configuration
            data: Time series data (if None, it will be loaded)
            target_column: Target column to predict
        """
        self.config = config
        self.data = data
        self.target_column = target_column
        self.model = None
        self.model_fit = None
        self.train_data = None
        self.test_data = None
        self.model_type = config.model.model_type.lower()
        
        logger.info(f"ModelTrainer initialized with model type: {self.model_type}")

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

    def split_data(
        self, test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            logger.error("Data not loaded. Please load data first.")
            raise ValueError("Data not loaded. Please load data first.")
            
        if test_size is None:
            test_size = self.config.model.test_size
            
        # Split based on time
        split_idx = int(len(self.data) * (1 - test_size))
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        logger.info(
            f"Data split: train_size={len(self.train_data)}, "
            f"test_size={len(self.test_data)}"
        )
        
        return self.train_data, self.test_data

    def train_sarima_model(
        self, 
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        exog_columns: Optional[List[str]] = None,
    ) -> None:
        """Train a SARIMA model.
        
        Args:
            order: SARIMA order (p, d, q)
            seasonal_order: SARIMA seasonal order (P, D, Q, s)
            exog_columns: List of exogenous variables to include
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise ValueError("Data not split. Please split data first.")
            
        # Use config values if not provided
        if order is None:
            order = self.config.sarima.order
            
        if seasonal_order is None:
            seasonal_order = self.config.sarima.seasonal_order
            
        # Prepare exogenous variables if specified
        train_exog = None
        if exog_columns:
            if any(col not in self.train_data.columns for col in exog_columns):
                missing_cols = [col for col in exog_columns if col not in self.train_data.columns]
                logger.error(f"Exogenous columns not found in data: {missing_cols}")
                raise ValueError(f"Exogenous columns not found in data: {missing_cols}")
                
            train_exog = self.train_data[exog_columns]
            
        logger.info(
            f"Training SARIMA model with order={order}, "
            f"seasonal_order={seasonal_order}"
        )
        
        # Create and fit model
        self.model = SARIMAX(
            self.train_data[self.target_column],
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=self.config.sarima.enforce_stationarity,
            enforce_invertibility=self.config.sarima.enforce_invertibility,
        )
        
        self.model_fit = self.model.fit(disp=False)
        
        logger.info("SARIMA model training completed successfully")
        
        # Print model summary
        logger.info("Model summary:")
        print(self.model_fit.summary())
        
        # Store model metadata
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
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            exog_columns: List of exogenous variables to include as regressors
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise ValueError("Data not split. Please split data first.")
            
        logger.info(
            f"Training Prophet model with "
            f"yearly_seasonality={yearly_seasonality}, "
            f"weekly_seasonality={weekly_seasonality}, "
            f"daily_seasonality={daily_seasonality}"
        )
        
        # Prophet requires a specific dataframe format with 'ds' and 'y' columns
        prophet_data = self.train_data.copy().reset_index()
        prophet_data.rename(columns={prophet_data.columns[0]: 'ds', 
                                    self.target_column: 'y'}, 
                          inplace=True)
        
        # Initialize Prophet model with seasonality options
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
        )
        
        # Add regressors if specified
        if exog_columns:
            for col in exog_columns:
                if col not in self.train_data.columns:
                    logger.error(f"Regressor column not found in data: {col}")
                    raise ValueError(f"Regressor column not found in data: {col}")
                
                self.model.add_regressor(col)
                logger.info(f"Added regressor: {col}")
        
        # Fit the model
        self.model_fit = self.model.fit(prophet_data)
        
        logger.info("Prophet model training completed successfully")
        
        # Store model metadata
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
        m: int = 7,  # Seasonal periodicity
        exog_columns: Optional[List[str]] = None,
    ) -> None:
        """Train an Auto ARIMA model using pmdarima.
        
        Args:
            max_p: Maximum p order to consider
            max_d: Maximum d order to consider
            max_q: Maximum q order to consider
            max_P: Maximum P order to consider
            max_D: Maximum D order to consider
            max_Q: Maximum Q order to consider
            m: Seasonal periodicity
            exog_columns: List of exogenous variables to include
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise ValueError("Data not split. Please split data first.")
        
        logger.info(
            f"Training Auto ARIMA model with "
            f"max_p={max_p}, max_d={max_d}, max_q={max_q}, "
            f"max_P={max_P}, max_D={max_D}, max_Q={max_Q}, m={m}"
        )
        
        # Prepare exogenous variables if specified
        train_exog = None
        if exog_columns:
            if any(col not in self.train_data.columns for col in exog_columns):
                missing_cols = [col for col in exog_columns if col not in self.train_data.columns]
                logger.error(f"Exogenous columns not found in data: {missing_cols}")
                raise ValueError(f"Exogenous columns not found in data: {missing_cols}")
                
            train_exog = self.train_data[exog_columns].values
        
        # Fit Auto ARIMA model
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
            d=None,  # auto-determine differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
        )
        
        self.model_fit = self.model
        
        logger.info(f"Auto ARIMA model training completed successfully")
        logger.info(f"Best model: {self.model}")
        
        # Store model metadata
        best_order = self.model.order
        best_seasonal_order = self.model.seasonal_order
        
        self.model_meta = {
            "type": "auto_arima",
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "exog_columns": exog_columns,
        }

    def train_exponential_smoothing_model(
        self,
        trend: Optional[str] = None,  # 'add', 'mul', None
        seasonal: Optional[str] = None,  # 'add', 'mul', None
        seasonal_periods: Optional[int] = None,
    ) -> None:
        """Train an Exponential Smoothing model.
        
        Args:
            trend: Type of trend component ('add', 'mul', or None)
            seasonal: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a season
        """
        if self.train_data is None:
            logger.error("Data not split. Please split data first.")
            raise ValueError("Data not split. Please split data first.")
        
        logger.info(
            f"Training Exponential Smoothing model with "
            f"trend={trend}, seasonal={seasonal}, "
            f"seasonal_periods={seasonal_periods}"
        )
        
        # If seasonal is specified but not seasonal_periods, determine it
        if seasonal is not None and seasonal_periods is None:
            # Default to weekly seasonality (7 days)
            seasonal_periods = 7
            logger.info(f"No seasonal_periods specified, defaulting to {seasonal_periods}")
        
        # Create and fit the model
        self.model = ExponentialSmoothing(
            self.train_data[self.target_column],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        
        self.model_fit = self.model.fit()
        
        logger.info("Exponential Smoothing model training completed successfully")
        
        # Store model metadata
        self.model_meta = {
            "type": "exponential_smoothing",
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
        }

    def train_model(self, model_type: Optional[str] = None, **kwargs) -> None:
        """Train a time series forecasting model.
        
        Args:
            model_type: Type of model to train ('sarima', 'prophet', 'auto_arima', 'exp_smoothing')
            **kwargs: Additional arguments for specific model types
        """
        if model_type is None:
            model_type = self.model_type
        
        logger.info(f"Training model of type: {model_type}")
        
        if model_type == 'sarima':
            self.train_sarima_model(**kwargs)
        elif model_type == 'prophet':
            self.train_prophet_model(**kwargs)
        elif model_type == 'auto_arima':
            self.train_auto_arima_model(**kwargs)
        elif model_type == 'exp_smoothing':
            self.train_exponential_smoothing_model(**kwargs)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

    def evaluate_model(
        self, exog_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the trained model on test data.
        
        Args:
            exog_columns: List of exogenous variables to include
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model_fit is None:
            logger.error("Model not trained. Please train a model first.")
            raise ValueError("Model not trained. Please train a model first.")
        
        logger.info(f"Evaluating {self.model_meta['type']} model")
        
        # Different prediction approach based on model type
        if self.model_meta['type'] == 'sarima':
            # Prepare exogenous variables if specified
            test_exog = None
            if exog_columns:
                if any(col not in self.test_data.columns for col in exog_columns):
                    missing_cols = [col for col in exog_columns if col not in self.test_data.columns]
                    logger.error(f"Exogenous columns not found in data: {missing_cols}")
                    raise ValueError(f"Exogenous columns not found in data: {missing_cols}")
                    
                test_exog = self.test_data[exog_columns]
                
            # Make predictions
            logger.info("Generating predictions for test data")
            predictions = self.model_fit.get_forecast(
                steps=len(self.test_data), exog=test_exog
            ).predicted_mean
            
            # Ensure predictions index matches test data
            predictions.index = self.test_data.index
            
        elif self.model_meta['type'] == 'prophet':
            # Create future DataFrame for Prophet
            future = self.model_fit.make_future_dataframe(
                periods=len(self.test_data),
                freq=pd.infer_freq(self.data.index)
            )
            
            # Add regressors if specified
            if exog_columns:
                test_data_reset = self.test_data.copy().reset_index()
                for col in exog_columns:
                    # Add regressor values for the forecast period
                    future.loc[len(future) - len(self.test_data):, col] = test_data_reset[col].values
            
            # Generate forecast
            forecast = self.model_fit.predict(future)
            
            # Extract predictions for the test period
            predictions = forecast.iloc[-len(self.test_data):]['yhat']
            predictions.index = self.test_data.index
            
        elif self.model_meta['type'] == 'auto_arima':
            # Prepare exogenous variables if specified
            test_exog = None
            if exog_columns:
                if any(col not in self.test_data.columns for col in exog_columns):
                    missing_cols = [col for col in exog_columns if col not in self.test_data.columns]
                    logger.error(f"Exogenous columns not found in data: {missing_cols}")
                    raise ValueError(f"Exogenous columns not found in data: {missing_cols}")
                    
                test_exog = self.test_data[exog_columns].values
            
            # Generate predictions
            predictions, conf_int = self.model_fit.predict(
                n_periods=len(self.test_data),
                exogenous=test_exog,
                return_conf_int=True
            )
            
            # Convert predictions to Series with proper index
            predictions = pd.Series(predictions, index=self.test_data.index)
            
        elif self.model_meta['type'] == 'exponential_smoothing':
            # Generate predictions
            predictions = self.model_fit.forecast(len(self.test_data))
            
            # Ensure predictions index matches test data
            predictions.index = self.test_data.index
            
        else:
            logger.error(f"Unsupported model type for evaluation: {self.model_meta['type']}")
            raise ValueError(f"Unsupported model type for evaluation: {self.model_meta['type']}")
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(self.test_data[self.target_column], predictions)
        mse = mean_squared_error(self.test_data[self.target_column], predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.test_data[self.target_column], predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(
            np.abs(
                (self.test_data[self.target_column] - predictions) 
                / self.test_data[self.target_column]
            )
        ) * 100
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Store predictions for later use
        self.predictions = predictions
        
        return metrics

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Save the trained model to a file.
        
        Args:
            file_path: Path to save the model
        """
        if self.model_fit is None:
            logger.error("Model not trained. Please train a model first.")
            raise ValueError("Model not trained. Please train a model first.")
            
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        try:
            # Create a dictionary with model and metadata
            model_package = {
                "model": self.model_fit,
                "metadata": self.model_meta,
                "target_column": self.target_column,
                "training_end_date": self.train_data.index[-1],
            }
            
            with open(file_path, "wb") as f:
                pickle.dump(model_package, f)
                
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {str(e)}")
            raise

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
            else:
                # Legacy format where only the model was saved
                self.model_fit = model_package
                self.model_meta = {"type": "unknown"}
                
            logger.info(f"Model loaded from {file_path}")
            logger.info(f"Model type: {self.model_meta.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {str(e)}")
            raise

    def plot_results(self, save_path: Optional[Path] = None) -> None:
        """Plot actual vs predicted values.
        
        Args:
            save_path: Path to save the plot
        """
        if self.model_fit is None or not hasattr(self, "predictions"):
            logger.error("Model not evaluated. Please evaluate the model first.")
            raise ValueError("Model not evaluated. Please evaluate the model first.")
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        
        # Plot train data
        plt.plot(
            self.train_data.index, 
            self.train_data[self.target_column],
            label="Train Data",
            color="blue",
        )
        
        # Plot test data
        plt.plot(
            self.test_data.index,
            self.test_data[self.target_column],
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
        
        # Get model type from metadata
        model_type = self.model_meta.get('type', 'Unknown')
        
        plt.title(f"{model_type.upper()} Model: Actual vs Predicted Values")
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            ensure_directory(save_path.parent)
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def train_and_evaluate_model(
    config: Config,
    data: Optional[pd.DataFrame] = None,
    file_path: Optional[Union[str, Path]] = None,
    target_column: str = "close",
    exog_columns: Optional[List[str]] = None,
    save_model: bool = True,
    save_plot: bool = True,
    model_type: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Train and evaluate a time series model.
    
    Args:
        config: Application configuration
        data: Time series data (if None, it will be loaded from file_path)
        file_path: Path to data file (used if data is None)
        target_column: Target column to predict
        exog_columns: List of exogenous variables to include
        save_model: Whether to save the trained model
        save_plot: Whether to save the results plot
        model_type: Override model type from config
        model_params: Additional parameters for the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get model type from params or config
    if model_type is None:
        model_type = config.model.model_type.lower()
    
    # Initialize default model params if not provided
    if model_params is None:
        model_params = {}
    
    # Initialize trainer
    trainer = ModelTrainer(config, data, target_column)
    
    # Load data if not provided
    if data is None and file_path:
        trainer.load_data(file_path)
    elif data is None:
        preprocessed_dir = Path(config.output_dir) / "preprocessed"
        symbol = "MSFT"  # Default symbol
        file_path = preprocessed_dir / f"{symbol}_preprocessed.csv"
        trainer.load_data(file_path)
        
    # Split data
    trainer.split_data()
    
    # Train model based on type
    logger.info(f"Training {model_type} model with params: {model_params}")
    
    try:
        if model_type == "sarima":
            trainer.train_sarima_model(exog_columns=exog_columns, **model_params)
        elif model_type == "prophet":
            trainer.train_prophet_model(exog_columns=exog_columns, **model_params)
        elif model_type == "auto_arima":
            trainer.train_auto_arima_model(exog_columns=exog_columns, **model_params)
        elif model_type == "exp_smoothing":
            trainer.train_exponential_smoothing_model(**model_params)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate model
        metrics = trainer.evaluate_model(exog_columns=exog_columns)
        
        # Save model
        if save_model:
            model_dir = Path(config.output_dir) / "models"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"{model_type}_{timestamp}.pkl"
            trainer.save_model(model_path)
            
        # Plot results
        if save_plot:
            plots_dir = Path(config.output_dir) / "plots"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plots_dir / f"{model_type}_results_{timestamp}.png"
            trainer.plot_results(plot_path)
        else:
            trainer.plot_results()
            
        return metrics
    
    except Exception as e:
        logger.error(f"Error training and evaluating model: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate time series model")
    parser.add_argument("--file", help="Path to preprocessed data file")
    parser.add_argument("--symbol", default="MSFT", help="Stock symbol to analyze")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    parser.add_argument("--target", default="close", help="Target column to predict")
    parser.add_argument("--exog", help="Comma-separated list of exogenous variables")
    parser.add_argument("--model", choices=["sarima", "prophet", "auto_arima", "exp_smoothing"],
                        help="Model type to use (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.env_file)
    
    # Parse exogenous variables
    exog_columns = None
    if args.exog:
        exog_columns = [col.strip() for col in args.exog.split(",")]
    
    # Load data
    if args.file:
        file_path = args.file
    else:
        # Use data from previous step
        from data_collection_and_preprocessing import collect_and_preprocess_data
        
        data = collect_and_preprocess_data(config, args.symbol)
        file_path = None
    
    # Train and evaluate model
    metrics = train_and_evaluate_model(
        config,
        data=None if args.file else data,
        file_path=args.file,
        target_column=args.target,
        exog_columns=exog_columns,
        model_type=args.model,
    )
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
