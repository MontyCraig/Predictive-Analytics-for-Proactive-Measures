"""Configuration for Predictive Analytics for Proactive Measures."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, SecretStr, validator
from dotenv import load_dotenv
import os

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class APIConfig(BaseModel):
    """API Configuration for data sources."""

    alpha_vantage_api_key: SecretStr = Field(
        ..., description="Alpha Vantage API key for market data"
    )
    alpha_vantage_base_url: str = Field(
        "https://www.alphavantage.co/query",
        description="Base URL for Alpha Vantage API",
    )


class ModelConfig(BaseModel):
    """Configuration for predictive models."""

    model_type: str = Field(
        "sarima", description="Model type (sarima, prophet, auto_arima, exp_smoothing)"
    )
    train_size: float = Field(0.8, description="Training data proportion")
    test_size: float = Field(0.2, description="Test data proportion")
    random_state: int = Field(42, description="Random seed for reproducibility")

    @validator("test_size")
    def validate_test_size(cls, v, values):
        """Validate that train_size + test_size = 1."""
        if "train_size" in values and abs(values["train_size"] + v - 1.0) > 1e-10:
            raise ValueError("train_size + test_size must equal 1")
        return v


class SARIMAConfig(BaseModel):
    """Configuration for SARIMA models."""

    order: tuple = Field((1, 1, 1), description="SARIMA model order (p, d, q)")
    seasonal_order: tuple = Field(
        (1, 1, 1, 12), description="SARIMA seasonal order (P, D, Q, s)"
    )
    enforce_stationarity: bool = Field(
        True, description="Whether to enforce stationarity"
    )
    enforce_invertibility: bool = Field(
        True, description="Whether to enforce invertibility"
    )


class ProphetConfig(BaseModel):
    """Configuration for Prophet models."""

    yearly_seasonality: bool = Field(
        True, description="Whether to include yearly seasonality"
    )
    weekly_seasonality: bool = Field(
        True, description="Whether to include weekly seasonality"
    )
    daily_seasonality: bool = Field(
        False, description="Whether to include daily seasonality"
    )
    seasonality_mode: str = Field(
        "additive", description="Seasonality mode (additive or multiplicative)"
    )
    changepoint_prior_scale: float = Field(
        0.05, description="Changepoint prior scale parameter"
    )
    seasonality_prior_scale: float = Field(
        10.0, description="Seasonality prior scale parameter"
    )


class AutoARIMAConfig(BaseModel):
    """Configuration for Auto ARIMA models."""

    max_p: int = Field(5, description="Maximum p order to consider")
    max_d: int = Field(2, description="Maximum d order to consider")
    max_q: int = Field(5, description="Maximum q order to consider")
    max_P: int = Field(2, description="Maximum P order to consider")
    max_D: int = Field(1, description="Maximum D order to consider")
    max_Q: int = Field(2, description="Maximum Q order to consider")
    m: int = Field(7, description="Seasonal periodicity")
    seasonal: bool = Field(True, description="Whether to fit seasonal components")
    stepwise: bool = Field(True, description="Whether to use stepwise selection")


class ExponentialSmoothingConfig(BaseModel):
    """Configuration for Exponential Smoothing models."""

    trend: Optional[str] = Field(
        None, description="Trend component (None, 'add', or 'mul')"
    )
    seasonal: Optional[str] = Field(
        None, description="Seasonal component (None, 'add', or 'mul')"
    )
    seasonal_periods: Optional[int] = Field(
        None, description="Number of periods in a season"
    )
    damped_trend: bool = Field(
        False, description="Whether to use a damped trend component"
    )


class Config(BaseModel):
    """Main configuration for the application."""

    api: APIConfig = Field(..., description="API configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    sarima: Optional[SARIMAConfig] = Field(None, description="SARIMA configuration")
    prophet: Optional[ProphetConfig] = Field(None, description="Prophet configuration")
    auto_arima: Optional[AutoARIMAConfig] = Field(
        None, description="Auto ARIMA configuration"
    )
    exp_smoothing: Optional[ExponentialSmoothingConfig] = Field(
        None, description="Exponential Smoothing configuration"
    )
    data_file: Optional[str] = Field(
        None, description="Path to data file if using local data"
    )
    output_dir: str = Field("output", description="Directory to store outputs")
    log_level: str = Field("INFO", description="Logging level")

    @validator("sarima", "prophet", "auto_arima", "exp_smoothing", pre=True)
    def validate_model_config(cls, v, values):
        """Ensure appropriate model config is present based on model_type."""
        if "model" in values:
            model_type = values["model"].model_type.lower()
            if model_type == "sarima" and v is None:
                return SARIMAConfig()
            elif model_type == "prophet" and v is None:
                return ProphetConfig()
            elif model_type == "auto_arima" and v is None:
                return AutoARIMAConfig()
            elif model_type == "exp_smoothing" and v is None:
                return ExponentialSmoothingConfig()
        return v


def get_config(env_file: Optional[str] = None) -> Config:
    """Load configuration from environment variables or .env file.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        Config object with application settings
    """
    # Load environment variables from .env file if provided
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()  # Look for .env in the current directory
    
    # Get Alpha Vantage API key from environment variables
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY "
            "environment variable or provide a .env file."
        )
    
    # Get model configuration from environment variables
    model_type = os.getenv("MODEL_TYPE", "sarima").lower()
    train_size = float(os.getenv("TRAIN_SIZE", "0.8"))
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    
    # Get output directory from environment variables
    output_dir = os.getenv("OUTPUT_DIR", "output")
    
    # Get log level from environment variables
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create API configuration
    api_config = APIConfig(
        alpha_vantage_api_key=api_key,
        alpha_vantage_base_url=os.getenv(
            "ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"
        ),
    )
    
    # Create model configuration
    model_config = ModelConfig(
        model_type=model_type,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Create SARIMA configuration if applicable
    sarima_config = None
    if model_type == "sarima":
        sarima_config = SARIMAConfig(
            order=(
                int(os.getenv("SARIMA_P", "1")),
                int(os.getenv("SARIMA_D", "1")),
                int(os.getenv("SARIMA_Q", "1")),
            ),
            seasonal_order=(
                int(os.getenv("SARIMA_P_SEASONAL", "1")),
                int(os.getenv("SARIMA_D_SEASONAL", "1")),
                int(os.getenv("SARIMA_Q_SEASONAL", "1")),
                int(os.getenv("SARIMA_S", "12")),
            ),
            enforce_stationarity=os.getenv("SARIMA_ENFORCE_STATIONARITY", "True").lower() == "true",
            enforce_invertibility=os.getenv("SARIMA_ENFORCE_INVERTIBILITY", "True").lower() == "true",
        )
    
    # Create Prophet configuration if applicable
    prophet_config = None
    if model_type == "prophet":
        prophet_config = ProphetConfig(
            yearly_seasonality=os.getenv("PROPHET_YEARLY_SEASONALITY", "True").lower() == "true",
            weekly_seasonality=os.getenv("PROPHET_WEEKLY_SEASONALITY", "True").lower() == "true",
            daily_seasonality=os.getenv("PROPHET_DAILY_SEASONALITY", "False").lower() == "true",
            seasonality_mode=os.getenv("PROPHET_SEASONALITY_MODE", "additive"),
            changepoint_prior_scale=float(os.getenv("PROPHET_CHANGEPOINT_PRIOR_SCALE", "0.05")),
            seasonality_prior_scale=float(os.getenv("PROPHET_SEASONALITY_PRIOR_SCALE", "10.0")),
        )
    
    # Create Auto ARIMA configuration if applicable
    auto_arima_config = None
    if model_type == "auto_arima":
        auto_arima_config = AutoARIMAConfig(
            max_p=int(os.getenv("AUTO_ARIMA_MAX_P", "5")),
            max_d=int(os.getenv("AUTO_ARIMA_MAX_D", "2")),
            max_q=int(os.getenv("AUTO_ARIMA_MAX_Q", "5")),
            max_P=int(os.getenv("AUTO_ARIMA_MAX_P_SEASONAL", "2")),
            max_D=int(os.getenv("AUTO_ARIMA_MAX_D_SEASONAL", "1")),
            max_Q=int(os.getenv("AUTO_ARIMA_MAX_Q_SEASONAL", "2")),
            m=int(os.getenv("AUTO_ARIMA_M", "7")),
            seasonal=os.getenv("AUTO_ARIMA_SEASONAL", "True").lower() == "true",
            stepwise=os.getenv("AUTO_ARIMA_STEPWISE", "True").lower() == "true",
        )
    
    # Create Exponential Smoothing configuration if applicable
    exp_smoothing_config = None
    if model_type == "exp_smoothing":
        trend = os.getenv("EXP_SMOOTHING_TREND", "None")
        trend = None if trend.lower() == "none" else trend
        
        seasonal = os.getenv("EXP_SMOOTHING_SEASONAL", "None")
        seasonal = None if seasonal.lower() == "none" else seasonal
        
        seasonal_periods_str = os.getenv("EXP_SMOOTHING_SEASONAL_PERIODS", "None")
        seasonal_periods = None if seasonal_periods_str.lower() == "none" else int(seasonal_periods_str)
        
        exp_smoothing_config = ExponentialSmoothingConfig(
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=os.getenv("EXP_SMOOTHING_DAMPED_TREND", "False").lower() == "true",
        )
    
    # Create main configuration
    config = Config(
        api=api_config,
        model=model_config,
        sarima=sarima_config,
        prophet=prophet_config,
        auto_arima=auto_arima_config,
        exp_smoothing=exp_smoothing_config,
        data_file=os.getenv("DATA_FILE"),
        output_dir=output_dir,
        log_level=log_level,
    )
    
    return config 