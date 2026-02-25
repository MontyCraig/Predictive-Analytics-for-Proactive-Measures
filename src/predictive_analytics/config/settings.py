"""Application settings for Predictive Analytics for Proactive Measures.

This module provides Pydantic v2 validated configuration models for every
component of the predictive-analytics pipeline.  Configuration is loaded
from environment variables (optionally backed by a ``.env`` file) via
:func:`get_config`.

Typical usage::

    from predictive_analytics.config.settings import AppConfig, get_config

    config: AppConfig = get_config()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from predictive_analytics.exceptions import ConfigurationError

__all__ = [
    "APIConfig",
    "ModelConfig",
    "SARIMAConfig",
    "ProphetConfig",
    "AutoARIMAConfig",
    "ExponentialSmoothingConfig",
    "AppConfig",
    "get_config",
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR",
]

# ---------------------------------------------------------------------------
# Project-level directory constants
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
"""Absolute path to the project root directory."""

DATA_DIR: Path = ROOT_DIR / "data"
"""Default directory for data artefacts."""

MODELS_DIR: Path = ROOT_DIR / "models"
"""Default directory for persisted model artefacts."""

# Ensure directories exist at import time.
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class APIConfig(BaseModel):
    """API connection settings for external data sources.

    Attributes:
        alpha_vantage_api_key: Secret API key for Alpha Vantage.
        alpha_vantage_base_url: Base URL for the Alpha Vantage REST API.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    alpha_vantage_api_key: SecretStr = Field(
        ...,
        description="Alpha Vantage API key for market data",
    )
    alpha_vantage_base_url: str = Field(
        "https://www.alphavantage.co/query",
        description="Base URL for Alpha Vantage API",
    )


class ModelConfig(BaseModel):
    """General model hyper-parameter settings.

    Attributes:
        model_type: Which forecasting model to use.
        train_size: Proportion of data reserved for training.
        test_size: Proportion of data reserved for testing.
        random_state: Random seed for reproducibility.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    model_type: str = Field(
        "sarima",
        description="Model type to use for forecasting",
        pattern=r"^(sarima|prophet|auto_arima|exp_smoothing)$",
    )
    train_size: float = Field(
        0.8,
        description="Training data proportion",
        ge=0.0,
        le=1.0,
    )
    test_size: float = Field(
        0.2,
        description="Test data proportion",
        ge=0.0,
        le=1.0,
    )
    random_state: int = Field(
        42,
        description="Random seed for reproducibility",
    )

    @model_validator(mode="after")
    def _check_split_sums_to_one(self) -> ModelConfig:
        """Validate that ``train_size + test_size`` is approximately 1.0."""
        total = self.train_size + self.test_size
        if abs(total - 1.0) > 1e-10:
            raise ConfigurationError(
                f"train_size ({self.train_size}) + test_size ({self.test_size}) "
                f"must equal 1.0, got {total}"
            )
        return self


class SARIMAConfig(BaseModel):
    """Configuration for SARIMA (Seasonal ARIMA) models.

    Attributes:
        order: Non-seasonal ``(p, d, q)`` order of the model.
        seasonal_order: Seasonal ``(P, D, Q, s)`` order of the model.
        enforce_stationarity: Whether to enforce stationarity in the AR component.
        enforce_invertibility: Whether to enforce invertibility in the MA component.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    order: tuple[int, int, int] = Field(
        (1, 1, 1),
        description="SARIMA model order (p, d, q)",
    )
    seasonal_order: tuple[int, int, int, int] = Field(
        (1, 1, 1, 12),
        description="SARIMA seasonal order (P, D, Q, s)",
    )
    enforce_stationarity: bool = Field(
        True,
        description="Whether to enforce stationarity",
    )
    enforce_invertibility: bool = Field(
        True,
        description="Whether to enforce invertibility",
    )


class ProphetConfig(BaseModel):
    """Configuration for Facebook Prophet models.

    Attributes:
        yearly_seasonality: Include yearly seasonality component.
        weekly_seasonality: Include weekly seasonality component.
        daily_seasonality: Include daily seasonality component.
        seasonality_mode: ``"additive"`` or ``"multiplicative"``.
        changepoint_prior_scale: Flexibility of the automatic changepoint selection.
        seasonality_prior_scale: Strength of the seasonality model.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    yearly_seasonality: bool = Field(
        True,
        description="Whether to include yearly seasonality",
    )
    weekly_seasonality: bool = Field(
        True,
        description="Whether to include weekly seasonality",
    )
    daily_seasonality: bool = Field(
        False,
        description="Whether to include daily seasonality",
    )
    seasonality_mode: str = Field(
        "additive",
        description="Seasonality mode (additive or multiplicative)",
    )
    changepoint_prior_scale: float = Field(
        0.05,
        description="Changepoint prior scale parameter",
    )
    seasonality_prior_scale: float = Field(
        10.0,
        description="Seasonality prior scale parameter",
    )


class AutoARIMAConfig(BaseModel):
    """Configuration for Auto ARIMA (automatic order selection) models.

    Attributes:
        max_p: Maximum AR order to search.
        max_d: Maximum differencing order to search.
        max_q: Maximum MA order to search.
        max_P: Maximum seasonal AR order to search.
        max_D: Maximum seasonal differencing order to search.
        max_Q: Maximum seasonal MA order to search.
        m: Number of observations per seasonal cycle.
        seasonal: Whether to fit seasonal ARIMA models.
        stepwise: Whether to use stepwise algorithm for order selection.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

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
    """Configuration for Exponential Smoothing (Holt-Winters) models.

    Attributes:
        trend: Trend component type (``None``, ``"add"``, or ``"mul"``).
        seasonal: Seasonal component type (``None``, ``"add"``, or ``"mul"``).
        seasonal_periods: Number of observations per complete seasonal cycle.
        damped_trend: Whether to damp the trend component.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    trend: Optional[str] = Field(
        None,
        description="Trend component (None, 'add', or 'mul')",
    )
    seasonal: Optional[str] = Field(
        None,
        description="Seasonal component (None, 'add', or 'mul')",
    )
    seasonal_periods: Optional[int] = Field(
        None,
        description="Number of periods in a season",
    )
    damped_trend: bool = Field(
        False,
        description="Whether to use a damped trend component",
    )


# ---------------------------------------------------------------------------
# Main application configuration
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Top-level application configuration.

    Bundles API credentials, model hyper-parameters, and all
    model-specific sub-configurations into a single validated object.

    Attributes:
        api: Alpha Vantage API configuration.
        model: General model settings (type, train/test split, seed).
        sarima: Optional SARIMA-specific settings.
        prophet: Optional Prophet-specific settings.
        auto_arima: Optional Auto ARIMA-specific settings.
        exp_smoothing: Optional Exponential Smoothing-specific settings.
        data_file: Path to a local data file (overrides API collection).
        output_dir: Directory where outputs are written.
        log_level: Python logging level name.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        frozen=False,
    )

    api: APIConfig = Field(..., description="API configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    sarima: Optional[SARIMAConfig] = Field(
        None,
        description="SARIMA configuration",
    )
    prophet: Optional[ProphetConfig] = Field(
        None,
        description="Prophet configuration",
    )
    auto_arima: Optional[AutoARIMAConfig] = Field(
        None,
        description="Auto ARIMA configuration",
    )
    exp_smoothing: Optional[ExponentialSmoothingConfig] = Field(
        None,
        description="Exponential Smoothing configuration",
    )
    data_file: Optional[str] = Field(
        None,
        description="Path to data file if using local data",
    )
    output_dir: str = Field(
        "output",
        description="Directory to store outputs",
    )
    log_level: str = Field(
        "INFO",
        description="Logging level",
    )

    @model_validator(mode="after")
    def _auto_populate_model_sub_config(self) -> AppConfig:
        """Ensure the sub-config matching ``model.model_type`` is populated.

        If the user selected ``sarima`` but did not supply a
        ``SARIMAConfig``, a default instance is created automatically.
        The same applies for every other supported model type.
        """
        model_type = self.model.model_type.lower()

        defaults: dict[str, tuple[str, type[BaseModel]]] = {
            "sarima": ("sarima", SARIMAConfig),
            "prophet": ("prophet", ProphetConfig),
            "auto_arima": ("auto_arima", AutoARIMAConfig),
            "exp_smoothing": ("exp_smoothing", ExponentialSmoothingConfig),
        }

        if model_type in defaults:
            attr_name, config_cls = defaults[model_type]
            if getattr(self, attr_name) is None:
                object.__setattr__(self, attr_name, config_cls())

        return self


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_config(env_file: Optional[str] = None) -> AppConfig:
    """Load application configuration from environment variables.

    Reads the ``.env`` file (if present or explicitly given), pulls all
    relevant ``ALPHA_VANTAGE_*``, ``MODEL_*``, and model-specific
    environment variables, validates them through Pydantic, and returns
    a fully populated :class:`AppConfig`.

    Args:
        env_file: Optional path to a ``.env`` file.  When ``None`` the
            function falls back to ``load_dotenv()`` which searches the
            current working directory.

    Returns:
        A validated :class:`AppConfig` instance.

    Raises:
        ConfigurationError: If required configuration values are
            missing or invalid.
    """
    # Load environment variables from .env file
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # ---- API key (required) ------------------------------------------------
    api_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "Alpha Vantage API key not found. Please set the "
            "ALPHA_VANTAGE_API_KEY environment variable or provide a "
            ".env file."
        )

    # ---- Model configuration -----------------------------------------------
    model_type: str = os.getenv("MODEL_TYPE", "sarima").lower()
    train_size: float = float(os.getenv("TRAIN_SIZE", "0.8"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    api_config = APIConfig(
        alpha_vantage_api_key=SecretStr(api_key),
        alpha_vantage_base_url=os.getenv(
            "ALPHA_VANTAGE_BASE_URL",
            "https://www.alphavantage.co/query",
        ),
    )

    model_config = ModelConfig(
        model_type=model_type,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )

    # ---- Model-specific configurations -------------------------------------
    sarima_config: SARIMAConfig | None = None
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
            enforce_stationarity=(
                os.getenv("SARIMA_ENFORCE_STATIONARITY", "True").lower() == "true"
            ),
            enforce_invertibility=(
                os.getenv("SARIMA_ENFORCE_INVERTIBILITY", "True").lower() == "true"
            ),
        )

    prophet_config: ProphetConfig | None = None
    if model_type == "prophet":
        prophet_config = ProphetConfig(
            yearly_seasonality=(os.getenv("PROPHET_YEARLY_SEASONALITY", "True").lower() == "true"),
            weekly_seasonality=(os.getenv("PROPHET_WEEKLY_SEASONALITY", "True").lower() == "true"),
            daily_seasonality=(os.getenv("PROPHET_DAILY_SEASONALITY", "False").lower() == "true"),
            seasonality_mode=os.getenv("PROPHET_SEASONALITY_MODE", "additive"),
            changepoint_prior_scale=float(os.getenv("PROPHET_CHANGEPOINT_PRIOR_SCALE", "0.05")),
            seasonality_prior_scale=float(os.getenv("PROPHET_SEASONALITY_PRIOR_SCALE", "10.0")),
        )

    auto_arima_config: AutoARIMAConfig | None = None
    if model_type == "auto_arima":
        auto_arima_config = AutoARIMAConfig(
            max_p=int(os.getenv("AUTO_ARIMA_MAX_P", "5")),
            max_d=int(os.getenv("AUTO_ARIMA_MAX_D", "2")),
            max_q=int(os.getenv("AUTO_ARIMA_MAX_Q", "5")),
            max_P=int(os.getenv("AUTO_ARIMA_MAX_P_SEASONAL", "2")),
            max_D=int(os.getenv("AUTO_ARIMA_MAX_D_SEASONAL", "1")),
            max_Q=int(os.getenv("AUTO_ARIMA_MAX_Q_SEASONAL", "2")),
            m=int(os.getenv("AUTO_ARIMA_M", "7")),
            seasonal=(os.getenv("AUTO_ARIMA_SEASONAL", "True").lower() == "true"),
            stepwise=(os.getenv("AUTO_ARIMA_STEPWISE", "True").lower() == "true"),
        )

    exp_smoothing_config: ExponentialSmoothingConfig | None = None
    if model_type == "exp_smoothing":
        trend_raw: str = os.getenv("EXP_SMOOTHING_TREND", "None")
        trend_val: str | None = None if trend_raw.lower() == "none" else trend_raw

        seasonal_raw: str = os.getenv("EXP_SMOOTHING_SEASONAL", "None")
        seasonal_val: str | None = None if seasonal_raw.lower() == "none" else seasonal_raw

        periods_raw: str = os.getenv("EXP_SMOOTHING_SEASONAL_PERIODS", "None")
        seasonal_periods_val: int | None = (
            None if periods_raw.lower() == "none" else int(periods_raw)
        )

        exp_smoothing_config = ExponentialSmoothingConfig(
            trend=trend_val,
            seasonal=seasonal_val,
            seasonal_periods=seasonal_periods_val,
            damped_trend=(os.getenv("EXP_SMOOTHING_DAMPED_TREND", "False").lower() == "true"),
        )

    # ---- Assemble top-level config -----------------------------------------
    try:
        config = AppConfig(
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
    except Exception as exc:
        raise ConfigurationError(f"Failed to build application configuration: {exc}") from exc

    return config
