"""Tests for the predictive_analytics.config.settings module.

Covers:
- Directory constants (ROOT_DIR, DATA_DIR, MODELS_DIR).
- All Pydantic sub-configuration models and their validation rules.
- The AppConfig model validator (_auto_populate_model_sub_config).
- The get_config() factory function for every model_type branch,
  including missing API keys and invalid configuration values.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from predictive_analytics.config.settings import (
    APIConfig,
    AppConfig,
    AutoARIMAConfig,
    DATA_DIR,
    ExponentialSmoothingConfig,
    MODELS_DIR,
    ModelConfig,
    ProphetConfig,
    ROOT_DIR,
    SARIMAConfig,
    get_config,
)
from predictive_analytics.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------


class TestDirectoryConstants:
    """Verify project-level path constants."""

    def test_root_dir_is_absolute(self) -> None:
        assert ROOT_DIR.is_absolute()

    def test_data_dir_under_root(self) -> None:
        assert DATA_DIR == ROOT_DIR / "data"

    def test_models_dir_under_root(self) -> None:
        assert MODELS_DIR == ROOT_DIR / "models"

    def test_data_dir_exists(self) -> None:
        assert DATA_DIR.exists()

    def test_models_dir_exists(self) -> None:
        assert MODELS_DIR.exists()


# ---------------------------------------------------------------------------
# APIConfig
# ---------------------------------------------------------------------------


class TestAPIConfig:
    """Tests for the APIConfig Pydantic model."""

    def test_valid_api_config(self) -> None:
        cfg = APIConfig(alpha_vantage_api_key=SecretStr("my-key"))
        assert cfg.alpha_vantage_api_key.get_secret_value() == "my-key"
        assert cfg.alpha_vantage_base_url == "https://www.alphavantage.co/query"

    def test_custom_base_url(self) -> None:
        cfg = APIConfig(
            alpha_vantage_api_key=SecretStr("key"),
            alpha_vantage_base_url="https://custom.api/query",
        )
        assert cfg.alpha_vantage_base_url == "https://custom.api/query"

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValidationError):
            APIConfig()  # type: ignore[call-arg]

    def test_secret_str_hides_value(self) -> None:
        cfg = APIConfig(alpha_vantage_api_key=SecretStr("secret-key"))
        # repr / str should not leak the key
        assert "secret-key" not in repr(cfg.alpha_vantage_api_key)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for the ModelConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.model_type == "sarima"
        assert cfg.train_size == 0.8
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42

    def test_valid_model_types(self) -> None:
        for mtype in ("sarima", "prophet", "auto_arima", "exp_smoothing"):
            cfg = ModelConfig(model_type=mtype)
            assert cfg.model_type == mtype

    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ModelConfig(model_type="xgboost")

    def test_train_test_must_sum_to_one(self) -> None:
        with pytest.raises((ValidationError, ConfigurationError)):
            ModelConfig(train_size=0.5, test_size=0.3)

    def test_train_test_exact_one(self) -> None:
        cfg = ModelConfig(train_size=0.7, test_size=0.3)
        assert abs(cfg.train_size + cfg.test_size - 1.0) < 1e-10

    def test_train_size_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(train_size=1.5)

    def test_test_size_negative(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(test_size=-0.1)

    def test_custom_random_state(self) -> None:
        cfg = ModelConfig(random_state=123)
        assert cfg.random_state == 123


# ---------------------------------------------------------------------------
# SARIMAConfig
# ---------------------------------------------------------------------------


class TestSARIMAConfig:
    """Tests for SARIMAConfig."""

    def test_defaults(self) -> None:
        cfg = SARIMAConfig()
        assert cfg.order == (1, 1, 1)
        assert cfg.seasonal_order == (1, 1, 1, 12)
        assert cfg.enforce_stationarity is True
        assert cfg.enforce_invertibility is True

    def test_custom_order(self) -> None:
        cfg = SARIMAConfig(order=(2, 0, 1), seasonal_order=(1, 0, 1, 7))
        assert cfg.order == (2, 0, 1)
        assert cfg.seasonal_order == (1, 0, 1, 7)

    def test_stationarity_can_be_disabled(self) -> None:
        cfg = SARIMAConfig(enforce_stationarity=False)
        assert cfg.enforce_stationarity is False

    def test_invertibility_can_be_disabled(self) -> None:
        cfg = SARIMAConfig(enforce_invertibility=False)
        assert cfg.enforce_invertibility is False


# ---------------------------------------------------------------------------
# ProphetConfig
# ---------------------------------------------------------------------------


class TestProphetConfig:
    """Tests for ProphetConfig."""

    def test_defaults(self) -> None:
        cfg = ProphetConfig()
        assert cfg.yearly_seasonality is True
        assert cfg.weekly_seasonality is True
        assert cfg.daily_seasonality is False
        assert cfg.seasonality_mode == "additive"
        assert cfg.changepoint_prior_scale == 0.05
        assert cfg.seasonality_prior_scale == 10.0

    def test_multiplicative_seasonality(self) -> None:
        cfg = ProphetConfig(seasonality_mode="multiplicative")
        assert cfg.seasonality_mode == "multiplicative"

    def test_custom_scales(self) -> None:
        cfg = ProphetConfig(
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=5.0,
        )
        assert cfg.changepoint_prior_scale == 0.1
        assert cfg.seasonality_prior_scale == 5.0


# ---------------------------------------------------------------------------
# AutoARIMAConfig
# ---------------------------------------------------------------------------


class TestAutoARIMAConfig:
    """Tests for AutoARIMAConfig."""

    def test_defaults(self) -> None:
        cfg = AutoARIMAConfig()
        assert cfg.max_p == 5
        assert cfg.max_d == 2
        assert cfg.max_q == 5
        assert cfg.max_P == 2
        assert cfg.max_D == 1
        assert cfg.max_Q == 2
        assert cfg.m == 7
        assert cfg.seasonal is True
        assert cfg.stepwise is True

    def test_non_seasonal(self) -> None:
        cfg = AutoARIMAConfig(seasonal=False, stepwise=False)
        assert cfg.seasonal is False
        assert cfg.stepwise is False

    def test_custom_max_orders(self) -> None:
        cfg = AutoARIMAConfig(max_p=3, max_q=3, m=12)
        assert cfg.max_p == 3
        assert cfg.max_q == 3
        assert cfg.m == 12


# ---------------------------------------------------------------------------
# ExponentialSmoothingConfig
# ---------------------------------------------------------------------------


class TestExponentialSmoothingConfig:
    """Tests for ExponentialSmoothingConfig."""

    def test_defaults(self) -> None:
        cfg = ExponentialSmoothingConfig()
        assert cfg.trend is None
        assert cfg.seasonal is None
        assert cfg.seasonal_periods is None
        assert cfg.damped_trend is False

    def test_additive_trend_and_seasonal(self) -> None:
        cfg = ExponentialSmoothingConfig(
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            damped_trend=True,
        )
        assert cfg.trend == "add"
        assert cfg.seasonal == "add"
        assert cfg.seasonal_periods == 12
        assert cfg.damped_trend is True

    def test_multiplicative(self) -> None:
        cfg = ExponentialSmoothingConfig(trend="mul", seasonal="mul")
        assert cfg.trend == "mul"
        assert cfg.seasonal == "mul"


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    """Tests for the top-level AppConfig and its model validator."""

    def _make_app_config(
        self,
        model_type: str = "sarima",
        **kwargs: object,
    ) -> AppConfig:
        api = APIConfig(alpha_vantage_api_key=SecretStr("test-key"))
        model = ModelConfig(model_type=model_type)
        return AppConfig(api=api, model=model, **kwargs)

    def test_sarima_auto_populated(self) -> None:
        cfg = self._make_app_config("sarima")
        assert cfg.sarima is not None
        assert isinstance(cfg.sarima, SARIMAConfig)

    def test_prophet_auto_populated(self) -> None:
        cfg = self._make_app_config("prophet")
        assert cfg.prophet is not None
        assert isinstance(cfg.prophet, ProphetConfig)

    def test_auto_arima_auto_populated(self) -> None:
        cfg = self._make_app_config("auto_arima")
        assert cfg.auto_arima is not None
        assert isinstance(cfg.auto_arima, AutoARIMAConfig)

    def test_exp_smoothing_auto_populated(self) -> None:
        cfg = self._make_app_config("exp_smoothing")
        assert cfg.exp_smoothing is not None
        assert isinstance(cfg.exp_smoothing, ExponentialSmoothingConfig)

    def test_explicit_sub_config_not_overridden(self) -> None:
        custom_sarima = SARIMAConfig(order=(2, 1, 0))
        cfg = self._make_app_config("sarima", sarima=custom_sarima)
        assert cfg.sarima is not None
        assert cfg.sarima.order == (2, 1, 0)

    def test_default_output_dir(self) -> None:
        cfg = self._make_app_config()
        assert cfg.output_dir == "output"

    def test_default_log_level(self) -> None:
        cfg = self._make_app_config()
        assert cfg.log_level == "INFO"

    def test_custom_data_file(self) -> None:
        cfg = self._make_app_config(data_file="/tmp/data.csv")
        assert cfg.data_file == "/tmp/data.csv"

    def test_data_file_defaults_to_none(self) -> None:
        cfg = self._make_app_config()
        assert cfg.data_file is None

    def test_other_sub_configs_remain_none(self) -> None:
        cfg = self._make_app_config("sarima")
        assert cfg.prophet is None
        assert cfg.auto_arima is None
        assert cfg.exp_smoothing is None


# ---------------------------------------------------------------------------
# get_config() factory function
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for the get_config() factory function."""

    def _env(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """Build a minimal valid env-var mapping."""
        base = {"ALPHA_VANTAGE_API_KEY": "test-key-abc"}
        if overrides:
            base.update(overrides)
        return base

    def test_missing_api_key_raises_configuration_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            with pytest.raises(ConfigurationError, match="API key"):
                get_config()

    def test_default_sarima_config(self) -> None:
        with patch.dict("os.environ", self._env(), clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.model_type == "sarima"
            assert cfg.sarima is not None

    def test_prophet_config_from_env(self) -> None:
        env = self._env(
            {
                "MODEL_TYPE": "prophet",
                "PROPHET_YEARLY_SEASONALITY": "False",
                "PROPHET_WEEKLY_SEASONALITY": "True",
                "PROPHET_DAILY_SEASONALITY": "True",
                "PROPHET_SEASONALITY_MODE": "multiplicative",
                "PROPHET_CHANGEPOINT_PRIOR_SCALE": "0.1",
                "PROPHET_SEASONALITY_PRIOR_SCALE": "5.0",
            }
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.model_type == "prophet"
            assert cfg.prophet is not None
            assert cfg.prophet.yearly_seasonality is False
            assert cfg.prophet.daily_seasonality is True
            assert cfg.prophet.seasonality_mode == "multiplicative"
            assert cfg.prophet.changepoint_prior_scale == 0.1
            assert cfg.prophet.seasonality_prior_scale == 5.0

    def test_auto_arima_config_from_env(self) -> None:
        env = self._env(
            {
                "MODEL_TYPE": "auto_arima",
                "AUTO_ARIMA_MAX_P": "3",
                "AUTO_ARIMA_MAX_D": "1",
                "AUTO_ARIMA_MAX_Q": "3",
                "AUTO_ARIMA_MAX_P_SEASONAL": "1",
                "AUTO_ARIMA_MAX_D_SEASONAL": "0",
                "AUTO_ARIMA_MAX_Q_SEASONAL": "1",
                "AUTO_ARIMA_M": "12",
                "AUTO_ARIMA_SEASONAL": "False",
                "AUTO_ARIMA_STEPWISE": "False",
            }
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.model_type == "auto_arima"
            assert cfg.auto_arima is not None
            assert cfg.auto_arima.max_p == 3
            assert cfg.auto_arima.max_d == 1
            assert cfg.auto_arima.max_q == 3
            assert cfg.auto_arima.max_P == 1
            assert cfg.auto_arima.max_D == 0
            assert cfg.auto_arima.max_Q == 1
            assert cfg.auto_arima.m == 12
            assert cfg.auto_arima.seasonal is False
            assert cfg.auto_arima.stepwise is False

    def test_exp_smoothing_config_from_env(self) -> None:
        env = self._env(
            {
                "MODEL_TYPE": "exp_smoothing",
                "EXP_SMOOTHING_TREND": "add",
                "EXP_SMOOTHING_SEASONAL": "mul",
                "EXP_SMOOTHING_SEASONAL_PERIODS": "12",
                "EXP_SMOOTHING_DAMPED_TREND": "True",
            }
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.model_type == "exp_smoothing"
            assert cfg.exp_smoothing is not None
            assert cfg.exp_smoothing.trend == "add"
            assert cfg.exp_smoothing.seasonal == "mul"
            assert cfg.exp_smoothing.seasonal_periods == 12
            assert cfg.exp_smoothing.damped_trend is True

    def test_exp_smoothing_none_values(self) -> None:
        """When env vars are set to 'None', the config should store None."""
        env = self._env(
            {
                "MODEL_TYPE": "exp_smoothing",
                "EXP_SMOOTHING_TREND": "None",
                "EXP_SMOOTHING_SEASONAL": "None",
                "EXP_SMOOTHING_SEASONAL_PERIODS": "None",
                "EXP_SMOOTHING_DAMPED_TREND": "False",
            }
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.exp_smoothing is not None
            assert cfg.exp_smoothing.trend is None
            assert cfg.exp_smoothing.seasonal is None
            assert cfg.exp_smoothing.seasonal_periods is None
            assert cfg.exp_smoothing.damped_trend is False

    def test_sarima_env_overrides(self) -> None:
        env = self._env(
            {
                "MODEL_TYPE": "sarima",
                "SARIMA_P": "2",
                "SARIMA_D": "0",
                "SARIMA_Q": "2",
                "SARIMA_P_SEASONAL": "1",
                "SARIMA_D_SEASONAL": "0",
                "SARIMA_Q_SEASONAL": "1",
                "SARIMA_S": "7",
                "SARIMA_ENFORCE_STATIONARITY": "False",
                "SARIMA_ENFORCE_INVERTIBILITY": "False",
            }
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.sarima is not None
            assert cfg.sarima.order == (2, 0, 2)
            assert cfg.sarima.seasonal_order == (1, 0, 1, 7)
            assert cfg.sarima.enforce_stationarity is False
            assert cfg.sarima.enforce_invertibility is False

    def test_custom_base_url_from_env(self) -> None:
        env = self._env(
            {"ALPHA_VANTAGE_BASE_URL": "https://custom.api/v1"}
        )
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.api.alpha_vantage_base_url == "https://custom.api/v1"

    def test_custom_output_dir_and_log_level(self) -> None:
        env = self._env({"OUTPUT_DIR": "/tmp/out", "LOG_LEVEL": "DEBUG"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.output_dir == "/tmp/out"
            assert cfg.log_level == "DEBUG"

    def test_data_file_from_env(self) -> None:
        env = self._env({"DATA_FILE": "/data/prices.csv"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.data_file == "/data/prices.csv"

    def test_train_test_sizes_from_env(self) -> None:
        env = self._env({"TRAIN_SIZE": "0.7", "TEST_SIZE": "0.3"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.train_size == 0.7
            assert cfg.model.test_size == 0.3

    def test_invalid_train_test_split_raises(self) -> None:
        env = self._env({"TRAIN_SIZE": "0.5", "TEST_SIZE": "0.3"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            with pytest.raises(ConfigurationError):
                get_config()

    def test_env_file_passed_to_load_dotenv(self) -> None:
        env = self._env()
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ) as mock_load:
            get_config(env_file="/path/to/.env")
            mock_load.assert_called_once_with("/path/to/.env")

    def test_no_env_file_calls_load_dotenv_without_args(self) -> None:
        env = self._env()
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ) as mock_load:
            get_config()
            mock_load.assert_called_once_with()

    def test_random_state_from_env(self) -> None:
        env = self._env({"RANDOM_STATE": "99"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.random_state == 99

    def test_model_type_case_insensitive(self) -> None:
        env = self._env({"MODEL_TYPE": "SARIMA"})
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ):
            cfg = get_config()
            assert cfg.model.model_type == "sarima"

    def test_generic_exception_during_app_config_build(self) -> None:
        """Trigger the generic except-Exception catch in get_config (line 546)."""
        env = self._env()
        with patch.dict("os.environ", env, clear=True), patch(
            "predictive_analytics.config.settings.load_dotenv"
        ), patch(
            "predictive_analytics.config.settings.AppConfig",
            side_effect=TypeError("unexpected type"),
        ):
            with pytest.raises(ConfigurationError, match="Failed to build"):
                get_config()


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestSettingsModuleExports:
    """Verify that __all__ exports the expected names."""

    def test_all_contains_expected_names(self) -> None:
        from predictive_analytics.config import settings

        expected = {
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
        }
        assert set(settings.__all__) == expected
