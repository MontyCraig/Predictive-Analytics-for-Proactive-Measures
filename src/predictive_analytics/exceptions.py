"""Custom exception hierarchy for the Predictive Analytics package.

This module defines all domain-specific exceptions used throughout the
application, providing structured error handling with clear semantics
for each failure mode.
"""

from __future__ import annotations

__all__ = [
    "PredictiveAnalyticsError",
    "ConfigurationError",
    "DataCollectionError",
    "APIRateLimitError",
    "DataPreprocessingError",
    "DataNotLoadedError",
    "ModelTrainingError",
    "ModelNotTrainedError",
    "ForecastingError",
    "DisruptionAnalysisError",
]


class PredictiveAnalyticsError(Exception):
    """Base exception for all predictive-analytics errors.

    Every domain-specific exception in this package inherits from this
    class, making it possible to catch all application errors with a
    single ``except PredictiveAnalyticsError`` clause.
    """


class ConfigurationError(PredictiveAnalyticsError):
    """Raised when application configuration is invalid or incomplete.

    Examples include missing API keys, invalid model parameter
    combinations, or malformed environment files.
    """


class DataCollectionError(PredictiveAnalyticsError):
    """Raised when data collection from an external source fails.

    This covers network errors, unexpected API response formats, and
    any other failure that prevents raw data from being retrieved.
    """


class APIRateLimitError(DataCollectionError):
    """Raised when an external API signals a rate-limit condition.

    Callers should back off and retry after the period indicated by
    the upstream service.
    """


class DataPreprocessingError(PredictiveAnalyticsError):
    """Raised when data preprocessing encounters an unrecoverable problem.

    Examples include entirely empty DataFrames, missing required
    columns, or numerical operations that produce invalid results.
    """


class DataNotLoadedError(PredictiveAnalyticsError):
    """Raised when an operation requires data that has not been loaded yet.

    Typically occurs when analysis methods are called before the
    underlying DataFrame has been populated.
    """


class ModelTrainingError(PredictiveAnalyticsError):
    """Raised when model training fails.

    Covers convergence failures, invalid parameter combinations, and
    any other error that prevents a model from being fitted.
    """


class ModelNotTrainedError(PredictiveAnalyticsError):
    """Raised when a prediction or evaluation is attempted before training.

    Indicates that the model has not yet been fitted and therefore
    cannot produce forecasts or evaluation metrics.
    """


class ForecastingError(PredictiveAnalyticsError):
    """Raised when forecast generation fails.

    This covers failures during the prediction step, including invalid
    forecast horizons or corrupted model state.
    """


class DisruptionAnalysisError(PredictiveAnalyticsError):
    """Raised when disruption analysis encounters an error.

    Examples include missing required columns in the forecast data or
    invalid threshold parameters.
    """
