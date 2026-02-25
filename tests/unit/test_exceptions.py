"""Tests for the predictive_analytics.exceptions module.

Verifies that every custom exception:
- Can be instantiated with a message.
- Stores the message correctly (via ``args`` / ``str``).
- Follows the intended inheritance hierarchy.
"""

from __future__ import annotations

import pytest

from predictive_analytics.exceptions import (
    APIRateLimitError,
    ConfigurationError,
    DataCollectionError,
    DataNotLoadedError,
    DataPreprocessingError,
    DisruptionAnalysisError,
    ForecastingError,
    ModelNotTrainedError,
    ModelTrainingError,
    PredictiveAnalyticsError,
)


# ---------------------------------------------------------------------------
# Inheritance checks
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify that the inheritance tree is wired correctly."""

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(PredictiveAnalyticsError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            DataCollectionError,
            DataPreprocessingError,
            DataNotLoadedError,
            ModelTrainingError,
            ModelNotTrainedError,
            ForecastingError,
            DisruptionAnalysisError,
        ],
    )
    def test_direct_children_inherit_from_base(
        self,
        exc_cls: type[PredictiveAnalyticsError],
    ) -> None:
        assert issubclass(exc_cls, PredictiveAnalyticsError)

    def test_api_rate_limit_inherits_from_data_collection(self) -> None:
        assert issubclass(APIRateLimitError, DataCollectionError)

    def test_api_rate_limit_inherits_from_base(self) -> None:
        assert issubclass(APIRateLimitError, PredictiveAnalyticsError)


# ---------------------------------------------------------------------------
# Instantiation and message tests
# ---------------------------------------------------------------------------


class TestExceptionInstantiation:
    """Ensure every exception can be raised and carries its message."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            PredictiveAnalyticsError,
            ConfigurationError,
            DataCollectionError,
            APIRateLimitError,
            DataPreprocessingError,
            DataNotLoadedError,
            ModelTrainingError,
            ModelNotTrainedError,
            ForecastingError,
            DisruptionAnalysisError,
        ],
    )
    def test_instantiation_with_message(
        self,
        exc_cls: type[PredictiveAnalyticsError],
    ) -> None:
        msg = f"Test message for {exc_cls.__name__}"
        exc = exc_cls(msg)
        assert str(exc) == msg
        assert exc.args == (msg,)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            PredictiveAnalyticsError,
            ConfigurationError,
            DataCollectionError,
            APIRateLimitError,
            DataPreprocessingError,
            DataNotLoadedError,
            ModelTrainingError,
            ModelNotTrainedError,
            ForecastingError,
            DisruptionAnalysisError,
        ],
    )
    def test_exception_can_be_raised_and_caught(
        self,
        exc_cls: type[PredictiveAnalyticsError],
    ) -> None:
        with pytest.raises(exc_cls, match="boom"):
            raise exc_cls("boom")

    def test_catch_data_collection_catches_rate_limit(self) -> None:
        """APIRateLimitError should be catchable as DataCollectionError."""
        with pytest.raises(DataCollectionError):
            raise APIRateLimitError("rate limited")

    def test_catch_base_catches_all(self) -> None:
        """All domain exceptions should be catchable as the base."""
        for exc_cls in (
            ConfigurationError,
            DataCollectionError,
            APIRateLimitError,
            DataPreprocessingError,
            DataNotLoadedError,
            ModelTrainingError,
            ModelNotTrainedError,
            ForecastingError,
            DisruptionAnalysisError,
        ):
            with pytest.raises(PredictiveAnalyticsError):
                raise exc_cls("test")

    @pytest.mark.parametrize(
        "exc_cls",
        [
            PredictiveAnalyticsError,
            ConfigurationError,
            DataCollectionError,
            APIRateLimitError,
            DataPreprocessingError,
            DataNotLoadedError,
            ModelTrainingError,
            ModelNotTrainedError,
            ForecastingError,
            DisruptionAnalysisError,
        ],
    )
    def test_instantiation_without_message(
        self,
        exc_cls: type[PredictiveAnalyticsError],
    ) -> None:
        exc = exc_cls()
        assert exc.args == ()


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify that __all__ contains the expected names."""

    def test_all_contains_expected_exceptions(self) -> None:
        from predictive_analytics import exceptions

        expected = {
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
        }
        assert set(exceptions.__all__) == expected
