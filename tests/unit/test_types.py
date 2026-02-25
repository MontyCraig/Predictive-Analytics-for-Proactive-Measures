"""Tests for the predictive_analytics.types module.

Verifies that all public type aliases exist, are importable, and have the
expected structure.
"""

from __future__ import annotations

from typing import get_args, get_origin


class TestTypeAliases:
    """Ensure every public type alias is importable and well-formed."""

    def test_sarima_order_alias_exists(self) -> None:
        from predictive_analytics.types import SARIMAOrder

        # SARIMAOrder should be tuple[int, int, int]
        assert get_origin(SARIMAOrder) is tuple
        args = get_args(SARIMAOrder)
        assert args == (int, int, int)

    def test_sarima_seasonal_order_alias_exists(self) -> None:
        from predictive_analytics.types import SARIMASeasonalOrder

        assert get_origin(SARIMASeasonalOrder) is tuple
        args = get_args(SARIMASeasonalOrder)
        assert args == (int, int, int, int)

    def test_metrics_dict_alias_exists(self) -> None:
        from predictive_analytics.types import MetricsDict

        assert get_origin(MetricsDict) is dict
        args = get_args(MetricsDict)
        assert args == (str, float)

    def test_disruption_report_alias_exists(self) -> None:
        from predictive_analytics.types import DisruptionReport

        assert get_origin(DisruptionReport) is dict
        args = get_args(DisruptionReport)
        assert args[0] is str
        # The value type is int | float | list[str]

    def test_sarima_order_is_valid_type_annotation(self) -> None:
        """Confirm the alias can be used as a type annotation at runtime."""
        from predictive_analytics.types import SARIMAOrder

        val: SARIMAOrder = (1, 1, 1)
        assert isinstance(val, tuple)
        assert len(val) == 3

    def test_sarima_seasonal_order_is_valid_type_annotation(self) -> None:
        from predictive_analytics.types import SARIMASeasonalOrder

        val: SARIMASeasonalOrder = (1, 1, 1, 12)
        assert isinstance(val, tuple)
        assert len(val) == 4

    def test_metrics_dict_is_valid_type_annotation(self) -> None:
        from predictive_analytics.types import MetricsDict

        val: MetricsDict = {"rmse": 0.5, "mae": 0.3}
        assert isinstance(val, dict)

    def test_disruption_report_is_valid_type_annotation(self) -> None:
        from predictive_analytics.types import DisruptionReport

        val: DisruptionReport = {
            "count": 3,
            "severity": 0.8,
            "dates": ["2024-01-01", "2024-02-01"],
        }
        assert isinstance(val, dict)
