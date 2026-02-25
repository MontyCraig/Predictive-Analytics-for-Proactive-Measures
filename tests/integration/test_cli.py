"""Integration tests for predictive_analytics.cli module.

Uses typer.testing.CliRunner to test the CLI commands, mocking out
the heavy pipeline stages to keep tests fast and isolated.

Note: The Typer app has a single registered command (``run``), so Typer
promotes it to the top level.  Arguments are passed directly -- do NOT
prefix invocations with the ``run`` subcommand name.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from predictive_analytics.cli import (
    _ensure_directory,
    _resolve_output_dir,
    _run_collection,
    _run_disruptions,
    _run_eda,
    _run_forecasting,
    _run_training,
    app,
)

runner = CliRunner()

# The ``run`` command imports get_config locally, so we patch at the source.
_PATCH_GET_CONFIG = "predictive_analytics.config.settings.get_config"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from rich/typer output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestEnsureDirectory:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "sub"
        result = _ensure_directory(new_dir)
        assert result.exists()
        assert result == new_dir

    def test_existing_directory_is_noop(self, tmp_path: Path) -> None:
        result = _ensure_directory(tmp_path)
        assert result == tmp_path


class TestResolveOutputDir:
    def test_explicit_output_dir(self, tmp_path: Path) -> None:
        explicit = str(tmp_path / "custom")
        result = _resolve_output_dir(explicit, "AAPL")
        assert result == Path(explicit)
        assert result.exists()

    def test_default_output_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = _resolve_output_dir(None, "msft")
        expected = Path("output") / "MSFT"
        assert result == expected
        assert (tmp_path / "output" / "MSFT").exists()


# ---------------------------------------------------------------------------
# CLI --help
# ---------------------------------------------------------------------------


class TestCliHelp:
    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        # Should mention the app or the run command options
        assert "symbol" in clean.lower() or "usage" in clean.lower()

    def test_help_shows_options(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "--api-key" in clean
        assert "--skip-collection" in clean
        assert "--forecast-steps" in clean


# ---------------------------------------------------------------------------
# CLI `run` command -- full pipeline mocked
# ---------------------------------------------------------------------------


class TestRunCommand:
    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_full_pipeline(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "AAPL",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0, f"Unexpected failure: {result.output}"
        mock_collection.assert_called_once()
        mock_eda.assert_called_once()
        mock_training.assert_called_once()
        mock_forecasting.assert_called_once()
        mock_disruptions.assert_called_once()
        clean = _strip_ansi(result.output)
        assert "Pipeline Complete" in clean or "results saved" in clean.lower()

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_skip_all_stages(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "MSFT",
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-eda",
                "--skip-training",
                "--skip-forecasting",
                "--skip-disruptions",
            ],
        )
        assert result.exit_code == 0
        mock_collection.assert_not_called()
        mock_eda.assert_not_called()
        mock_training.assert_not_called()
        mock_forecasting.assert_not_called()
        mock_disruptions.assert_not_called()

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_skip_some_stages(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "TSLA",
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-training",
            ],
        )
        assert result.exit_code == 0
        mock_collection.assert_not_called()
        mock_eda.assert_called_once()
        mock_training.assert_not_called()
        mock_forecasting.assert_called_once()
        mock_disruptions.assert_called_once()

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_custom_forecast_steps(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "GOOG",
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-eda",
                "--skip-training",
                "--skip-disruptions",
                "--forecast-steps",
                "60",
            ],
        )
        assert result.exit_code == 0
        mock_forecasting.assert_called_once()
        # Verify forecast_steps=60 was passed as a positional arg
        call_args = mock_forecasting.call_args[0]
        assert 60 in call_args

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_default_symbol(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When no symbol is provided, 'MSFT' is the default."""
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-eda",
                "--skip-training",
                "--skip-forecasting",
                "--skip-disruptions",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRunCommandErrors:
    @patch(_PATCH_GET_CONFIG)
    def test_generic_exception_exits_with_code_1(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        mock_get_config.side_effect = RuntimeError("config boom")
        result = runner.invoke(app, ["--symbol", "AAPL"])
        assert result.exit_code == 1
        clean = _strip_ansi(result.output).lower()
        assert "config boom" in clean or "error" in clean

    @patch(_PATCH_GET_CONFIG)
    def test_keyboard_interrupt_exits_with_code_130(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        mock_get_config.side_effect = KeyboardInterrupt()
        result = runner.invoke(app, ["--symbol", "AAPL"])
        assert result.exit_code == 130

    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_stage_failure_exits_with_code_1(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        mock_collection.side_effect = Exception("collection failed")
        result = runner.invoke(app, ["--symbol", "AAPL"])
        assert result.exit_code == 1

    @patch(_PATCH_GET_CONFIG)
    def test_missing_api_key_error(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        from predictive_analytics.exceptions import ConfigurationError

        mock_get_config.side_effect = ConfigurationError("API key not found")
        result = runner.invoke(app, ["--symbol", "AAPL"])
        assert result.exit_code == 1
        clean = _strip_ansi(result.output).lower()
        assert "api key" in clean or "error" in clean


# ---------------------------------------------------------------------------
# CLI options / env_file
# ---------------------------------------------------------------------------


class TestRunCommandOptions:
    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_env_file_option(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        env_file = tmp_path / ".env"
        env_file.write_text("ALPHA_VANTAGE_API_KEY=test123\n")
        result = runner.invoke(
            app,
            [
                "--symbol",
                "AAPL",
                "--env-file",
                str(env_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-eda",
                "--skip-training",
                "--skip-forecasting",
                "--skip-disruptions",
            ],
        )
        assert result.exit_code == 0
        # get_config should have been called with the env_file path
        mock_get_config.assert_called_once_with(str(env_file))

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_show_plots_option(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "MSFT",
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-collection",
                "--skip-training",
                "--show-plots",
            ],
        )
        assert result.exit_code == 0
        # EDA should receive show_plots=True as a positional arg
        mock_eda.assert_called_once()
        eda_args = mock_eda.call_args[0]
        assert True in eda_args

    @patch("predictive_analytics.cli._run_disruptions")
    @patch("predictive_analytics.cli._run_forecasting")
    @patch("predictive_analytics.cli._run_training")
    @patch("predictive_analytics.cli._run_eda")
    @patch("predictive_analytics.cli._run_collection")
    @patch(_PATCH_GET_CONFIG)
    def test_api_key_option(
        self,
        mock_get_config: MagicMock,
        mock_collection: MagicMock,
        mock_eda: MagicMock,
        mock_training: MagicMock,
        mock_forecasting: MagicMock,
        mock_disruptions: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_get_config.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "--symbol",
                "AAPL",
                "--api-key",
                "my-test-key",
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-eda",
                "--skip-training",
                "--skip-forecasting",
                "--skip-disruptions",
            ],
        )
        assert result.exit_code == 0
        mock_collection.assert_called_once()
        # The api_key argument should be "my-test-key"
        call_args = mock_collection.call_args[0]
        assert "my-test-key" in call_args


# ---------------------------------------------------------------------------
# Direct tests of the stage runner functions (cover lines 88-211)
# ---------------------------------------------------------------------------


class TestStageRunners:
    """Test the internal _run_* helpers to cover their function bodies."""

    @patch("predictive_analytics.cli.typer")
    @patch("predictive_analytics.cli.console")
    @patch("predictive_analytics.collection.preprocessing.collect_and_preprocess_data")
    def test_run_collection(
        self,
        mock_collect: MagicMock,
        mock_console: MagicMock,
        mock_typer: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = MagicMock()
        _run_collection("AAPL", "fake-key", tmp_path, config)
        mock_collect.assert_called_once_with(
            symbol="AAPL",
            api_key="fake-key",
            output_dir=tmp_path,
            config=config,
        )

    @patch("predictive_analytics.cli.typer")
    @patch("predictive_analytics.cli.console")
    @patch("predictive_analytics.analysis.explorer.TimeSeriesExplorer")
    def test_run_eda(
        self,
        mock_explorer_cls: MagicMock,
        mock_console: MagicMock,
        mock_typer: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_explorer = MagicMock()
        mock_explorer_cls.return_value = mock_explorer
        _run_eda("MSFT", tmp_path, True)
        mock_explorer_cls.assert_called_once_with(
            symbol="MSFT",
            data_dir=tmp_path,
            show_plots=True,
        )
        mock_explorer.run.assert_called_once()

    @patch("predictive_analytics.cli.typer")
    @patch("predictive_analytics.cli.console")
    @patch("predictive_analytics.modeling.trainer.train_and_evaluate_model")
    def test_run_training(
        self,
        mock_train: MagicMock,
        mock_console: MagicMock,
        mock_typer: MagicMock,
        tmp_path: Path,
    ) -> None:
        _run_training("GOOG", tmp_path)
        mock_train.assert_called_once_with(
            symbol="GOOG",
            data_dir=tmp_path,
        )

    @patch("predictive_analytics.cli.typer")
    @patch("predictive_analytics.cli.console")
    @patch("predictive_analytics.modeling.forecaster.generate_forecast")
    def test_run_forecasting(
        self,
        mock_forecast: MagicMock,
        mock_console: MagicMock,
        mock_typer: MagicMock,
        tmp_path: Path,
    ) -> None:
        _run_forecasting("TSLA", tmp_path, 60, False)
        mock_forecast.assert_called_once_with(
            symbol="TSLA",
            data_dir=tmp_path,
            steps=60,
            show_plots=False,
        )

    @patch("predictive_analytics.cli.typer")
    @patch("predictive_analytics.cli.console")
    @patch("predictive_analytics.disruption.analyzer.analyze_disruptions")
    def test_run_disruptions(
        self,
        mock_analyze: MagicMock,
        mock_console: MagicMock,
        mock_typer: MagicMock,
        tmp_path: Path,
    ) -> None:
        _run_disruptions("AMD", tmp_path, True)
        mock_analyze.assert_called_once_with(
            symbol="AMD",
            data_dir=tmp_path,
            show_plots=True,
        )
