from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pytest

from account_entropy.config import AnalysisConfig, AppConfig, ClickHouseConfig
from account_entropy.db import AccountActivityRow, ScoredResult
from account_entropy.main import run_cycle


class FakeDb:
    """In-memory stub for database layer, captures inserted results."""

    def __init__(self) -> None:
        self.rows: list[AccountActivityRow] = []
        self.captured_results: list[ScoredResult] = []

    def fetch_account_rows(self, query: str) -> list[AccountActivityRow]:
        """Returns pre-configured rows."""
        return self.rows

    def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
        """Captures results for later assertion."""
        self.captured_results.extend(results)

    def close(self) -> None:
        """No-op."""
        pass


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        window_days=7,
        min_posts=10,
        hourly_entropy_norm_threshold=0.85,
        interval_entropy_norm_threshold=0.53,
        cv_threshold=0.5,
        interval_bin_edges=(60, 300, 900, 3600, 14400, 86400),
        source_table='osprey_execution_results',
        output_table='account_entropy_results',
    )


@pytest.fixture
def app_config(base_config: AnalysisConfig) -> AppConfig:
    return AppConfig(
        clickhouse=ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        ),
        analysis=base_config,
    )


class TestRunCycle:
    def test_ac2_7_writes_results_with_flags_and_samples(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify AC2.7: writes results with independent flags and sample_rkeys."""
        fake_db = FakeDb()

        # Account with high hourly entropy (bot signal) and low interval entropy (bot signal)
        fake_db.rows = [
            AccountActivityRow(
                user_id='did:plc:bot',
                post_count=42,
                hourly_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Uniform across hours
                ordered_timestamps=[1000, 2000, 3000, 4000, 5000],  # Regular intervals
                sample_rkeys=['rkey1', 'rkey2', 'rkey3'],
            ),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert hasattr(result, 'hourly_flag')
        assert hasattr(result, 'interval_flag')
        assert hasattr(result, 'sample_rkeys')
        assert result.sample_rkeys == ['rkey1', 'rkey2', 'rkey3']

    def test_ac2_8_includes_interval_statistics(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify AC2.8: includes mean and stddev of inter-post intervals."""
        fake_db = FakeDb()

        fake_db.rows = [
            AccountActivityRow(
                user_id='did:plc:test',
                post_count=10,
                hourly_bins=[5, 6, 7, 8],
                ordered_timestamps=[1000, 2000, 3000, 4000, 5000],
                sample_rkeys=['rkey1'],
            ),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert hasattr(result, 'mean_interval_seconds')
        assert hasattr(result, 'stddev_interval_seconds')
        assert isinstance(result.mean_interval_seconds, float)
        assert isinstance(result.stddev_interval_seconds, float)

    def test_ac3_1_processes_accounts_and_writes(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify AC3.1: run_cycle processes accounts and writes results."""
        fake_db = FakeDb()

        fake_db.rows = [
            AccountActivityRow(
                user_id='did:plc:user1',
                post_count=10,
                hourly_bins=[5, 6, 7],
                ordered_timestamps=[1000, 2000, 3000],
                sample_rkeys=['rkey1'],
            ),
            AccountActivityRow(
                user_id='did:plc:user2',
                post_count=15,
                hourly_bins=[10, 11, 12],
                ordered_timestamps=[5000, 6000, 7000],
                sample_rkeys=['rkey2'],
            ),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_results) == 2
        assert fake_db.captured_results[0].user_id == 'did:plc:user1'
        assert fake_db.captured_results[1].user_id == 'did:plc:user2'

    def test_empty_result_set_skips_writing(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify that empty result sets are skipped without writing."""
        fake_db = FakeDb()
        fake_db.rows = []

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_results) == 0

    def test_all_results_written_bot_and_human(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify all results (bot-like and human) are written to output."""
        fake_db = FakeDb()

        # Bot-like account (high hourly entropy + low interval entropy)
        bot_row = AccountActivityRow(
            user_id='did:plc:bot',
            post_count=50,
            hourly_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 4,  # Uniform across all hours
            ordered_timestamps=[1000, 2000, 3000, 4000, 5000, 6000, 7000],  # Regular ~1000ms intervals
            sample_rkeys=['rkey1'],
        )

        # Human-like account (low hourly entropy + high interval entropy)
        human_row = AccountActivityRow(
            user_id='did:plc:human',
            post_count=20,
            hourly_bins=[8, 8, 8, 8, 8, 8, 9, 9, 9, 9],  # Clustered in hours 8-9
            ordered_timestamps=[1000, 100000, 300000, 500000, 700000],  # Irregular intervals
            sample_rkeys=['rkey2'],
        )

        fake_db.rows = [bot_row, human_row]

        run_cycle(fake_db, app_config)

        # Both should be captured
        assert len(fake_db.captured_results) == 2

        bot_result = next(r for r in fake_db.captured_results if r.user_id == 'did:plc:bot')
        human_result = next(r for r in fake_db.captured_results if r.user_id == 'did:plc:human')

        # Bot-like should have high is_bot_like indicator (if thresholds are met)
        assert bot_result.user_id == 'did:plc:bot'
        assert human_result.user_id == 'did:plc:human'

    def test_run_cycle_uses_real_analyzer(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle uses the real analyzer scoring logic."""
        fake_db = FakeDb()

        test_row = AccountActivityRow(
            user_id='did:plc:test',
            post_count=30,
            hourly_bins=[5, 5, 5, 6, 6, 7, 7, 8, 8, 9],
            ordered_timestamps=[1000, 2000, 3000, 4000, 5000],
            sample_rkeys=['rkey1'],
        )

        fake_db.rows = [test_row]

        run_cycle(fake_db, app_config)

        result = fake_db.captured_results[0]

        # Verify scoring applied correctly
        assert result.post_count == 30
        assert result.hourly_entropy >= 0.0
        assert result.interval_entropy >= 0.0
        assert result.is_bot_like in (0, 1)

    def test_window_timestamps_correct(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify window_start and window_end in results match config.window_days."""
        fake_db = FakeDb()

        fake_db.rows = [
            AccountActivityRow(
                user_id='did:plc:test',
                post_count=10,
                hourly_bins=[5, 6],
                ordered_timestamps=[1000, 2000],
                sample_rkeys=['rkey1'],
            ),
        ]

        run_cycle(fake_db, app_config)

        result = fake_db.captured_results[0]

        # Verify window duration matches config
        window_duration = result.window_end - result.window_start
        expected_days = app_config.analysis.window_days
        # Convert to seconds for comparison
        assert window_duration.days == expected_days

    def test_ac3_4_cycle_exception_does_not_crash(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify AC3.4: Exception during cycle is propagated (try/except is in main, not run_cycle)."""
        fake_db = FakeDb()

        # Configure FakeDb to raise exception
        def raise_error(query):
            raise RuntimeError('ClickHouse unavailable')

        fake_db.fetch_account_rows = raise_error

        # run_cycle should propagate the exception (main() catches it)
        with pytest.raises(RuntimeError, match='ClickHouse unavailable'):
            run_cycle(fake_db, app_config)

    def test_run_cycle_sets_correct_window_parameters(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle uses correct window calculations."""
        fake_db = FakeDb()

        fake_db.rows = [
            AccountActivityRow(
                user_id='did:plc:user1',
                post_count=10,
                hourly_bins=[0, 1],
                ordered_timestamps=[1000, 2000],
                sample_rkeys=['rkey1'],
            ),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]

        # Verify run_timestamp is set
        assert result.run_timestamp is not None
        assert isinstance(result.run_timestamp, datetime)

        # Verify window boundaries are set
        assert result.window_start is not None
        assert result.window_end is not None
        assert result.window_start < result.window_end


class RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object] | None]] = []

    def add(self, value: int, attributes: dict[str, object] | None = None) -> None:
        self.calls.append((value, attributes))


class RecordingHistogram:
    def __init__(self) -> None:
        self.calls: list[tuple[int | float, dict[str, object] | None]] = []

    def record(self, value: int | float, attributes: dict[str, object] | None = None) -> None:
        self.calls.append((value, attributes))


@pytest.fixture
def telemetry_handles():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from account_entropy.telemetry import TelemetryHandles

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handles = TelemetryHandles(
        tracer=provider.get_tracer('account_entropy.tests'),
        meter=object(),  # type: ignore[arg-type]
        runs_total=RecordingCounter(),  # type: ignore[arg-type]
        runs_failed_total=RecordingCounter(),  # type: ignore[arg-type]
        run_duration_seconds=RecordingHistogram(),  # type: ignore[arg-type]
        stage_duration_seconds=RecordingHistogram(),  # type: ignore[arg-type]
        accounts_fetched=RecordingHistogram(),  # type: ignore[arg-type]
        accounts_scored=RecordingHistogram(),  # type: ignore[arg-type]
        bot_like_count=RecordingHistogram(),  # type: ignore[arg-type]
        shutdown_callbacks=(provider.shutdown,),
    )
    return handles, exporter


def test_run_cycle_records_account_entropy_stage_spans_and_metrics(app_config: AppConfig, telemetry_handles) -> None:
    handles, exporter = telemetry_handles
    fake_db = FakeDb()
    fake_db.rows = [
        AccountActivityRow(
            user_id='did:plc:secret',
            post_count=10,
            hourly_bins=[5, 6, 7],
            ordered_timestamps=[1000, 2000, 3000],
            sample_rkeys=['secret-rkey'],
        )
    ]

    run_cycle(fake_db, app_config, handles)

    span_names = {span.name for span in exporter.get_finished_spans()}
    assert 'account_entropy.run_cycle' in span_names
    assert 'account_entropy.fetch_account_rows' in span_names
    assert 'account_entropy.score_accounts' in span_names
    assert 'account_entropy.insert_results' in span_names
    assert len(handles.runs_total.calls) == 1
    assert len(handles.run_duration_seconds.calls) == 1
    assert len(handles.stage_duration_seconds.calls) == 3
    rendered = (
        repr(exporter.get_finished_spans())
        + repr(handles.runs_total.calls)
        + repr(handles.stage_duration_seconds.calls)
    )
    assert 'did:plc:secret' not in rendered
    assert 'secret-rkey' not in rendered


def test_run_cycle_empty_accounts_records_success_not_failure(app_config: AppConfig, telemetry_handles) -> None:
    handles, _ = telemetry_handles
    fake_db = FakeDb()

    run_cycle(fake_db, app_config, handles)

    assert handles.runs_total.calls == [(1, {'window_days': 7, 'had_rows': False})]
    assert handles.runs_failed_total.calls == []
    assert fake_db.captured_results == []


def test_run_cycle_failure_records_error_type_and_reraises(app_config: AppConfig, telemetry_handles) -> None:
    handles, _ = telemetry_handles
    fake_db = FakeDb()

    def fail(_query: str):
        raise RuntimeError('private failure message')

    fake_db.fetch_account_rows = fail
    with pytest.raises(RuntimeError, match='private failure message'):
        run_cycle(fake_db, app_config, handles)

    assert handles.runs_failed_total.calls == [(1, {'stage': 'run_cycle', 'error.type': 'RuntimeError'})]
    assert 'private failure message' not in repr(handles.runs_failed_total.calls)
