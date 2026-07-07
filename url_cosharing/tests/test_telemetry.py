# pattern: Imperative Shell tests
from __future__ import annotations

from datetime import date

from url_cosharing.config import TelemetryConfig
from url_cosharing.db import RunMetadata
from url_cosharing.telemetry import (
    low_cardinality_attributes,
    record_failure,
    record_run_metrics,
    setup_telemetry,
)


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


class FakeHandles:
    def __init__(self) -> None:
        self.runs_total = RecordingCounter()
        self.runs_failed_total = RecordingCounter()
        self.knee_not_found_total = RecordingCounter()
        self.guardrail_triggered_total = RecordingCounter()
        self.run_duration_seconds = RecordingHistogram()
        self.accounts_raw = RecordingHistogram()
        self.accounts_eligible = RecordingHistogram()
        self.urls_eligible = RecordingHistogram()
        self.graph_edges = RecordingHistogram()
        self.flagged_accounts = RecordingHistogram()
        self.cluster_count = RecordingHistogram()


def test_setup_telemetry_disabled_returns_usable_handles() -> None:
    handles = setup_telemetry(TelemetryConfig.disabled())

    handles.runs_total.add(1, {})
    handles.run_duration_seconds.record(0.1, {})
    handles.shutdown()


def test_low_cardinality_attributes_drop_forbidden_keys_and_values() -> None:
    attrs = low_cardinality_attributes(
        {
            'run_date': '2026-07-07',
            'window_days': 7,
            'did': 'did:plc:abc',
            'url': 'https://example.com',
            'domain': 'example.com',
            'cluster_id': '2026-07-07-0001',
            'sample_urls': ('https://example.com',),
            'sample_dids': ('did:plc:abc',),
            'unexpected': 'value',
        }
    )

    assert attrs == {'run_date': '2026-07-07', 'window_days': 7}


def test_metric_attributes_omit_run_date() -> None:
    attrs = low_cardinality_attributes({'run_date': '2026-07-07', 'stage': 'fetch'}, include_run_date=False)

    assert attrs == {'stage': 'fetch'}


def test_record_failure_uses_error_type_not_message() -> None:
    handles = FakeHandles()
    exc = RuntimeError('contains https://example.com and did:plc:abc')

    record_failure(handles, 'fetch', exc)  # type: ignore[arg-type]

    assert handles.runs_failed_total.calls == [(1, {'stage': 'fetch', 'error.type': 'RuntimeError'})]


def test_record_run_metrics_records_low_cardinality_values() -> None:
    handles = FakeHandles()
    run = RunMetadata(
        run_date=date(2026, 7, 7),
        window_days=7,
        accounts_raw=100,
        accounts_eligible=10,
        urls_eligible=20,
        graph_edges=30,
        edge_quantile=0.5,
        centrality_quantile=0.6,
        min_component_density=0.7,
        knee_found=False,
        guardrail_triggered=True,
        flagged_accounts=4,
        cluster_count=1,
    )

    record_run_metrics(handles, run, 1.5)  # type: ignore[arg-type]

    assert handles.runs_total.calls[0][0] == 1
    assert handles.knee_not_found_total.calls == [(1, {'window_days': 7})]
    assert handles.guardrail_triggered_total.calls == [(1, {'window_days': 7})]
    assert handles.accounts_raw.calls[0][0] == 100
    metric_attrs = handles.runs_total.calls[0][1]
    assert metric_attrs == {'window_days': 7, 'knee_found': False, 'guardrail_triggered': True}
