from __future__ import annotations

from account_entropy.config import TelemetryConfig
from account_entropy.telemetry import low_cardinality_attributes, record_failure, setup_telemetry


class RecordingCounter:
    def __init__(self) -> None:
        self.calls = []

    def add(self, value, attributes=None) -> None:
        self.calls.append((value, attributes or {}))


def test_setup_telemetry_disabled_returns_usable_handles(monkeypatch) -> None:
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    handles = setup_telemetry(TelemetryConfig.disabled())
    handles.runs_total.add(1, {})
    handles.run_duration_seconds.record(0.1, {})
    handles.shutdown()


def test_low_cardinality_attributes_filters_forbidden_keys() -> None:
    attrs = low_cardinality_attributes(
        {
            'window_days': 7,
            'user_id': 'SECRET-user_id',
            'sample_rkeys': 'SECRET-sample_rkeys',
            'rkey': 'SECRET-rkey',
            'exception.message': 'secret message',
            'query': 'select secret',
        }
    )
    assert attrs == {'window_days': 7}
    rendered = repr(attrs)
    for secret in ['SECRET-', 'secret message', 'select secret']:
        assert secret not in rendered


def test_record_failure_uses_error_type_not_exception_message() -> None:
    handles = setup_telemetry(TelemetryConfig.disabled())
    counter = RecordingCounter()
    object.__setattr__(handles, 'runs_failed_total', counter)
    exc = RuntimeError('this exception message is private')
    record_failure(handles, 'run_cycle', exc)
    assert counter.calls == [(1, {'stage': 'run_cycle', 'error.type': 'RuntimeError'})]
    assert 'private' not in repr(counter.calls)
