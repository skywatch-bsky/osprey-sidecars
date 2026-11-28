import pytest

from faction_analysis.config import TelemetryConfig
from faction_analysis.telemetry import (
    ALLOWED_STRING_ATTRIBUTE_VALUES,
    FORBIDDEN_ATTRIBUTE_KEYS,
    LOW_CARDINALITY_METRIC_ATTRIBUTES,
    LOW_CARDINALITY_RUN_ATTRIBUTES,
    low_cardinality_attributes,
    record_failure,
    setup_telemetry,
)


class RecordingCounter:
    def __init__(self) -> None:
        self.calls = []

    def add(self, value, attributes=None) -> None:
        self.calls.append((value, attributes or {}))


def test_attribute_keys_are_fixed() -> None:
    assert LOW_CARDINALITY_METRIC_ATTRIBUTES == frozenset({'stage', 'run_kind', 'had_rows', 'error.type'})
    assert LOW_CARDINALITY_RUN_ATTRIBUTES == frozenset(
        {'run_kind', 'window_days', 'accounts_placed', 'accounts_unplaced', 'factions', 'moves', 'had_rows'}
    )
    assert {'did', 'handle', 'faction_id', 'uri', 'query', 'message', 'run_id'} <= FORBIDDEN_ATTRIBUTE_KEYS
    assert set(ALLOWED_STRING_ATTRIBUTE_VALUES) == {'stage', 'run_kind'}
    assert LOW_CARDINALITY_METRIC_ATTRIBUTES.isdisjoint(FORBIDDEN_ATTRIBUTE_KEYS)
    assert LOW_CARDINALITY_RUN_ATTRIBUTES.isdisjoint(FORBIDDEN_ATTRIBUTE_KEYS)


def test_setup_telemetry_disabled_returns_usable_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    handles = setup_telemetry(TelemetryConfig.disabled())
    handles.runs_total.add(1, {})
    handles.run_duration_seconds.record(0.1, {})
    handles.shutdown()


def test_low_cardinality_attributes_filters_forbidden_keys() -> None:
    attrs = low_cardinality_attributes(
        {
            'had_rows': True,
            'factions': 12,
            'did': 'SECRET-did',
            'handle': 'SECRET-handle',
            'faction_id': 'SECRET-faction_id',
            'uri': 'SECRET-uri',
            'exception.message': 'secret message',
            'query': 'select secret',
            'run_id': 'SECRET-run-id',
        }
    )
    assert attrs == {'had_rows': True, 'factions': 12}
    rendered = repr(attrs)
    for secret in ['SECRET-', 'secret message', 'select secret']:
        assert secret not in rendered


def test_record_failure_uses_error_type_not_exception_message() -> None:
    handles = setup_telemetry(TelemetryConfig.disabled())
    counter = RecordingCounter()
    object.__setattr__(handles, 'runs_failed_total', counter)
    exc = RuntimeError('this exception message is private')
    record_failure(handles, 'partition', exc)
    assert counter.calls == [(1, {'stage': 'partition', 'error.type': 'RuntimeError'})]
    assert 'private' not in repr(counter.calls)


def test_low_cardinality_attributes_drops_unexpected_string_values() -> None:
    attrs = low_cardinality_attributes({'stage': 'partition', 'error.type': 'ValueError'})
    assert attrs == {'stage': 'partition', 'error.type': 'ValueError'}

    filtered = low_cardinality_attributes({'stage': 'did:plc:secret', 'run_kind': 'did:plc:secret'})
    assert filtered == {}


def test_setup_telemetry_enabled_falls_back_to_noop_on_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import faction_analysis.telemetry as telemetry_module

    def fail_setup(_config: TelemetryConfig) -> None:
        raise RuntimeError('private setup failure')

    monkeypatch.setattr(telemetry_module, '_setup_telemetry', fail_setup)
    handles = setup_telemetry(
        TelemetryConfig(
            enabled=True,
            service_name='faction-analysis',
            service_version='0.1.0',
            environment='test',
            otlp_endpoint=None,
            traces_enabled=True,
            metrics_enabled=True,
        )
    )
    handles.runs_total.add(1, {})
    handles.shutdown()


def test_shutdown_runs_remaining_callbacks_when_one_fails() -> None:
    handles = setup_telemetry(TelemetryConfig.disabled())
    calls: list[str] = []

    def fail_callback(**_kwargs: object) -> None:
        calls.append('fail')
        raise RuntimeError('private shutdown failure')

    def success_callback(**_kwargs: object) -> None:
        calls.append('success')

    object.__setattr__(handles, 'shutdown_callbacks', (fail_callback, success_callback))
    handles.shutdown()
    assert calls == ['fail', 'success']
