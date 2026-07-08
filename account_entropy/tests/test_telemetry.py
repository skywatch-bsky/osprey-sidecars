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


def test_low_cardinality_attributes_drops_unexpected_string_values() -> None:
    attrs = low_cardinality_attributes({'stage': 'fetch_account_rows', 'error.type': 'ValueError'})
    assert attrs == {'stage': 'fetch_account_rows', 'error.type': 'ValueError'}

    filtered = low_cardinality_attributes({'stage': 'did:plc:secret'})
    assert filtered == {}


def test_setup_telemetry_enabled_falls_back_to_noop_on_setup_failure(monkeypatch) -> None:
    import account_entropy.telemetry as telemetry_module

    def fail_setup(_config):
        raise RuntimeError('private setup failure')

    monkeypatch.setattr(telemetry_module, '_setup_telemetry', fail_setup)
    handles = setup_telemetry(
        TelemetryConfig(
            enabled=True,
            service_name='account-entropy',
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

    def fail_callback(**_kwargs) -> None:
        calls.append('fail')
        raise RuntimeError('private shutdown failure')

    def success_callback(**_kwargs) -> None:
        calls.append('success')

    object.__setattr__(handles, 'shutdown_callbacks', (fail_callback, success_callback))
    handles.shutdown()
    assert calls == ['fail', 'success']
