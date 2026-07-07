# pattern: Imperative Shell
"""OpenTelemetry setup and low-cardinality helpers for url-cosharing."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer

from url_cosharing.config import TelemetryConfig
from url_cosharing.db import RunMetadata

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:  # pragma: no cover - dependency import guard
    OTLPMetricExporter = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]

LOW_CARDINALITY_RUN_ATTRIBUTES = {
    'run_date',
    'window_days',
    'accounts_raw',
    'accounts_eligible',
    'urls_eligible',
    'graph_edges',
    'knee_found',
    'guardrail_triggered',
    'flagged_accounts',
    'cluster_count',
}
LOW_CARDINALITY_METRIC_ATTRIBUTES = {'stage', 'window_days', 'knee_found', 'guardrail_triggered', 'error.type'}
FORBIDDEN_ATTRIBUTE_KEYS = {'did', 'url', 'domain', 'cluster_id', 'sample_urls', 'sample_dids'}


@dataclass(frozen=True)
class TelemetryHandles:
    tracer: Tracer
    meter: Meter
    runs_total: Counter
    runs_failed_total: Counter
    knee_not_found_total: Counter
    guardrail_triggered_total: Counter
    run_duration_seconds: Histogram
    stage_duration_seconds: Histogram
    accounts_raw: Histogram
    accounts_eligible: Histogram
    urls_eligible: Histogram
    graph_edges: Histogram
    flagged_accounts: Histogram
    cluster_count: Histogram
    shutdown_callbacks: tuple[Any, ...] = ()

    def shutdown(self) -> None:
        for callback in self.shutdown_callbacks:
            callback()


def _resource(config: TelemetryConfig) -> Resource:
    return Resource.create(
        {
            'service.name': config.service_name,
            'service.version': config.service_version,
            'deployment.environment': config.environment,
        }
    )


def _make_instruments(tracer: Tracer, meter: Meter, shutdown_callbacks: tuple[Any, ...] = ()) -> TelemetryHandles:
    return TelemetryHandles(
        tracer=tracer,
        meter=meter,
        runs_total=meter.create_counter('url_cosharing.runs_total'),
        runs_failed_total=meter.create_counter('url_cosharing.runs_failed_total'),
        knee_not_found_total=meter.create_counter('url_cosharing.knee_not_found_total'),
        guardrail_triggered_total=meter.create_counter('url_cosharing.guardrail_triggered_total'),
        run_duration_seconds=meter.create_histogram('url_cosharing.run_duration_seconds'),
        stage_duration_seconds=meter.create_histogram('url_cosharing.stage_duration_seconds'),
        accounts_raw=meter.create_histogram('url_cosharing.accounts_raw'),
        accounts_eligible=meter.create_histogram('url_cosharing.accounts_eligible'),
        urls_eligible=meter.create_histogram('url_cosharing.urls_eligible'),
        graph_edges=meter.create_histogram('url_cosharing.graph_edges'),
        flagged_accounts=meter.create_histogram('url_cosharing.flagged_accounts'),
        cluster_count=meter.create_histogram('url_cosharing.cluster_count'),
        shutdown_callbacks=shutdown_callbacks,
    )


def noop_telemetry() -> TelemetryHandles:
    return _make_instruments(trace.get_tracer('url_cosharing'), metrics.get_meter('url_cosharing'))


def setup_telemetry(config: TelemetryConfig) -> TelemetryHandles:
    if not config.enabled or (not config.traces_enabled and not config.metrics_enabled):
        return noop_telemetry()

    resource = _resource(config)
    shutdown_callbacks: list[Any] = []

    if config.traces_enabled:
        if OTLPSpanExporter is None:
            raise RuntimeError('opentelemetry OTLP span exporter is unavailable')
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint) if config.otlp_endpoint else OTLPSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        shutdown_callbacks.append(tracer_provider.shutdown)
        tracer = tracer_provider.get_tracer('url_cosharing')
    else:
        tracer = trace.get_tracer('url_cosharing')

    if config.metrics_enabled:
        if OTLPMetricExporter is None:
            raise RuntimeError('opentelemetry OTLP metric exporter is unavailable')
        metric_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint) if config.otlp_endpoint else OTLPMetricExporter()
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        shutdown_callbacks.append(meter_provider.shutdown)
        meter = meter_provider.get_meter('url_cosharing')
    else:
        meter = metrics.get_meter('url_cosharing')

    return _make_instruments(tracer, meter, tuple(shutdown_callbacks))


def low_cardinality_attributes(attrs: dict[str, Any], *, include_run_date: bool = True) -> dict[str, Any]:
    allowed = set(LOW_CARDINALITY_METRIC_ATTRIBUTES) | set(LOW_CARDINALITY_RUN_ATTRIBUTES)
    if not include_run_date:
        allowed.discard('run_date')
    filtered: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in FORBIDDEN_ATTRIBUTE_KEYS or key not in allowed or value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            filtered[key] = value
    return filtered


def set_run_attributes(span: Span, attrs: dict[str, Any]) -> None:
    for key, value in low_cardinality_attributes(attrs, include_run_date=True).items():
        span.set_attribute(key, value)


@contextmanager
def stage_span(handles: TelemetryHandles, name: str, **attrs: Any) -> Iterator[Span]:
    start = time.monotonic()
    span_attrs = low_cardinality_attributes(attrs, include_run_date=True)
    metric_attrs = low_cardinality_attributes({'stage': name.rsplit('.', 1)[-1]}, include_run_date=False)
    with handles.tracer.start_as_current_span(name, attributes=span_attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute('error.type', type(exc).__name__)
            raise
        finally:
            handles.stage_duration_seconds.record(time.monotonic() - start, metric_attrs)


def record_run_metrics(handles: TelemetryHandles, run: RunMetadata, duration_seconds: float) -> None:
    metric_attrs = low_cardinality_attributes(
        {
            'window_days': run.window_days,
            'knee_found': run.knee_found,
            'guardrail_triggered': run.guardrail_triggered,
        },
        include_run_date=False,
    )
    handles.runs_total.add(1, metric_attrs)
    handles.run_duration_seconds.record(duration_seconds, metric_attrs)
    if not run.knee_found:
        handles.knee_not_found_total.add(1, {'window_days': run.window_days})
    if run.guardrail_triggered:
        handles.guardrail_triggered_total.add(1, {'window_days': run.window_days})
    handles.accounts_raw.record(run.accounts_raw, metric_attrs)
    handles.accounts_eligible.record(run.accounts_eligible, metric_attrs)
    handles.urls_eligible.record(run.urls_eligible, metric_attrs)
    handles.graph_edges.record(run.graph_edges, metric_attrs)
    handles.flagged_accounts.record(run.flagged_accounts, metric_attrs)
    handles.cluster_count.record(run.cluster_count, metric_attrs)


def record_failure(handles: TelemetryHandles, stage: str, exc: BaseException) -> None:
    attrs = low_cardinality_attributes({'stage': stage, 'error.type': type(exc).__name__}, include_run_date=False)
    handles.runs_failed_total.add(1, attrs)
