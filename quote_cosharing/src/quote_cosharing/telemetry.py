# pattern: Imperative Shell
"""OpenTelemetry setup and low-cardinality helpers for quote-cosharing."""

from __future__ import annotations

import logging
import threading
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

from quote_cosharing.config import TelemetryConfig

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:  # pragma: no cover - dependency import guard
    OTLPMetricExporter = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]

LOW_CARDINALITY_RUN_ATTRIBUTES = {
    'run_date',
    'pairs_fetched',
    'graph_nodes',
    'graph_edges',
    'cluster_count',
    'membership_rows',
    'had_pairs',
}
LOW_CARDINALITY_METRIC_ATTRIBUTES = {'stage', 'had_pairs', 'error.type'}
ALLOWED_STRING_ATTRIBUTE_VALUES = {
    'stage': {
        'run_cycle',
        'fetch_pairs',
        'build_graph',
        'cluster_graph',
        'fetch_member_timestamps',
        'compute_temporal_metrics',
        'fetch_historical_membership',
        'compute_evolution',
        'delete_stale_run_date',
        'persist_clusters',
        'persist_membership',
    },
}
SHUTDOWN_TIMEOUT_MILLIS = 5_000
logger = logging.getLogger(__name__)
FORBIDDEN_ATTRIBUTE_KEYS = {
    'did',
    'user_id',
    'account',
    'url',
    'domain',
    'quoted_uri',
    'pds_host',
    'cluster_id',
    'rkey',
    'sample_urls',
    'sample_dids',
    'sample_rkeys',
    'shared_uris',
    'table',
    'query',
    'exception.message',
    'exception_message',
    'message',
}


@dataclass(frozen=True)
class TelemetryHandles:
    tracer: Tracer
    meter: Meter
    runs_total: Counter
    runs_failed_total: Counter
    run_duration_seconds: Histogram
    stage_duration_seconds: Histogram
    pairs_fetched: Histogram
    graph_nodes: Histogram
    graph_edges: Histogram
    cluster_count: Histogram
    membership_rows: Histogram
    shutdown_callbacks: tuple[Any, ...] = ()

    def shutdown(self) -> None:
        for callback in self.shutdown_callbacks:
            _run_shutdown_callback(callback)


def _run_shutdown_callback(callback: Any) -> None:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            try:
                callback(timeout_millis=SHUTDOWN_TIMEOUT_MILLIS)
            except TypeError:
                callback()
        except Exception as exc:  # pragma: no cover - defensive shutdown guard
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(SHUTDOWN_TIMEOUT_MILLIS / 1_000)
    if thread.is_alive():
        logger.warning('telemetry shutdown callback timed out')
    for exc in errors:
        logger.warning('telemetry shutdown callback failed', extra={'error_type': type(exc).__name__})


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
        runs_total=meter.create_counter('quote_cosharing.runs_total'),
        runs_failed_total=meter.create_counter('quote_cosharing.runs_failed_total'),
        run_duration_seconds=meter.create_histogram('quote_cosharing.run_duration_seconds'),
        stage_duration_seconds=meter.create_histogram('quote_cosharing.stage_duration_seconds'),
        pairs_fetched=meter.create_histogram('quote_cosharing.pairs_fetched'),
        graph_nodes=meter.create_histogram('quote_cosharing.graph_nodes'),
        graph_edges=meter.create_histogram('quote_cosharing.graph_edges'),
        cluster_count=meter.create_histogram('quote_cosharing.cluster_count'),
        membership_rows=meter.create_histogram('quote_cosharing.membership_rows'),
        shutdown_callbacks=shutdown_callbacks,
    )


def noop_telemetry() -> TelemetryHandles:
    return _make_instruments(trace.get_tracer('quote_cosharing'), metrics.get_meter('quote_cosharing'))


def setup_telemetry(config: TelemetryConfig) -> TelemetryHandles:
    if not config.enabled or (not config.traces_enabled and not config.metrics_enabled):
        return noop_telemetry()

    try:
        return _setup_telemetry(config)
    except Exception as exc:  # pragma: no cover - defensive setup guard
        logger.warning('telemetry setup failed; falling back to no-op', extra={'error_type': type(exc).__name__})
        return noop_telemetry()


def _setup_telemetry(config: TelemetryConfig) -> TelemetryHandles:
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
        tracer = tracer_provider.get_tracer('quote_cosharing')
    else:
        tracer = trace.get_tracer('quote_cosharing')

    if config.metrics_enabled:
        if OTLPMetricExporter is None:
            raise RuntimeError('opentelemetry OTLP metric exporter is unavailable')
        metric_exporter = (
            OTLPMetricExporter(endpoint=config.otlp_endpoint) if config.otlp_endpoint else OTLPMetricExporter()
        )
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        shutdown_callbacks.append(meter_provider.shutdown)
        meter = meter_provider.get_meter('quote_cosharing')
    else:
        meter = metrics.get_meter('quote_cosharing')

    return _make_instruments(tracer, meter, tuple(shutdown_callbacks))


def low_cardinality_attributes(attrs: dict[str, Any], *, include_run_date: bool = True) -> dict[str, Any]:
    allowed = set(LOW_CARDINALITY_METRIC_ATTRIBUTES) | set(LOW_CARDINALITY_RUN_ATTRIBUTES)
    if not include_run_date:
        allowed.discard('run_date')
    filtered: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in FORBIDDEN_ATTRIBUTE_KEYS or key not in allowed or value is None:
            continue
        if isinstance(value, str):
            allowed_values = ALLOWED_STRING_ATTRIBUTE_VALUES.get(key)
            if allowed_values is not None and value not in allowed_values:
                continue
            filtered[key] = value
        elif isinstance(value, (bool, int, float)):
            filtered[key] = value
    return filtered


def set_run_attributes(span: Span, attrs: dict[str, Any]) -> None:
    for key, value in low_cardinality_attributes(attrs, include_run_date=True).items():
        span.set_attribute(key, value)


@contextmanager
def stage_span(handles: TelemetryHandles, name: str, **attrs: Any) -> Iterator[Span]:
    start = time.monotonic()
    span_attrs = low_cardinality_attributes(attrs, include_run_date=True)
    metric_attrs = low_cardinality_attributes({'stage': name.rsplit('.', 1)[-1], **attrs}, include_run_date=False)
    with handles.tracer.start_as_current_span(
        name,
        attributes=span_attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_attribute('error.type', type(exc).__name__)
            raise
        finally:
            handles.stage_duration_seconds.record(time.monotonic() - start, metric_attrs)


def record_run_metrics(
    handles: TelemetryHandles,
    duration_seconds: float,
    *,
    pairs_fetched: int,
    graph_nodes: int,
    graph_edges: int,
    cluster_count: int,
    membership_rows: int,
    had_pairs: bool,
) -> None:
    attrs = low_cardinality_attributes({'had_pairs': had_pairs}, include_run_date=False)
    handles.runs_total.add(1, attrs)
    handles.run_duration_seconds.record(duration_seconds, attrs)
    handles.pairs_fetched.record(pairs_fetched, attrs)
    handles.graph_nodes.record(graph_nodes, attrs)
    handles.graph_edges.record(graph_edges, attrs)
    handles.cluster_count.record(cluster_count, attrs)
    handles.membership_rows.record(membership_rows, attrs)


def record_failure(handles: TelemetryHandles, stage: str, exc: BaseException) -> None:
    attrs = low_cardinality_attributes({'stage': stage, 'error.type': type(exc).__name__}, include_run_date=False)
    handles.runs_failed_total.add(1, attrs)
