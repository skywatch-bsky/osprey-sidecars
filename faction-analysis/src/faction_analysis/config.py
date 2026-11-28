# pattern: Functional Core
"""Configuration parsed from the environment.

``from_mapping`` is pure. ``from_env`` is the only function here that reads the process environment.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Objective = Literal['cpm', 'modularity']
RunKind = Literal['weekly', 'full']

_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')
_MAX_WINDOW_DAYS = 60


class ConfigError(ValueError):
    """An environment value is missing or outside its allowed range."""


def _text(mapping: Mapping[str, str], name: str, default: str) -> str:
    value = mapping.get(name, default).strip()
    if not value:
        raise ConfigError(f'{name} must not be empty')
    return value


def _int(mapping: Mapping[str, str], name: str, default: int, *, minimum: int | None = None) -> int:
    raw = mapping.get(name)
    try:
        value = default if raw is None else int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f'{name} must be an integer: {raw!r}') from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f'{name} must be at least {minimum}: {value}')
    return value


def _float(mapping: Mapping[str, str], name: str, default: float, *, minimum: float | None = None) -> float:
    raw = mapping.get(name)
    try:
        value = default if raw is None else float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f'{name} must be a number: {raw!r}') from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f'{name} must be at least {minimum}: {value}')
    return value


def _bool(mapping: Mapping[str, str], name: str, default: bool) -> bool:
    raw = mapping.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ConfigError(f'{name} must be a boolean (1/0, true/false, yes/no, on/off): {raw!r}')


def _int_list(mapping: Mapping[str, str], name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = mapping.get(name)
    if raw is None:
        return default
    raw = raw.split(',')
    try:
        values = tuple(int(part.strip()) for part in raw)
    except ValueError as exc:
        raise ConfigError(f'{name} must be a comma-separated list of integers: {mapping.get(name)!r}') from exc
    if not values:
        raise ConfigError(f'{name} must list at least one value')
    return values


def _table_name(mapping: Mapping[str, str], name: str, default: str) -> str:
    value = _text(mapping, name, default)
    if not _TABLE_NAME_PATTERN.fullmatch(value):
        raise ConfigError(f'{name} is not a valid table name: {value!r}')
    return value


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> ClickHouseConfig:
        return cls(
            host=_text(mapping, 'OSPREY_CLICKHOUSE_HOST', 'localhost'),
            port=_int(mapping, 'OSPREY_CLICKHOUSE_PORT', 8123, minimum=1),
            user=_text(mapping, 'OSPREY_CLICKHOUSE_USER', 'default'),
            password=mapping.get('OSPREY_CLICKHOUSE_PASSWORD', 'clickhouse'),
            database=_table_name(mapping, 'OSPREY_CLICKHOUSE_DB', 'default'),
        )


@dataclass(frozen=True)
class QueryConfig:
    source_table: str
    table_prefix: str
    max_threads: int
    max_execution_time: int
    max_memory_usage: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> QueryConfig:
        return cls(
            source_table=_table_name(mapping, 'FACTION_ANALYSIS_SOURCE_TABLE', 'osprey_execution_results'),
            table_prefix=_table_name(mapping, 'FACTION_ANALYSIS_TABLE_PREFIX', 'faction_analysis'),
            max_threads=_int(mapping, 'FACTION_ANALYSIS_MAX_THREADS', 4, minimum=1),
            max_execution_time=_int(mapping, 'FACTION_ANALYSIS_MAX_EXECUTION_TIME', 3600, minimum=1),
            max_memory_usage=_int(mapping, 'FACTION_ANALYSIS_MAX_MEMORY_USAGE', 20_000_000_000, minimum=1),
        )

    @property
    def settings(self) -> dict[str, int]:
        return {
            'max_threads': self.max_threads,
            'max_execution_time': self.max_execution_time,
            'max_memory_usage': self.max_memory_usage,
        }

    def table(self, noun: str) -> str:
        return f'{self.table_prefix}_{noun}'


@dataclass(frozen=True)
class GraphConfig:
    window_days: int
    mutual_follow_weight: float
    repost_alpha: float
    damp_large_accounts: bool

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> GraphConfig:
        return cls(
            window_days=_int(mapping, 'FACTION_ANALYSIS_WINDOW_DAYS', 60, minimum=1),
            mutual_follow_weight=_float(mapping, 'FACTION_ANALYSIS_MUTUAL_FOLLOW_WEIGHT', 2.0, minimum=0.0),
            repost_alpha=_float(mapping, 'FACTION_ANALYSIS_REPOST_ALPHA', 0.5, minimum=0.0),
            damp_large_accounts=_bool(mapping, 'FACTION_ANALYSIS_DAMP_LARGE_ACCOUNTS', True),
        )


@dataclass(frozen=True)
class PartitionConfig:
    objective: Objective
    resolution: float
    seed: int
    n_iterations: int
    min_faction_size: int
    split_share: float
    confidence_seeds: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> PartitionConfig:
        # The defaults are the configuration the production measurement chose. See docs/MEASUREMENT.md.
        raw_objective = _text(mapping, 'FACTION_ANALYSIS_OBJECTIVE', 'modularity').lower()
        objective: Objective
        if raw_objective == 'cpm':
            objective = 'cpm'
        elif raw_objective == 'modularity':
            objective = 'modularity'
        else:
            raise ConfigError(f'FACTION_ANALYSIS_OBJECTIVE must be cpm or modularity: {raw_objective!r}')
        return cls(
            objective=objective,
            resolution=_float(mapping, 'FACTION_ANALYSIS_RESOLUTION', 0.5, minimum=0.0),
            seed=_int(mapping, 'FACTION_ANALYSIS_SEED', 42, minimum=0),
            n_iterations=_int(mapping, 'FACTION_ANALYSIS_N_ITERATIONS', 2),
            min_faction_size=_int(mapping, 'FACTION_ANALYSIS_MIN_FACTION_SIZE', 500, minimum=1),
            split_share=_float(mapping, 'FACTION_ANALYSIS_SPLIT_SHARE', 0.1, minimum=0.0),
            confidence_seeds=_int(mapping, 'FACTION_ANALYSIS_CONFIDENCE_SEEDS', 3, minimum=0),
        )


@dataclass(frozen=True)
class ConflictConfig:
    conflict_window_days: tuple[int, ...]
    pileon_min_repliers: int
    pileon_window_minutes: int
    swarm_min_quoters: int
    swarm_window_minutes: int
    cell_prior_strength: float
    min_text_scored: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> ConflictConfig:
        return cls(
            conflict_window_days=_int_list(mapping, 'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS', (7, 60)),
            pileon_min_repliers=_int(mapping, 'FACTION_ANALYSIS_PILEON_MIN_REPLIERS', 5, minimum=1),
            pileon_window_minutes=_int(mapping, 'FACTION_ANALYSIS_PILEON_WINDOW_MINUTES', 60, minimum=1),
            swarm_min_quoters=_int(mapping, 'FACTION_ANALYSIS_SWARM_MIN_QUOTERS', 5, minimum=1),
            swarm_window_minutes=_int(mapping, 'FACTION_ANALYSIS_SWARM_WINDOW_MINUTES', 60, minimum=1),
            cell_prior_strength=_float(mapping, 'FACTION_ANALYSIS_CELL_PRIOR_STRENGTH', 20.0, minimum=0.0),
            min_text_scored=_int(mapping, 'FACTION_ANALYSIS_MIN_TEXT_SCORED', 20, minimum=0),
        )


@dataclass(frozen=True)
class LabelConfig:
    block_positive_minutes: int
    block_negative_horizon_days: int
    jev_stratum_positives: int
    jev_stratum_random: int
    max_weak_positives: int
    negative_sample: int
    weak_positive_weight: float
    negative_weight: float
    jev_max_requests: int
    jev_max_input_tokens: int
    jev_concurrency: int
    jev_max_text_chars: int
    jev_model: str
    prompt_version: str
    min_test_auc: float
    min_test_positives: int
    model_max_age_days: int
    example_min_h: float
    example_sample_modulus: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> LabelConfig:
        return cls(
            block_positive_minutes=_int(mapping, 'FACTION_ANALYSIS_BLOCK_POSITIVE_MINUTES', 60, minimum=1),
            block_negative_horizon_days=_int(mapping, 'FACTION_ANALYSIS_BLOCK_NEGATIVE_HORIZON_DAYS', 7, minimum=1),
            jev_stratum_positives=_int(mapping, 'FACTION_ANALYSIS_JEV_STRATUM_POSITIVES', 100_000, minimum=0),
            jev_stratum_random=_int(mapping, 'FACTION_ANALYSIS_JEV_STRATUM_RANDOM', 100_000, minimum=0),
            max_weak_positives=_int(mapping, 'FACTION_ANALYSIS_MAX_WEAK_POSITIVES', 300_000, minimum=0),
            negative_sample=_int(mapping, 'FACTION_ANALYSIS_NEGATIVE_SAMPLE', 500_000, minimum=0),
            weak_positive_weight=_float(mapping, 'FACTION_ANALYSIS_WEAK_POSITIVE_WEIGHT', 1.0, minimum=0.0),
            negative_weight=_float(mapping, 'FACTION_ANALYSIS_NEGATIVE_WEIGHT', 1.0, minimum=0.0),
            jev_max_requests=_int(mapping, 'FACTION_ANALYSIS_JEV_MAX_REQUESTS', 200_000, minimum=0),
            jev_max_input_tokens=_int(mapping, 'FACTION_ANALYSIS_JEV_MAX_INPUT_TOKENS', 60_000_000, minimum=0),
            jev_concurrency=_int(mapping, 'FACTION_ANALYSIS_JEV_CONCURRENCY', 8, minimum=1),
            jev_max_text_chars=_int(mapping, 'FACTION_ANALYSIS_JEV_MAX_TEXT_CHARS', 2000, minimum=1),
            jev_model=_text(mapping, 'FACTION_ANALYSIS_JEV_MODEL', 'jev-1.14.0'),
            prompt_version=_text(mapping, 'FACTION_ANALYSIS_PROMPT_VERSION', 'hostility-v1'),
            min_test_auc=_float(mapping, 'FACTION_ANALYSIS_MIN_TEST_AUC', 0.70, minimum=0.0),
            min_test_positives=_int(mapping, 'FACTION_ANALYSIS_MIN_TEST_POSITIVES', 200, minimum=0),
            model_max_age_days=_int(mapping, 'FACTION_ANALYSIS_MODEL_MAX_AGE_DAYS', 28, minimum=1),
            example_min_h=_float(mapping, 'FACTION_ANALYSIS_EXAMPLE_MIN_H', 0.5, minimum=0.0),
            example_sample_modulus=_int(mapping, 'FACTION_ANALYSIS_EXAMPLE_SAMPLE_MODULUS', 200, minimum=1),
        )


@dataclass(frozen=True)
class ScheduleConfig:
    poll_interval_seconds: int
    weekly_after_days: int
    monthly_after_days: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> ScheduleConfig:
        return cls(
            poll_interval_seconds=_int(mapping, 'FACTION_ANALYSIS_POLL_INTERVAL_SECONDS', 3600, minimum=1),
            weekly_after_days=_int(mapping, 'FACTION_ANALYSIS_WEEKLY_AFTER_DAYS', 7, minimum=1),
            monthly_after_days=_int(mapping, 'FACTION_ANALYSIS_MONTHLY_AFTER_DAYS', 28, minimum=1),
        )


@dataclass(frozen=True)
class StorageConfig:
    model_dir: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> StorageConfig:
        return cls(model_dir=_text(mapping, 'FACTION_ANALYSIS_MODEL_DIR', '/data/models'))


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool
    service_name: str
    service_version: str
    environment: str
    otlp_endpoint: str | None
    traces_enabled: bool
    metrics_enabled: bool

    @classmethod
    def disabled(cls) -> TelemetryConfig:
        return cls(
            enabled=False,
            service_name='faction-analysis',
            service_version='0.1.0',
            environment='local',
            otlp_endpoint=None,
            traces_enabled=False,
            metrics_enabled=False,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> TelemetryConfig:
        enabled = _bool(mapping, 'FACTION_ANALYSIS_OTEL_ENABLED', False)
        traces_enabled = _bool(mapping, 'FACTION_ANALYSIS_OTEL_TRACES_ENABLED', enabled)
        metrics_enabled = _bool(mapping, 'FACTION_ANALYSIS_OTEL_METRICS_ENABLED', enabled)
        endpoint = mapping.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        return cls(
            enabled=enabled,
            service_name=_text(mapping, 'FACTION_ANALYSIS_OTEL_SERVICE_NAME', 'faction-analysis'),
            service_version=_text(mapping, 'FACTION_ANALYSIS_OTEL_SERVICE_VERSION', '0.1.0'),
            environment=_text(mapping, 'FACTION_ANALYSIS_OTEL_ENVIRONMENT', 'local'),
            otlp_endpoint=None if not endpoint else endpoint,
            traces_enabled=traces_enabled,
            metrics_enabled=metrics_enabled,
        )


@dataclass(frozen=True)
class AppConfig:
    clickhouse: ClickHouseConfig
    query: QueryConfig
    graph: GraphConfig
    partition: PartitionConfig
    conflict: ConflictConfig
    label: LabelConfig
    schedule: ScheduleConfig
    storage: StorageConfig
    telemetry: TelemetryConfig

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> AppConfig:
        config = cls(
            clickhouse=ClickHouseConfig.from_mapping(mapping),
            query=QueryConfig.from_mapping(mapping),
            graph=GraphConfig.from_mapping(mapping),
            partition=PartitionConfig.from_mapping(mapping),
            conflict=ConflictConfig.from_mapping(mapping),
            label=LabelConfig.from_mapping(mapping),
            schedule=ScheduleConfig.from_mapping(mapping),
            storage=StorageConfig.from_mapping(mapping),
            telemetry=TelemetryConfig.from_mapping(mapping),
        )
        config.validate()
        return config

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls.from_mapping(os.environ)

    def validate(self) -> None:
        if self.graph.window_days > _MAX_WINDOW_DAYS:
            raise ConfigError(
                f'FACTION_ANALYSIS_WINDOW_DAYS must not exceed {_MAX_WINDOW_DAYS}: {self.graph.window_days}'
            )
        windows = self.conflict.conflict_window_days
        outside = [days for days in windows if days < 1 or days > self.graph.window_days]
        if outside:
            raise ConfigError(
                'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS must be between 1 and FACTION_ANALYSIS_WINDOW_DAYS '
                f'({self.graph.window_days}): {outside}'
            )
        if len(set(windows)) != len(windows):
            raise ConfigError('FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS must not repeat a value')
