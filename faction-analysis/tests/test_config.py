import pytest

from faction_analysis.config import AppConfig, ConfigError


def test_defaults_load_from_an_empty_mapping() -> None:
    config = AppConfig.from_mapping({})
    assert config.graph.window_days == 60
    assert config.conflict.conflict_window_days == (7, 60)
    assert config.partition.min_faction_size == 500
    assert config.partition.confidence_seeds == 3
    assert config.conflict.cell_prior_strength == 20.0
    assert config.label.jev_max_input_tokens == 60_000_000
    assert config.schedule.monthly_after_days == 28


def test_partition_defaults_are_the_measured_configuration() -> None:
    """The production measurement chose modularity at resolution 0.5 in two iterations (docs/MEASUREMENT.md).

    Resolution 0.5 gives 25 factions on the measured graph, and two seeds agree at an adjusted Rand index of
    0.88. Resolution 1.0 gives 52 factions and its seeds agree at 0.69.
    """
    partition = AppConfig.from_mapping({}).partition
    assert (partition.objective, partition.resolution, partition.n_iterations, partition.seed) == (
        'modularity',
        0.5,
        2,
        42,
    )


def test_window_above_sixty_days_is_rejected() -> None:
    with pytest.raises(ConfigError, match='FACTION_ANALYSIS_WINDOW_DAYS'):
        AppConfig.from_mapping({'FACTION_ANALYSIS_WINDOW_DAYS': '61'})


def test_conflict_window_longer_than_the_window_is_rejected() -> None:
    with pytest.raises(ConfigError, match='FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS'):
        AppConfig.from_mapping({'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS': '7,90'})
    with pytest.raises(ConfigError, match='FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS'):
        AppConfig.from_mapping({'FACTION_ANALYSIS_WINDOW_DAYS': '30', 'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS': '7,60'})


def test_conflict_windows_must_be_positive_and_distinct() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_mapping({'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS': '0,7'})
    with pytest.raises(ConfigError, match='must not repeat'):
        AppConfig.from_mapping({'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS': '7,7'})


def test_shorter_window_with_shorter_conflict_windows_is_accepted() -> None:
    config = AppConfig.from_mapping(
        {'FACTION_ANALYSIS_WINDOW_DAYS': '30', 'FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS': '3,30'}
    )
    assert config.graph.window_days == 30


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('FACTION_ANALYSIS_SEED', 'x'),
        ('FACTION_ANALYSIS_SEED', '-1'),
        ('FACTION_ANALYSIS_REPOST_ALPHA', 'abc'),
        ('FACTION_ANALYSIS_DAMP_LARGE_ACCOUNTS', 'maybe'),
        ('FACTION_ANALYSIS_OBJECTIVE', 'infomap'),
        ('FACTION_ANALYSIS_SOURCE_TABLE', 'osprey_results; DROP TABLE x'),
        ('FACTION_ANALYSIS_TABLE_PREFIX', ''),
        ('FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS', '7,'),
        ('FACTION_ANALYSIS_JEV_CONCURRENCY', '0'),
    ],
)
def test_invalid_values_name_their_variable(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        AppConfig.from_mapping({name: value})


def test_query_settings_and_table_names_come_from_config() -> None:
    config = AppConfig.from_mapping(
        {
            'FACTION_ANALYSIS_MAX_THREADS': '2',
            'FACTION_ANALYSIS_MAX_EXECUTION_TIME': '600',
            'FACTION_ANALYSIS_MAX_MEMORY_USAGE': '1000',
            'FACTION_ANALYSIS_TABLE_PREFIX': 'fa_test',
        }
    )
    assert config.query.settings == {
        'max_threads': 2,
        'max_execution_time': 600,
        'max_memory_usage': 1000,
    }
    assert config.query.table('runs') == 'fa_test_runs'
    assert config.query.table('member_features') == 'fa_test_member_features'


def test_whitespace_around_values_is_ignored() -> None:
    config = AppConfig.from_mapping({'FACTION_ANALYSIS_MIN_FACTION_SIZE': ' 250 '})
    assert config.partition.min_faction_size == 250
