from datetime import UTC, datetime
from pathlib import Path

import httpx2
import pandas as pd
import pytest
from typesafe_sdk import TypeSafeAuthenticationError

from faction_analysis import model_store, train
from faction_analysis.config import AppConfig
from faction_analysis.jev import JevLabel
from tests.training_world import NOW_MS, FakeJev, FakeTrainingRepo, empty_labels, world

QUOTAS = {
    'FACTION_ANALYSIS_JEV_STRATUM_POSITIVES': '1000',
    'FACTION_ANALYSIS_JEV_STRATUM_RANDOM': '1000',
    'FACTION_ANALYSIS_MAX_WEAK_POSITIVES': '500',
    'FACTION_ANALYSIS_NEGATIVE_SAMPLE': '1000',
    'FACTION_ANALYSIS_MIN_TEST_POSITIVES': '100',
}
NOW = datetime.fromtimestamp(NOW_MS / 1000, tz=UTC)


def config() -> AppConfig:
    return AppConfig.from_mapping(QUOTAS)


def refresh(repo: FakeTrainingRepo, model_dir: Path, **options: object) -> train.RefreshOutcome:
    return train.refresh(repo, config(), model_dir, now=NOW, **options)  # ty: ignore[invalid-argument-type]


def labels_of(repo: FakeTrainingRepo) -> pd.DataFrame:
    """Every label frame a repository received, in order. A refresh stores its answers in batches."""
    return pd.concat(repo.inserted['jev_labels'], ignore_index=True)


def test_refresh_labels_the_sample_and_promotes_a_model(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    jev = FakeJev()
    outcome = refresh(fake_world, tmp_path, factory=lambda: jev)
    assert outcome.status == 'promoted', outcome.reason
    assert outcome.labelled_now == 2000
    assert len(jev.calls) == 2000
    assert model_store.current_manifest(tmp_path) is not None

    labels = labels_of(fake_world)
    assert len(labels) == 2000
    assert set(labels['h'].round(2).tolist()) == {0.9, 0.05}
    assert (labels['model'] == 'fake').all()
    assert labels['uri_hash'].nunique() == 2000
    # The answers reach the cache in batches while the labelling runs, not in one write at the end.
    assert [len(frame) for frame in fake_world.inserted['jev_labels']] == [500, 500, 500, 500]

    dataset = fake_world.inserted['scorer_dataset'][0]
    assert dataset['dataset_version'].unique().tolist() == [outcome.dataset_version]
    # The labelled rows plus the weak positives and the random negatives.
    assert len(dataset) > 2000
    assert set(dataset['label_source'].tolist()) == {'jev', 'block', 'random'}
    assert (dataset['parent_uri'] != '').all()
    assert (dataset['text'].str.len() > 0).all()
    assert outcome.metrics is not None and outcome.metrics['test_auc'] is not None
    assert outcome.metrics['test_auc'] > 0.7


def test_jev_labels_are_reused_not_repurchased(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    refresh(fake_world, tmp_path, factory=lambda: FakeJev())
    stored = set(labels_of(fake_world)['uri_hash'].tolist())
    assert len(stored) == 2000

    # The next refresh labels pairs from the rest of the pool. None of them is one it already paid for.
    fake_world.existing = labels_of(fake_world)[['uri_hash', 'h']].sample(frac=1, random_state=3)
    fake_world.inserted.clear()
    jev = FakeJev()
    second = refresh(fake_world, tmp_path, factory=lambda: jev)
    bought_again = set(labels_of(fake_world)['uri_hash'].tolist()) & stored
    assert bought_again == set()
    assert second.labelled_now == len(labels_of(fake_world))

    # Every interaction now has a label. A third refresh buys nothing.
    fake_world.existing = pd.concat([fake_world.existing, labels_of(fake_world)[['uri_hash', 'h']]], ignore_index=True)
    fake_world.inserted.clear()
    third = refresh(fake_world, tmp_path, factory=lambda: FakeJev())
    assert third.labelled_now == 0
    assert 'jev_labels' not in fake_world.inserted
    assert third.status == 'promoted'


def test_answers_paid_before_a_failure_are_kept_and_not_bought_again(
    tmp_path: Path, fake_world: FakeTrainingRepo
) -> None:
    """The key stops working in the middle of the sample. The answers before that are in the cache."""
    calls = []

    class Failing(FakeJev):
        async def label(self, pair: object) -> JevLabel:
            if len(self.calls) >= 1500:
                raise TypeSafeAuthenticationError(401, {}, httpx2.Headers())
            return await super().label(pair)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeSafeAuthenticationError):
        refresh(fake_world, tmp_path, factory=lambda: Failing(calls))
    stored = labels_of(fake_world)
    assert len(stored) == 1500 and stored['uri_hash'].nunique() == 1500

    # The next refresh pays only for the pairs that have no label: the 500 positives that were not reached
    # and 1000 of the random stratum.
    fake_world.existing = stored[['uri_hash', 'h']]
    fake_world.inserted.clear()
    outcome = refresh(fake_world, tmp_path, factory=lambda: FakeJev())
    new = labels_of(fake_world)
    assert len(new) == 1500 and not set(new['uri_hash']) & set(stored['uri_hash'])
    assert outcome.labelled_now == 1500


def test_a_client_that_fails_to_close_does_not_hide_why_labelling_ended(tmp_path: Path) -> None:
    class Unclosable(FakeJev):
        async def label(self, pair: object) -> JevLabel:
            raise TypeSafeAuthenticationError(401, {}, httpx2.Headers())

        async def close(self) -> None:
            raise OSError('the socket is gone')

    fake_world = FakeTrainingRepo(*world(count=50), empty_labels())
    with pytest.raises(TypeSafeAuthenticationError) as lost:
        refresh(fake_world, tmp_path, factory=lambda: Unclosable())
    assert lost.value.__notes__ == ["the labeller could not be closed: OSError('the socket is gone')"]

    class Unlucky(FakeJev):
        async def close(self) -> None:
            raise OSError('the socket is gone')

    fake_world = FakeTrainingRepo(*world(count=50), empty_labels())
    with pytest.raises(OSError, match='socket'):
        refresh(fake_world, tmp_path, factory=lambda: Unlucky())


def test_without_a_labeller_the_weak_rules_train_the_model(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    outcome = refresh(fake_world, tmp_path, factory=None)
    # Without Jev labels there is no test split, so the model cannot be promoted. It is still trained and
    # reported, and the weak rows are written.
    assert outcome.status == 'kept'
    assert outcome.reason == 'the test split is empty'
    assert 'jev_labels' not in fake_world.inserted
    dataset = fake_world.inserted['scorer_dataset'][0]
    assert set(dataset['label_source'].tolist()) == {'block', 'random'}
    assert dataset['label_h'].isna().all()


def test_dry_run_writes_nothing(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    outcome = refresh(fake_world, tmp_path, factory=lambda: FakeJev(), dry_run=True)
    assert fake_world.inserted == {}
    assert model_store.current_manifest(tmp_path) is None
    assert outcome.status in {'promoted', 'kept'}


def test_a_model_below_the_gate_is_kept_back_with_its_reason(tmp_path: Path, fake_world: FakeTrainingRepo) -> None:
    class RandomJev(FakeJev):
        async def label(self, pair: object) -> JevLabel:
            return JevLabel(0.5, 0.25, 0.25, 'random', 10)

    outcome = refresh(fake_world, tmp_path, factory=lambda: RandomJev())
    assert outcome.status == 'kept'
    assert 'below' in outcome.reason or 'one class' in outcome.reason
    assert model_store.current_manifest(tmp_path) is None


def test_a_window_with_too_few_rows_makes_no_model(tmp_path: Path) -> None:
    inputs, texts, parents = world(count=2)
    small = FakeTrainingRepo(inputs, texts, parents, empty_labels())
    outcome = refresh(small, tmp_path, factory=lambda: FakeJev())
    assert outcome.status == 'no_model'
    assert 'rows' in outcome.reason
    assert model_store.current_manifest(tmp_path) is None
