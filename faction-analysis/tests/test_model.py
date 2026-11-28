from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from faction_analysis.labels import BLOCK, CALIBRATION, JEV, RANDOM, TRAIN
from faction_analysis.model import (
    N_FEATURES,
    TrainingError,
    TrainingRows,
    evaluate,
    featurize,
    may_promote,
    structural_matrix,
    text_frame,
    train,
)


def test_structural_matrix_matches_a_hand_computation() -> None:
    frame = text_frame(
        ['HEY!! @a @b'],
        np.asarray([7]),
        np.asarray([3]),
        np.asarray([2]),
        np.asarray([0]),
        np.asarray([2]),
        np.asarray([1]),
    )
    assert structural_matrix(frame)[0].tolist() == pytest.approx([np.log1p(7), 2, 0, 3 / 7, 2, 1])
    zero = np.asarray([0])
    empty = text_frame([''], zero, zero, zero, zero, zero, np.asarray([0]))
    assert structural_matrix(empty)[0].tolist() == pytest.approx([0, 0, 0, 0, 0, 0])


def test_featurize_stacks_ngrams_and_the_structural_features(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated(jev_rows=12, weak_rows=0, random_rows=0)
    matrix = featurize(rows.frame)
    assert matrix.shape == (12, N_FEATURES + 6)
    assert np.allclose(matrix[:, N_FEATURES:].toarray(), structural_matrix(rows.frame))


def test_trained_model_returns_finite_probabilities_in_the_unit_interval(
    fabricated: Callable[..., TrainingRows],
) -> None:
    rows = fabricated()
    probability = train(rows, seed=1).probabilities(rows.frame)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_the_model_separates_hostile_from_neutral_text(
    fabricated: Callable[..., TrainingRows], helpers: ModuleType
) -> None:
    model = train(fabricated(), seed=1)
    texts: list[str] = helpers.HOSTILE + helpers.NEUTRAL
    frame = text_frame(texts, kind=np.zeros(10, dtype=np.int8), **helpers.structural(texts))
    probability = model.probabilities(frame)
    assert probability[:5].mean() > 0.7
    assert probability[5:].mean() < 0.3


def test_calibration_uses_the_calibration_split(fabricated: Callable[..., TrainingRows]) -> None:
    assert train(fabricated(), seed=1).calibrated is True
    rows = fabricated()
    without = TrainingRows(
        frame=rows.frame,
        target=rows.target,
        weight=rows.weight,
        split=np.where(rows.split == CALIBRATION, TRAIN, rows.split),
        label_source=rows.label_source,
        languages=rows.languages,
    )
    assert train(without, seed=1).calibrated is False


def test_metrics_and_the_block_agreement_are_reported(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated()
    metrics = evaluate(train(rows, seed=1), rows)
    assert 0 < metrics.test_positives < metrics.test_rows
    assert metrics.test_auc is not None and metrics.test_auc > 0.8
    assert metrics.test_average_precision is not None and metrics.test_brier is not None
    assert metrics.block_auc is not None and metrics.block_auc > 0.5
    assert set(metrics.per_language_ap) == {'en', 'fr', 'de'}
    assert metrics.per_language_ap['en'] is not None
    assert metrics.per_language_ap['de'] is None


def test_a_metric_on_a_split_with_one_class_is_null(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated(test_all_positive=True)
    metrics = evaluate(train(rows, seed=1), rows)
    assert metrics.test_positives == metrics.test_rows
    assert metrics.test_auc is None
    assert metrics.test_average_precision is None
    assert metrics.test_brier is None


def test_one_class_test_split_refuses_promotion(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated(test_all_positive=True)
    metrics = evaluate(train(rows, seed=1), rows)
    promoted, reason = may_promote(metrics, min_test_auc=0.5, min_test_positives=10)
    assert not promoted and 'one class' in reason


def test_an_undersized_test_split_refuses_promotion(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated()
    metrics = evaluate(train(rows, seed=1), rows)
    promoted, reason = may_promote(metrics, min_test_auc=0.5, min_test_positives=metrics.test_positives + 1)
    assert not promoted and 'fewer than' in reason
    assert may_promote(metrics, min_test_auc=0.5, min_test_positives=metrics.test_positives)[0]


def test_a_model_below_the_minimum_auc_is_not_promoted(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated()
    metrics = evaluate(train(rows, seed=1), rows)
    assert metrics.test_auc is not None
    promoted, reason = may_promote(metrics, min_test_auc=metrics.test_auc + 1e-6, min_test_positives=10)
    assert not promoted and 'below' in reason
    assert may_promote(metrics, min_test_auc=metrics.test_auc - 1e-6, min_test_positives=10)[0]


def test_training_needs_both_classes_in_the_training_split(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated(test_all_positive=True)
    target = rows.target.copy()
    target[rows.label_source != JEV] = 1
    with pytest.raises(TrainingError, match='one class'):
        train(TrainingRows(rows.frame, target, rows.weight, rows.split, rows.label_source, rows.languages), seed=1)


def test_a_model_without_weak_rows_still_evaluates(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated(weak_rows=0, random_rows=0)
    assert evaluate(train(rows, seed=1), rows).block_auc is None


def test_rows_carry_the_three_label_sources(fabricated: Callable[..., TrainingRows]) -> None:
    rows = fabricated()
    assert set(rows.label_source.tolist()) == {JEV, BLOCK, RANDOM}
    assert (rows.split[rows.label_source != JEV] == TRAIN).all()
