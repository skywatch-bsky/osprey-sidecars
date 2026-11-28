# pattern: Functional Core
"""The hostility model: character n-grams and structural features in a logistic regression.

The model reads the text of the reply or the quote on its own. It is a linear model that scores tens of
millions of texts in a run; the parent text is in the exported dataset for a transformer trained elsewhere.
Probabilities are calibrated on the calibration split and are summed into cells, never thresholded early.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from faction_analysis.labels import BLOCK, CALIBRATION, RANDOM, TEST, TRAIN

TEXT = 'text'
LENGTH = 'text_length'
UPPERCASE = 'uppercase'
EXCLAMATIONS = 'exclamations'
QUESTIONS = 'questions'
MENTIONS = 'mentions'
KIND = 'kind'
FRAME_COLUMNS = (TEXT, LENGTH, UPPERCASE, EXCLAMATIONS, QUESTIONS, MENTIONS, KIND)
STRUCTURAL_FEATURES = ('log1p(text_length)', 'exclamations', 'questions', 'uppercase_ratio', 'mentions', 'kind')
N_FEATURES = 2**20
NGRAM_RANGE = (2, 5)
TOP_LANGUAGES = 5
MIN_LANGUAGE_POSITIVES = 50

FEATURE_CONFIG = {
    'analyzer': 'char',
    'ngram_range': list(NGRAM_RANGE),
    'n_features': N_FEATURES,
    'structural': list(STRUCTURAL_FEATURES),
}

_vectorizer = HashingVectorizer(
    analyzer='char', ngram_range=NGRAM_RANGE, n_features=N_FEATURES, lowercase=True, alternate_sign=False
)


class TrainingError(ValueError):
    """The rows cannot train a model."""


def text_frame(
    text: list[str],
    text_length: np.ndarray,
    uppercase: np.ndarray,
    exclamations: np.ndarray,
    questions: np.ndarray,
    mentions: np.ndarray,
    kind: np.ndarray,
) -> pd.DataFrame:
    """The columns the model reads. One frame is also what scoring hands over in batches."""
    return pd.DataFrame(
        {
            TEXT: pd.Series(text, dtype=object),
            LENGTH: np.asarray(text_length, dtype=np.int64),
            UPPERCASE: np.asarray(uppercase, dtype=np.int64),
            EXCLAMATIONS: np.asarray(exclamations, dtype=np.int64),
            QUESTIONS: np.asarray(questions, dtype=np.int64),
            MENTIONS: np.asarray(mentions, dtype=np.int64),
            KIND: np.asarray(kind, dtype=np.int8),
        }
    )


def structural_matrix(frame: pd.DataFrame) -> np.ndarray:
    length = frame[LENGTH].to_numpy(dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            length > 0, frame[UPPERCASE].to_numpy(dtype=np.float64) / np.where(length > 0, length, 1.0), 0.0
        )
    return np.column_stack(
        [
            np.log1p(length),
            frame[EXCLAMATIONS].to_numpy(dtype=np.float64),
            frame[QUESTIONS].to_numpy(dtype=np.float64),
            ratio,
            frame[MENTIONS].to_numpy(dtype=np.float64),
            frame[KIND].to_numpy(dtype=np.float64),
        ]
    )


def featurize(frame: pd.DataFrame) -> sparse.csr_matrix:
    """Character 2 to 5 grams of the text next to the six structural features."""
    if len(frame) == 0:
        return sparse.csr_matrix((0, N_FEATURES + len(STRUCTURAL_FEATURES)))
    texts = [value if isinstance(value, str) and value else '' for value in frame[TEXT].tolist()]
    return sparse.hstack([_vectorizer.transform(texts), sparse.csr_matrix(structural_matrix(frame))]).tocsr()


def _base_model(seed: int, c: float) -> Pipeline:
    return Pipeline(
        [
            ('features', FunctionTransformer(featurize, validate=False)),
            ('classifier', LogisticRegression(C=c, max_iter=1000, random_state=seed)),
        ]
    )


@dataclass(frozen=True)
class TrainingRows:
    """The labelled rows with the text columns the model reads."""

    frame: pd.DataFrame
    target: np.ndarray
    weight: np.ndarray
    split: np.ndarray
    label_source: np.ndarray
    languages: list[tuple[str, ...]]

    def rows(self, mask: np.ndarray) -> TrainingRows:
        return TrainingRows(
            self.frame.loc[mask].reset_index(drop=True),
            self.target[mask],
            self.weight[mask],
            self.split[mask],
            self.label_source[mask],
            [language for language, keep in zip(self.languages, mask, strict=True) if keep],
        )


@dataclass(frozen=True)
class HostilityModel:
    """A fitted model. ``probabilities`` returns ``p(hostile)`` for each row of a frame."""

    estimator: ProbabilityEstimator
    calibrated: bool

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(frame), dtype=np.float64)[:, 1]


class ProbabilityEstimator(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def train(rows: TrainingRows, *, seed: int, c: float = 1.0) -> HostilityModel:
    """Fit on the train split with sample weights, then calibrate on the calibration split."""
    training = rows.split == TRAIN
    if len(np.unique(rows.target[training])) < 2:
        raise TrainingError('the training split has one class')
    base = _base_model(seed, c)
    base.fit(rows.frame[training], rows.target[training], classifier__sample_weight=rows.weight[training])
    calibration = rows.split == CALIBRATION
    if len(np.unique(rows.target[calibration])) < 2:
        return HostilityModel(base, calibrated=False)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    calibrated.fit(rows.frame[calibration], rows.target[calibration])
    return HostilityModel(calibrated, calibrated=True)


@dataclass(frozen=True)
class Metrics:
    """Metrics on the Jev-labelled test split, and the agreement with the block labels."""

    test_rows: int
    test_positives: int
    test_auc: float | None
    test_average_precision: float | None
    test_brier: float | None
    per_language_ap: dict[str, float | None]
    block_auc: float | None
    calibrated: bool

    def as_dict(self) -> dict[str, object]:
        return {
            'test_rows': self.test_rows,
            'test_positives': self.test_positives,
            'test_auc': self.test_auc,
            'test_average_precision': self.test_average_precision,
            'test_brier': self.test_brier,
            'per_language_ap': dict(self.per_language_ap),
            'block_auc': self.block_auc,
            'calibrated': self.calibrated,
        }


def _auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, probability))


def evaluate(model: HostilityModel, rows: TrainingRows) -> Metrics:
    """A metric on a split with one class is ``None``.

    Per-language average precision is reported for the most common languages of the test split that have at
    least ``MIN_LANGUAGE_POSITIVES`` positives. ``block_auc`` is the ROC-AUC of block positives against the
    random negatives: how far the model agrees with the block labels it did not train on.
    """
    test = rows.rows(rows.split == TEST)
    if len(test.target) == 0:
        return Metrics(0, 0, None, None, None, {}, None, model.calibrated)
    probability = model.probabilities(test.frame)
    positives = int(test.target.sum())
    per_language: dict[str, float | None] = {}
    counts = Counter(language for languages in test.languages for language in languages)
    for language, _ in counts.most_common(TOP_LANGUAGES):
        mask = np.array([language in languages for languages in test.languages])
        if int(test.target[mask].sum()) >= MIN_LANGUAGE_POSITIVES:
            per_language[language] = float(average_precision_score(test.target[mask], probability[mask]))
        else:
            per_language[language] = None
    weak = rows.rows((rows.label_source == BLOCK) | (rows.label_source == RANDOM))
    weak_probability = model.probabilities(weak.frame) if len(weak.target) else np.empty(0)
    return Metrics(
        test_rows=len(test.target),
        test_positives=positives,
        test_auc=_auc(test.target, probability),
        test_average_precision=(
            float(average_precision_score(test.target, probability))
            if positives > 0 and positives < len(test.target)
            else None
        ),
        test_brier=(float(brier_score_loss(test.target, probability)) if 0 < positives < len(test.target) else None),
        per_language_ap=per_language,
        block_auc=_auc(weak.target, weak_probability),
        calibrated=model.calibrated,
    )


def may_promote(metrics: Metrics, *, min_test_auc: float, min_test_positives: int) -> tuple[bool, str]:
    """Whether a model may replace the current one, and the reason when it may not."""
    if metrics.test_rows == 0:
        return False, 'the test split is empty'
    if metrics.test_auc is None:
        return False, 'the test split has one class'
    if metrics.test_positives < min_test_positives:
        return False, f'the test split has {metrics.test_positives} positives, fewer than {min_test_positives}'
    if metrics.test_auc < min_test_auc:
        return False, f'test ROC-AUC {metrics.test_auc:.3f} is below {min_test_auc}'
    return True, 'promoted'
