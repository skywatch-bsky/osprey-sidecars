import os
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from faction_analysis import model_store
from faction_analysis.model import FEATURE_CONFIG, TrainingRows, train


def manifest_for(payload: bytes, version: str = 'h-1-abcdef012345') -> model_store.ModelManifest:
    import hashlib

    return model_store.ModelManifest(
        version=version,
        sha256=hashlib.sha256(payload).hexdigest(),
        trained_at='2026-11-27T00:00:00+00:00',
        dataset_version='ds-1',
        label_policy_hash='policy',
        feature_config=dict(FEATURE_CONFIG),
        row_counts={'train': 3, 'calibration': 1, 'test': 1},
        metrics={'test_auc': 0.9},
        calibrated=True,
    )


def trained(fabricated: Callable[..., TrainingRows]) -> tuple[bytes, TrainingRows]:
    rows = fabricated(jev_rows=200, weak_rows=50, random_rows=50)
    return model_store.serialise(train(rows, seed=1)), rows


def test_a_promoted_model_is_current_and_scores(tmp_path: Path, fabricated: Callable[..., TrainingRows]) -> None:
    payload, rows = trained(fabricated)
    assert model_store.load_current(tmp_path) is None
    assert model_store.is_stale(None, now=datetime.now(UTC), max_age_days=28)
    model_store.promote(tmp_path, payload, manifest_for(payload))
    loaded = model_store.load_current(tmp_path)
    assert loaded is not None
    model, manifest = loaded
    assert manifest.version == 'h-1-abcdef012345'
    assert manifest.row_counts == {'train': 3, 'calibration': 1, 'test': 1}
    probability = model.probabilities(rows.frame.iloc[:5])
    assert np.isfinite(probability).all()


def test_the_pointer_names_the_latest_promotion(tmp_path: Path, fabricated: Callable[..., TrainingRows]) -> None:
    payload, _rows = trained(fabricated)
    model_store.promote(tmp_path, payload, manifest_for(payload, 'h-1-first'))
    model_store.promote(tmp_path, payload, manifest_for(payload, 'h-2-second'))
    current = model_store.current_manifest(tmp_path)
    assert current is not None and current.version == 'h-2-second'


def test_a_version_is_never_overwritten(tmp_path: Path, fabricated: Callable[..., TrainingRows]) -> None:
    payload, _rows = trained(fabricated)
    model_store.promote(tmp_path, payload, manifest_for(payload))
    with pytest.raises(model_store.ModelStoreError, match='already exists'):
        model_store.promote(tmp_path, payload, manifest_for(payload))
    assert (tmp_path / model_store.MODELS_DIR / 'h-1-abcdef012345' / model_store.MODEL_FILE).is_file()
    assert not list((tmp_path / model_store.MODELS_DIR).glob('.staging-*'))


def test_a_digest_mismatch_changes_nothing(tmp_path: Path, fabricated: Callable[..., TrainingRows]) -> None:
    payload, _rows = trained(fabricated)
    broken = replace(manifest_for(payload), sha256='0' * 64)
    with pytest.raises(model_store.ModelStoreError, match='digest'):
        model_store.promote(tmp_path, payload, broken)
    assert model_store.load_current(tmp_path) is None
    assert not list((tmp_path / model_store.MODELS_DIR).glob('.staging-*'))


def test_a_model_file_that_does_not_match_its_digest_is_an_error(
    tmp_path: Path, fabricated: Callable[..., TrainingRows]
) -> None:
    payload, _rows = trained(fabricated)
    model_store.promote(tmp_path, payload, manifest_for(payload))
    (tmp_path / model_store.MODELS_DIR / 'h-1-abcdef012345' / model_store.MODEL_FILE).write_bytes(payload + b'x')
    with pytest.raises(model_store.ModelStoreError, match='digest'):
        model_store.load_current(tmp_path)


def test_an_unreadable_store_is_an_error_and_not_an_absent_model(tmp_path: Path) -> None:
    (tmp_path / model_store.CURRENT_FILE).write_text('{"version": "missing"}')
    with pytest.raises(model_store.ModelStoreError, match='no readable current model'):
        model_store.load_current(tmp_path)


def test_the_model_bytes_reach_the_disk_before_the_pointer_names_them(
    tmp_path: Path, fabricated: Callable[..., TrainingRows], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crash of the machine the pointer must never select a model whose bytes were not written."""
    payload, _rows = trained(fabricated)
    flushed: list[str] = []
    real_fsync = os.fsync

    def spying_fsync(fd: int) -> None:
        try:
            flushed.append(os.readlink(f'/proc/self/fd/{fd}'))
        except OSError:
            flushed.append(f'descriptor {fd}')
        real_fsync(fd)

    monkeypatch.setattr(model_store.os, 'fsync', spying_fsync)
    model_store.promote(tmp_path, payload, manifest_for(payload))
    model = next(index for index, path in enumerate(flushed) if path.endswith(model_store.MODEL_FILE))
    pointer = next(index for index, path in enumerate(flushed) if model_store.CURRENT_FILE in path)
    assert model < pointer


def test_staleness(tmp_path: Path, fabricated: Callable[..., TrainingRows]) -> None:
    payload, _rows = trained(fabricated)
    model_store.promote(tmp_path, payload, manifest_for(payload))
    manifest = model_store.current_manifest(tmp_path)
    assert manifest is not None
    trained_at = datetime.fromisoformat(manifest.trained_at)
    assert not model_store.is_stale(manifest, now=trained_at + timedelta(days=27, hours=23), max_age_days=28)
    assert model_store.is_stale(manifest, now=trained_at + timedelta(days=28), max_age_days=28)


def test_version_and_serialisation(fabricated: Callable[..., TrainingRows]) -> None:
    assert model_store.model_version(1_764_000_000_000, 'abcdef0123456789') == 'h-1764000000000-abcdef012345'
    payload, _rows = trained(fabricated)
    assert payload
    assert model_store.utc_now_iso().endswith('+00:00')
