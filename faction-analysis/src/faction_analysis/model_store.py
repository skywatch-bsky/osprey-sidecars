# pattern: Imperative Shell
"""The model store on the model volume.

Layout: ``models/<version>/model.joblib`` and ``manifest.json``, and ``current.json`` naming the version in
use. Promotion is atomic: a version directory is written under a temporary name and renamed into place, and
the pointer is replaced through a temporary file. A crash leaves either the previous model or the new one.
"""

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib

from faction_analysis.model import HostilityModel

MODELS_DIR = 'models'
CURRENT_FILE = 'current.json'
MODEL_FILE = 'model.joblib'
MANIFEST_FILE = 'manifest.json'


class ModelStoreError(RuntimeError):
    """The model store does not hold what it should."""


@dataclass(frozen=True)
class ModelManifest:
    """What a model is and how it was built."""

    version: str
    sha256: str
    trained_at: str
    dataset_version: str
    label_policy_hash: str
    feature_config: dict[str, Any]
    row_counts: dict[str, int]
    metrics: dict[str, Any]
    calibrated: bool


def model_version(trained_at_ms: int, digest: str) -> str:
    return f'h-{trained_at_ms}-{digest[:12]}'


def serialise(model: object) -> bytes:
    """The model as joblib bytes. The manifest records their digest."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_dir(path: Path) -> None:
    for child in sorted(path.rglob('*'), reverse=True):
        child.unlink()
    path.rmdir()


def promote(model_dir: Path, payload: bytes, manifest: ModelManifest) -> Path:
    """Write a serialised model and make it current. Nothing changes if a step fails."""
    if hashlib.sha256(payload).hexdigest() != manifest.sha256:
        raise ModelStoreError('the model bytes do not match the digest in the manifest')
    models = model_dir / MODELS_DIR
    models.mkdir(parents=True, exist_ok=True)
    final = models / manifest.version
    if final.exists():
        raise ModelStoreError(f'model version {manifest.version} already exists')
    staging = models / f'.staging-{manifest.version}-{os.getpid()}'
    staging.mkdir()
    try:
        # The bytes reach the disk before anything names them. A directory fsync alone does not flush a file.
        with open(staging / MODEL_FILE, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(staging / MANIFEST_FILE, asdict(manifest))
        _fsync_dir(staging)
        os.rename(staging, final)
    finally:
        if staging.exists():
            _remove_dir(staging)
    _fsync_dir(models)
    _write_json(model_dir / CURRENT_FILE, {'version': manifest.version}, sync_dir=True)
    return final


def current_manifest(model_dir: Path) -> ModelManifest | None:
    """The manifest of the current model, or ``None`` when there is none.

    A pointer or a manifest that cannot be read is an error, not an absent model: a corrupt store must not
    silently make the sidecar run without a model it paid for.
    """
    pointer = model_dir / CURRENT_FILE
    if not pointer.is_file():
        return None
    try:
        version = json.loads(pointer.read_text())['version']
        payload = json.loads((model_dir / MODELS_DIR / str(version) / MANIFEST_FILE).read_text())
        return ModelManifest(
            version=str(payload['version']),
            sha256=str(payload['sha256']),
            trained_at=str(payload['trained_at']),
            dataset_version=str(payload['dataset_version']),
            label_policy_hash=str(payload['label_policy_hash']),
            feature_config=dict(payload['feature_config']),
            row_counts={str(key): int(value) for key, value in dict(payload['row_counts']).items()},
            metrics=dict(payload['metrics']),
            calibrated=bool(payload['calibrated']),
        )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ModelStoreError(f'the model store holds no readable current model: {type(exc).__name__}') from exc


def load_current(model_dir: Path) -> tuple[HostilityModel, ModelManifest] | None:
    """The current model and its manifest, or ``None``. The joblib file is the sidecar's own."""
    manifest = current_manifest(model_dir)
    if manifest is None:
        return None
    path = model_dir / MODELS_DIR / manifest.version / MODEL_FILE
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.sha256:
        raise ModelStoreError(f'model {manifest.version} does not match its digest')
    model: HostilityModel = joblib.load(io.BytesIO(payload))
    return model, manifest


def is_stale(manifest: ModelManifest | None, *, now: datetime, max_age_days: int) -> bool:
    """True when there is no model or the current one is older than the maximum age."""
    if manifest is None:
        return True
    return now - datetime.fromisoformat(manifest.trained_at) >= timedelta(days=max_age_days)


def _write_json(path: Path, payload: object, *, sync_dir: bool = False) -> None:
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(temp, 'w') as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    if sync_dir:
        _fsync_dir(path.parent)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
