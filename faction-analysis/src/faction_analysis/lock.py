# pattern: Imperative Shell
"""The promotion-intent file and the instance lock.

The intent file lives in the model volume. It is the source of truth for the intent to promote a run, because
a local fsync is under the sidecar's control and a ClickHouse insert is not. It is installed through a
temporary file that is fsynced and renamed into place, so a crash leaves either no file or a complete one.
"""

import fcntl
import os
from pathlib import Path
from typing import Literal

from faction_analysis.promotion import IntentRecord, intent_document, parse_intent

INTENT_NAME = 'promote-pending.json'
TEMP_PREFIX = 'promote-pending'
CrashPoint = Literal['temp_written', 'temp_fsynced', 'renamed']


class AlreadyRunning(RuntimeError):
    """Another process on this host holds the lock."""


class SimulatedCrash(BaseException):
    """Stands in for the death of the process at a crash point.

    It is not an ``Exception``, so no handler that cleans up a failed run catches it.
    """

    def __init__(self, point: str) -> None:
        super().__init__(point)
        self.point = point


def intent_path(model_dir: Path) -> Path:
    return model_dir / INTENT_NAME


def _crash(fail_at: CrashPoint | None, point: CrashPoint) -> None:
    if fail_at == point:
        raise SimulatedCrash(point)


def install(model_dir: Path, record: IntentRecord, *, fail_at: CrashPoint | None = None) -> None:
    """Write the intent atomically: write a temporary file, fsync it, rename it over the intent, fsync the dir.

    ``fail_at`` raises ``SimulatedCrash`` at a crash point, for tests.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    temp = model_dir / f'{TEMP_PREFIX}.{os.getpid()}.tmp'
    with open(temp, 'w') as handle:
        handle.write(intent_document(record.run))
        _crash(fail_at, 'temp_written')
        handle.flush()
        os.fsync(handle.fileno())
    _crash(fail_at, 'temp_fsynced')
    os.replace(temp, intent_path(model_dir))
    _crash(fail_at, 'renamed')
    directory = os.open(model_dir, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(model_dir: Path) -> IntentRecord | None:
    """The recorded intent, or ``None`` when there is no intent file.

    An unreadable file raises ``IntentUnreadable``. It is never treated as absent: the run it describes is
    fully written and must not be cleaned up. Orphan temporary files are ignored.
    """
    path = intent_path(model_dir)
    if not path.is_file():
        return None
    return parse_intent(path.read_text())


def clear(model_dir: Path) -> None:
    intent_path(model_dir).unlink(missing_ok=True)


class InstanceLock:
    """An exclusive lock on a file, held for the lifetime of the process.

    One host runs one sidecar. A second process on the same host finds the lock taken and exits at once.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise AlreadyRunning(f'another faction-analysis process holds {self._path.name}') from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> InstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
