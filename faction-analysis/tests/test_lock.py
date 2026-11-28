from pathlib import Path

import pytest

from faction_analysis.lock import AlreadyRunning, InstanceLock


def test_concurrent_start_refused_by_lock(tmp_path: Path) -> None:
    path = tmp_path / 'models' / 'faction-analysis.lock'
    first = InstanceLock(path)
    first.acquire()
    try:
        # flock is held per open file description, so a second open in the same process is refused as well.
        second = InstanceLock(path)
        with pytest.raises(AlreadyRunning):
            second.acquire()
    finally:
        first.release()
    again = InstanceLock(path)
    again.acquire()
    again.release()


def test_lock_is_released_on_context_exit(tmp_path: Path) -> None:
    path = tmp_path / 'lock'
    with InstanceLock(path):
        with pytest.raises(AlreadyRunning):
            InstanceLock(path).acquire()
    InstanceLock(path).acquire()


def test_release_without_acquire_is_harmless(tmp_path: Path) -> None:
    InstanceLock(tmp_path / 'missing.lock').release()
