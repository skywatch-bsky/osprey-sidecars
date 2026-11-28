import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lockfile_is_not_ignored() -> None:
    completed = subprocess.run(
        ['git', 'check-ignore', '-q', 'faction-analysis/uv.lock'],
        cwd=ROOT.parent,
        check=False,
    )
    assert completed.returncode != 0


def test_lockfile_declares_python_314() -> None:
    project = (ROOT / 'pyproject.toml').read_text()
    assert 'requires-python = ">=3.14"' in project
    assert (ROOT / '.python-version').read_text().strip() == '3.14'
