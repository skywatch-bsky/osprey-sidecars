from __future__ import annotations

from pathlib import Path


def test_functional_core_has_no_opentelemetry_imports() -> None:
    package_dir = Path(__file__).resolve().parents[1] / 'src' / 'signup_anomaly'
    allowed = {'telemetry.py', 'main.py'}
    offenders = []
    for path in package_dir.rglob('*.py'):
        if path.name in allowed:
            continue
        if 'opentelemetry' in path.read_text():
            offenders.append(str(path.relative_to(package_dir)))
    assert offenders == []
