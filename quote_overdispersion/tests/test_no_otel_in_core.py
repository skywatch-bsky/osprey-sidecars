from __future__ import annotations

from pathlib import Path


def test_functional_core_has_no_opentelemetry_imports() -> None:
    package_dir = Path(__file__).resolve().parents[1] / 'src' / 'quote_overdispersion'
    allowed = {'telemetry.py', 'main.py'}
    offenders = []
    for path in package_dir.glob('*.py'):
        if path.name in allowed:
            continue
        if 'opentelemetry' in path.read_text():
            offenders.append(path.name)
    assert offenders == []
