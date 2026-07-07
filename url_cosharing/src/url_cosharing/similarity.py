# pattern: Functional Core
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UrlShareRow:
    did: str
    url: str
    share_count: int
