"""Common foundations, types, schemas, and verification interfaces for dataset extraction."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


def init_seed(seed: int = 42) -> random.Random:
    """Return a seeded random instance for deterministic generation."""
    return random.Random(seed)


@dataclass
class DatasetSample:
    """Base dataset record envelope."""
    dataset_id: str
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)
