"""Deterministic helpers for GLAM's adaptive architecture search."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def make_search_rng(seed: int) -> random.Random:
    """Create an isolated RNG so model training cannot perturb search order."""
    return random.Random(int(seed))


def stable_config_id(config: dict[str, Any]) -> str:
    """Return a stable, collision-resistant identifier for a model configuration."""
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]
