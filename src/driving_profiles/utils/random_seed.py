"""Centralized random seed handling for reproducible synthetic generation.

Every stage of the pipeline that involves randomness (clustering, synthetic
sampling, ...) should source its seed from `get_seed`/`get_rng` rather than
hard-coding its own, so a single project-wide default seed reproduces the
full pipeline's output end to end unless a caller deliberately overrides it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 42


def get_seed(seed: int | None = None) -> int:
    """Return `seed` if given, otherwise the project default seed.

    Centralizing this (rather than every caller writing `seed or 42`) means
    the project default lives in exactly one place.
    """
    return DEFAULT_SEED if seed is None else seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a numpy random Generator seeded via `get_seed`."""
    return np.random.default_rng(get_seed(seed))
