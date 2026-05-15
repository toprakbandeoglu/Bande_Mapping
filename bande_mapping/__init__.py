"""
Bande Mapping: Heuristic-Agnostic Geometric Preprocessing for TSP
==================================================================

A deterministic preprocessing layer that transforms spatial coordinates
into a virtual equilibrium space before any routing heuristic is applied.

Usage:
    from bande_mapping import bande_transform, anc_transform

    # Single n value
    V = bande_transform(points, n=5)

    # Auto-tuned via ANC (recommended)
    V, n_star = anc_transform(points, heuristic_fn)
"""

from .core import bande_transform, consensus_centroid
from .anc import anc_transform
from .heuristics import (
    nn_tsp, two_opt, or_opt, nearest_insertion, farthest_insertion,
    tour_cost
)

__version__ = "0.1.0"
__author__ = "Toprak Bandeoğlu"

__all__ = [
    "bande_transform",
    "consensus_centroid",
    "anc_transform",
    "nn_tsp",
    "two_opt",
    "or_opt",
    "nearest_insertion",
    "farthest_insertion",
    "tour_cost",
]
