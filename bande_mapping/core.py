"""
Core Bande Mapping implementation.

The method constructs local consensus regions as intersections of
neighborhood balls and maps each node to the centroid of its region.
"""

import numpy as np
from scipy.spatial import KDTree


def consensus_centroid(points, i, neighbor_idx, neighbor_dists,
                       mc_samples=3000, seed=None):
    """
    Compute the centroid of the consensus region C_i.

    The consensus region is the intersection of n balls centered at the
    n nearest neighbors with radii equal to the distance from P_i to
    each neighbor.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Original point coordinates.
    i : int
        Index of the node being processed.
    neighbor_idx : array-like, shape (n,)
        Indices of the n nearest neighbors of points[i].
    neighbor_dists : array-like, shape (n,)
        Distances from points[i] to its neighbors.
    mc_samples : int, optional
        Monte Carlo samples for centroid approximation (default 3000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (d,)
        Centroid of the consensus region C_i.
    """
    nbrs = points[neighbor_idx]
    radii = neighbor_dists
    d = points.shape[1]

    def in_consensus_region(x):
        return all(
            np.linalg.norm(x - p) <= r + 1e-9
            for p, r in zip(nbrs, radii)
        )

    # Fast path: neighbor centroid is often inside C_i for small n
    candidate = np.mean(nbrs, axis=0)
    if in_consensus_region(candidate):
        return candidate

    # Bounding box of C_i (intersection of axis-aligned bounding boxes of balls)
    lo = np.max(nbrs - radii[:, None], axis=0)
    hi = np.min(nbrs + radii[:, None], axis=0)

    if np.any(lo > hi + 1e-9):
        # Degenerate: bounding box is empty. P_i is guaranteed in C_i.
        return points[i]

    # Monte Carlo sampling inside bounding box, retain points in C_i
    rng = np.random.RandomState(seed if seed is not None else i)
    samples = rng.uniform(lo, hi, (mc_samples, d))
    mask = np.array([in_consensus_region(s) for s in samples])
    inside = samples[mask]

    if len(inside) > 0:
        return np.mean(inside, axis=0)

    # Fallback: P_i is always in C_i by construction (Theorem 2)
    return points[i]


def bande_transform(points, n, mc_samples=3000):
    """
    Apply Bande Mapping transformation to a point set.

    For each point P_i, compute the consensus region C_i (intersection
    of n neighborhood balls) and return its centroid V_i.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Original Euclidean coordinates.
    n : int
        Neighborhood degree (number of nearest neighbors).
        Must satisfy 1 <= n <= N-1.
    mc_samples : int, optional
        Monte Carlo samples for centroid approximation (default 3000).

    Returns
    -------
    np.ndarray, shape (N, d)
        Virtual equilibrium coordinates V.

    Examples
    --------
    >>> import numpy as np
    >>> from bande_mapping import bande_transform
    >>> points = np.random.uniform(0, 100, (50, 2))
    >>> V = bande_transform(points, n=5)
    >>> V.shape
    (50, 2)
    """
    points = np.asarray(points, dtype=float)
    N, d = points.shape

    if not (1 <= n <= N - 1):
        raise ValueError(f"n must satisfy 1 <= n <= N-1 (got n={n}, N={N})")

    tree = KDTree(points)
    distances, indices = tree.query(points, k=n + 1)

    virtual = np.zeros_like(points)
    for i in range(N):
        # Skip self (index 0 in query result)
        virtual[i] = consensus_centroid(
            points, i, indices[i, 1:], distances[i, 1:], mc_samples
        )

    return virtual


def bande_score(physical, virtual):
    """
    Compute the Bande Score: total displacement from physical to virtual.

    S(P, V) = sum_i ||P_i - V_i||_2

    Lower scores indicate higher geometric consistency.

    Parameters
    ----------
    physical : np.ndarray, shape (N, d)
        Original physical coordinates.
    virtual : np.ndarray, shape (N, d)
        Virtual equilibrium coordinates from bande_transform.

    Returns
    -------
    float
        Total Bande Score.
    """
    physical = np.asarray(physical)
    virtual = np.asarray(virtual)
    return float(np.sum(np.linalg.norm(physical - virtual, axis=1)))


def constraint_activation_rate(points, n):
    """
    Compute the fraction of nodes where the consensus constraint is binding.

    A node has an active constraint if the unconstrained neighbor centroid
    lies outside its consensus region C_i.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Point coordinates.
    n : int
        Neighborhood degree.

    Returns
    -------
    float
        Fraction of nodes with active constraint, in [0, 1].
    """
    points = np.asarray(points, dtype=float)
    N = len(points)
    n = min(n, N - 1)

    tree = KDTree(points)
    distances, indices = tree.query(points, k=n + 1)

    active = 0
    for i in range(N):
        nbrs = points[indices[i, 1:]]
        radii = distances[i, 1:]
        centroid = np.mean(nbrs, axis=0)
        outside = any(
            np.linalg.norm(centroid - p) > r + 1e-9
            for p, r in zip(nbrs, radii)
        )
        if outside:
            active += 1

    return active / N
