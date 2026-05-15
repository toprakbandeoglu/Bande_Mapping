"""
Adaptive Neighborhood Calibration (ANC).

Selects the optimal neighborhood degree n* by exhaustive search over a
range of candidate values, evaluating each on the post-heuristic tour
cost computed on physical coordinates.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .core import bande_transform
from .heuristics import tour_cost


def anc_transform(points, heuristic_fn, n_range=None, evaluator=None,
                  return_all_costs=False, verbose=False):
    """
    Apply Bande Mapping with Adaptive Neighborhood Calibration (ANC).

    Tests multiple values of n and returns the virtual space that
    minimizes the post-heuristic tour cost on physical coordinates.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
        Original Euclidean coordinates.
    heuristic_fn : callable
        Function that takes a distance matrix and returns a tour
        (list of node indices). Example: lambda D: nn_tsp(D, start=0).
    n_range : iterable of int, optional
        Range of n values to test. Default: range(3, min(20, N-1)).
    evaluator : callable, optional
        Custom evaluator. Default: tour_cost(physical_D, tour).
        Receives (physical_D, tour) and returns a float.
    return_all_costs : bool, optional
        If True, also return dict mapping n -> cost. Default False.
    verbose : bool, optional
        If True, print progress for each n. Default False.

    Returns
    -------
    V_best : np.ndarray, shape (N, d)
        Virtual coordinates at the optimal n*.
    n_star : int
        Selected neighborhood degree.
    costs : dict (optional)
        Mapping n -> tour_cost, returned only if return_all_costs=True.

    Examples
    --------
    >>> from bande_mapping import anc_transform, nn_tsp
    >>> import numpy as np
    >>> points = np.random.uniform(0, 100, (50, 2))
    >>> V, n_star = anc_transform(points, lambda D: nn_tsp(D))
    >>> print(f"Optimal n* = {n_star}")
    """
    points = np.asarray(points, dtype=float)
    N = len(points)

    if n_range is None:
        n_range = range(3, min(20, N - 1))

    physical_D = cdist(points, points)

    if evaluator is None:
        evaluator = lambda D, tour: tour_cost(D, tour)

    costs = {}
    best_cost = float('inf')
    best_n = None
    best_V = None

    for n in n_range:
        if n >= N - 1 or n < 1:
            continue

        V = bande_transform(points, n)
        virtual_D = cdist(V, V)
        tour = heuristic_fn(virtual_D)
        cost = evaluator(physical_D, tour)

        costs[n] = cost

        if cost < best_cost:
            best_cost = cost
            best_n = n
            best_V = V

        if verbose:
            print(f"  n={n:2d}: tour_cost={cost:.2f}"
                  f"{'  *' if cost == best_cost else ''}")

    if return_all_costs:
        return best_V, best_n, costs
    return best_V, best_n


def anc_topk(points, heuristic_fn, k=2, n_range=None):
    """
    Return the top-k values of n by NN-tour cost (for efficient ANC).

    Useful when a full pipeline (e.g. with 2-opt) is expensive: first
    screen all n values with a cheap heuristic, then run the full
    pipeline only on the best k.

    Parameters
    ----------
    points : np.ndarray, shape (N, d)
    heuristic_fn : callable
        Cheap heuristic for screening (e.g. plain NN).
    k : int, optional
        Number of top values to return. Default 2.
    n_range : iterable of int, optional
        Default: range(3, min(20, N-1)).

    Returns
    -------
    list of int
        The k best n values, sorted ascending by cost.
    """
    _, _, costs = anc_transform(
        points, heuristic_fn, n_range=n_range, return_all_costs=True
    )
    return sorted(costs, key=costs.get)[:k]
