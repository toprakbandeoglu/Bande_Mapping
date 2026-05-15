"""
Standard TSP heuristics for use with Bande Mapping.

These implementations are pure Python for clarity and pedagogical value.
For production use on large instances (N > 200), compiled implementations
(e.g., LKH, Concorde) are recommended.
"""

import numpy as np


def tour_cost(D, tour):
    """
    Compute the total length of a tour given a distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Distance matrix.
    tour : list or array of int
        Tour as a sequence of node indices (Hamiltonian cycle).

    Returns
    -------
    float
        Total tour length, including the closing edge.
    """
    N = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % N]] for i in range(N)))


def nn_tsp(D, start=0):
    """
    Nearest Neighbor TSP heuristic.

    Greedily constructs a tour by always visiting the nearest unvisited
    city.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Distance matrix.
    start : int, optional
        Starting node. Default 0.

    Returns
    -------
    list of int
        Tour visiting each node exactly once.
    """
    N = D.shape[0]
    unvisited = set(range(N))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        current = tour[-1]
        next_node = min(unvisited, key=lambda j: D[current, j])
        tour.append(next_node)
        unvisited.remove(next_node)
    return tour


def two_opt(D, tour, max_passes=None):
    """
    2-opt local search.

    Iteratively reverses tour segments to remove crossing edges.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
    tour : list of int
        Initial tour.
    max_passes : int or None, optional
        Maximum full scans. None for unrestricted (until convergence).

    Returns
    -------
    list of int
        Improved tour (2-opt local optimum).
    """
    tour = list(tour)
    N = len(tour)
    passes = 0
    improved = True
    while improved:
        if max_passes is not None and passes >= max_passes:
            break
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % N]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        passes += 1
    return tour


def or_opt(D, tour, segment_sizes=(3, 2, 1)):
    """
    Or-opt local search.

    Relocates segments of nodes to other positions in the tour.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
    tour : list of int
    segment_sizes : tuple of int, optional
        Segment lengths to try, in order. Default (3, 2, 1).

    Returns
    -------
    list of int
        Improved tour.
    """
    tour = list(tour)
    N = len(tour)
    improved = True
    while improved:
        improved = False
        for seg_len in segment_sizes:
            for i in range(N):
                seg_idx = [(i + k) % N for k in range(seg_len)]
                seg_nodes = [tour[s] for s in seg_idx]
                before = tour[(i - 1) % N]
                after = tour[(i + seg_len) % N]
                removal_gain = (D[before, seg_nodes[0]]
                                + D[seg_nodes[-1], after]
                                - D[before, after])
                best_inc, best_pos = float('inf'), -1
                for j in range(N):
                    if any(j % N == s or (j + 1) % N == s for s in seg_idx):
                        continue
                    if j % N == (i - 1) % N:
                        continue
                    a, b = tour[j % N], tour[(j + 1) % N]
                    inc = D[a, seg_nodes[0]] + D[seg_nodes[-1], b] - D[a, b]
                    if inc < best_inc:
                        best_inc = inc
                        best_pos = j
                if removal_gain - best_inc > 1e-10 and best_pos >= 0:
                    new_tour = []
                    seg_set = set(seg_idx)
                    for k in range(N):
                        if k not in seg_set:
                            new_tour.append(tour[k])
                    ins_node = tour[best_pos % N]
                    ins_idx = new_tour.index(ins_node) + 1
                    for k, s in enumerate(seg_nodes):
                        new_tour.insert(ins_idx + k, s)
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break
    return tour


def nearest_insertion(D):
    """
    Nearest Insertion heuristic.

    Repeatedly insert the closest unrouted city at the cheapest position
    in the current partial tour.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)

    Returns
    -------
    list of int
    """
    N = D.shape[0]
    in_tour = [False] * N
    # Start with two closest cities
    min_d = float('inf')
    s1, s2 = 0, 1
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] < min_d:
                min_d = D[i, j]
                s1, s2 = i, j
    tour = [s1, s2]
    in_tour[s1] = in_tour[s2] = True

    while len(tour) < N:
        # Find nearest city to any node in tour
        best_d, best_c = float('inf'), -1
        for c in range(N):
            if in_tour[c]:
                continue
            for t in tour:
                if D[c, t] < best_d:
                    best_d = D[c, t]
                    best_c = c
        # Insert at cheapest position
        best_inc, best_pos = float('inf'), 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            inc = D[tour[i], best_c] + D[best_c, tour[j]] - D[tour[i], tour[j]]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, best_c)
        in_tour[best_c] = True
    return tour


def farthest_insertion(D):
    """
    Farthest Insertion heuristic.

    Repeatedly insert the city farthest from the current tour at the
    cheapest position. Tends to capture global structure early.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)

    Returns
    -------
    list of int
    """
    N = D.shape[0]
    in_tour = [False] * N
    min_d = float('inf')
    s1, s2 = 0, 1
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] < min_d:
                min_d = D[i, j]
                s1, s2 = i, j
    tour = [s1, s2]
    in_tour[s1] = in_tour[s2] = True

    while len(tour) < N:
        # Find farthest unvisited city (max-min distance)
        best_d, best_c = -1, -1
        for c in range(N):
            if in_tour[c]:
                continue
            min_to_tour = min(D[c, t] for t in tour)
            if min_to_tour > best_d:
                best_d = min_to_tour
                best_c = c
        best_inc, best_pos = float('inf'), 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            inc = D[tour[i], best_c] + D[best_c, tour[j]] - D[tour[i], tour[j]]
            if inc < best_inc:
                best_inc = inc
                best_pos = i + 1
        tour.insert(best_pos, best_c)
        in_tour[best_c] = True
    return tour
