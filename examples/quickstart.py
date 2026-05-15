"""
Quickstart: Apply Bande Mapping to a TSP instance in 5 lines.

Run from repository root:
    python examples/quickstart.py
"""

from scipy.spatial.distance import cdist
from bande_mapping import anc_transform, nn_tsp, two_opt, tour_cost
from bande_mapping.instances import load_embedded

# Load benchmark instance
points, optimal = load_embedded("berlin52")
D_physical = cdist(points, points)

# --- Baseline: Nearest Neighbor + 2-opt on physical coordinates ---
baseline_tour = two_opt(D_physical, nn_tsp(D_physical, start=0))
baseline_cost = tour_cost(D_physical, baseline_tour)

# --- Bande Mapping: ANC-tuned, NN on virtual, 2-opt on physical ---
V, n_star = anc_transform(points, lambda D: nn_tsp(D, start=0))
D_virtual = cdist(V, V)
bande_tour = nn_tsp(D_virtual, start=0)          # construct on virtual
bande_tour = two_opt(D_physical, bande_tour)     # improve on physical
bande_cost = tour_cost(D_physical, bande_tour)

# --- Report ---
improvement = (bande_cost - baseline_cost) / baseline_cost * 100
print(f"Instance: berlin52 (N={len(points)}, optimal={optimal})")
print(f"Baseline (NN + 2-opt):  {baseline_cost:>8.1f}  "
      f"(gap +{(baseline_cost - optimal) / optimal * 100:.1f}%)")
print(f"Bande + NN + 2-opt:     {bande_cost:>8.1f}  "
      f"(gap +{(bande_cost - optimal) / optimal * 100:.1f}%)")
print(f"Selected n*:            {n_star}")
print(f"Improvement:            {improvement:+.2f}%")
