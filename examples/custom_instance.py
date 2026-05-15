"""
Custom instance example: apply Bande Mapping to your own coordinate data.

Run from repository root:
    python examples/custom_instance.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from bande_mapping import anc_transform, nn_tsp, two_opt, or_opt, tour_cost
from bande_mapping.core import bande_score, constraint_activation_rate

# --- Step 1: prepare your point set ---
# Replace this with np.loadtxt("my_coords.csv") or any (N, 2) array.
np.random.seed(42)
points = np.random.uniform(0, 1000, (60, 2))
D = cdist(points, points)

print(f"Instance: N={len(points)} points")

# --- Step 2: diagnostics on the consensus region ---
activation = constraint_activation_rate(points, n=5)
print(f"Constraint activation at n=5: {activation:.0%}")
print("(High activation means the consensus intersection is a binding "
      "constraint, not vacuous.)")

# --- Step 3: baseline pipeline ---
baseline = two_opt(D, nn_tsp(D, start=0))
baseline = or_opt(D, baseline)
baseline_cost = tour_cost(D, baseline)
print(f"\nBaseline (NN + 2-opt + or-opt): {baseline_cost:.1f}")

# --- Step 4: Bande + same pipeline ---
V, n_star = anc_transform(points, lambda D: nn_tsp(D, start=0))
Dv = cdist(V, V)

tour = nn_tsp(Dv, start=0)
tour = two_opt(D, tour)
tour = or_opt(D, tour)
bande_cost = tour_cost(D, tour)

score = bande_score(points, V)
delta = (bande_cost - baseline_cost) / baseline_cost * 100

print(f"Bande + same pipeline:           {bande_cost:.1f}")
print(f"Selected n*:                     {n_star}")
print(f"Bande Score (total tension):     {score:.1f}")
print(f"Improvement:                     {delta:+.2f}%")
