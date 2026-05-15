# Bande Mapping

**Heuristic-agnostic geometric preprocessing for combinatorial spatial optimization.**

Bande Mapping is a deterministic preprocessing layer that transforms spatial coordinates into a *virtual equilibrium space* before any routing heuristic is applied. It improves the performance of fundamentally different construction and local-search heuristics — without modifying their internal logic.

In a benchmark of **132 experiments** spanning **22 standard TSPLIB instances** and **6 heuristic pipelines** of increasing strength — from Nearest Neighbor + 2-opt up to a Lin–Kernighan-style local search — Bande Mapping wins **48% of matchups** ($p = 0.00007$, Wilcoxon signed-rank). The win–loss profile is systematically asymmetric: wins average $-2.61\%$ while losses average only $+1.29\%$, a $2.03\times$ ratio.

---

## Quick start

```bash
pip install numpy scipy
git clone https://github.com/toprakbandeoglu/bande-mapping.git
cd bande-mapping
PYTHONPATH=. python examples/quickstart.py
```

Or use it in your own code:

```python
import numpy as np
from scipy.spatial.distance import cdist
from bande_mapping import anc_transform, nn_tsp, two_opt, tour_cost

# Your point set (any shape, any scale, any dimension)
points = np.random.uniform(0, 1000, (100, 2))
D = cdist(points, points)

# Baseline: NN + 2-opt on physical coordinates
baseline = two_opt(D, nn_tsp(D, start=0))
baseline_cost = tour_cost(D, baseline)

# Bande: ANC selects optimal n, NN on virtual, 2-opt on physical
V, n_star = anc_transform(points, lambda D: nn_tsp(D, start=0))
Dv = cdist(V, V)
tour = two_opt(D, nn_tsp(Dv, start=0))
bande_cost = tour_cost(D, tour)

print(f"Improvement: {(bande_cost - baseline_cost) / baseline_cost * 100:+.2f}%")
print(f"Selected n*: {n_star}")
```

---

## What it does

For each point $P_i$ in your dataset, Bande Mapping:

1. Finds the $n$ nearest neighbors $N_n(P_i)$.
2. Forms the **consensus region** $C_i = \bigcap_{P_j \in N_n(P_i)} B(P_j, r_{ij})$, the intersection of balls centered at each neighbor.
3. Computes $V_i = \text{centroid}(C_i)$, the virtual equilibrium position.

The transformed coordinates $V = \{V_1, \dots, V_N\}$ define a smoother spatial manifold. Routing heuristics run on $V$, but the resulting tour is evaluated on the original physical coordinates.

The neighborhood degree $n$ is selected automatically by **Adaptive Neighborhood Calibration (ANC)**, which evaluates a range of $n$ values and picks the one minimizing the post-heuristic tour cost.

---

## Why it works

The consensus region $C_i$ is the locus of positions simultaneously *reachable* from all $n$ nearest neighbors. Its centroid is a point of minimum local conflict — the "fairest" position with respect to local geometry. Routing on these consensus positions encounters fewer geometric traps. Subsequent local search (2-opt, or-opt) then lands in a different — often superior — basin of attraction.

Crucially, Bande Mapping is **not equivalent to Laplacian smoothing**: the intersection constraint $V_i \in C_i$ is active for $>89\%$ of nodes in every benchmark instance, producing a measurably different embedding.

---

## Benchmark results

Win rate across **22 TSPLIB instances**, by pipeline strength:

| Pipeline                                  | Strength      | Wins   | Losses | Win rate | Avg $\Delta\%$        |
| ----------------------------------------- | ------------- | ------ | ------ | -------- | --------------------- |
| NN + 2-opt (3-pass limit)                 | weak          | 15/22  | 5/22   | 68%      | $-2.09\%$             |
| NI + 2-opt                                | medium        | 14/22  | 7/22   | 64%      | $-1.80\%$             |
| FI + 2-opt                                | medium        | 14/22  | 7/22   | 64%      | $-0.74\%$             |
| NN + 2-opt + or-opt                       | strong        | 8/22   | 6/22   | 36%      | $-0.20\%$             |
| Christofides + full 2-opt + or-opt        | stronger      | 6/22   | 3/22   | 27%      | $-0.48\%$             |
| LK-style (2-opt + or-opt + neighbor-3-opt)| **strongest** | 6/22   | 7/22   | **27%**  | $\mathbf{-0.09\%}$    |
| **All combined**                          |               | **63/132** | **35/132** | **48%** | $\mathbf{-0.90\%}$ |

The win rate declines monotonically from 68% down to 27% — but **the mean improvement stays net negative at every level**. Even at the strongest LK-style level, Bande produces substantial improvements on several instances: berlin52 $-2.03\%$, kroA100 $-1.69\%$, lin105 $-1.57\%$, rat99 $-1.04\%$, eil76 $-0.97\%$, swiss42 $-0.72\%$.

**lin105 wins on all 6 pipelines** — including the LK-style local search. This is the strongest demonstration of heuristic-agnostic preprocessing: the geometric transformation produces a virtual space that benefits routing decisions regardless of algorithm.

**Aggregate statistics across all 132 experiments:**

- Wilcoxon signed-rank: $p = 0.00007$
- One-sample t-test: $p = 0.00001$ (one-sided)
- 95% bootstrap CI for mean $\Delta$: $[-1.32\%, -0.51\%]$
- Win/loss magnitude ratio: $2.03\times$ (Mann–Whitney $p = 0.0006$)
- Largest single win: $-10.64\%$ (fri26, NN+2-opt)
- Largest single loss: $+5.51\%$ (swiss42, NI+2-opt)

### Scaling: large TSPLIB instances

To verify the effect persists at industrial scale, Bande Mapping was tested on four large TSPLIB instances using a vectorized NumPy 2-opt pipeline:

| Instance | $N$ | Optimum | Baseline | Bande | $n^*$ | Base gap | Bande gap | $\Delta$ |
|----------|-----|---------|----------|-------|-------|----------|-----------|----------|
| **lin318** | 318 | 42,029 | 43,792 | **43,468** | 5 | $+4.19\%$ | $+3.42\%$ | $\mathbf{-0.74\%}$ |
| pcb442 | 442 | 50,778 | **52,532** | 52,646 | 5 | $+3.45\%$ | $+3.68\%$ | $+0.22\%$ |
| **pr1002** | 1002 | 259,045 | 273,199 | **271,862** | 12 | $+5.46\%$ | $+4.95\%$ | $\mathbf{-0.49\%}$ |
| **vm1084** | 1084 | 239,297 | 252,900 | **251,895** | 3 | $+5.68\%$ | $+5.26\%$ | $\mathbf{-0.40\%}$ |

**3 wins, 1 small loss, mean $\Delta = -0.35\%$.** Bande Mapping reduces gap-to-optimum on three of four large instances and preserves the favorable asymmetric profile observed on medium-scale instances. The single loss (pcb442) occurs on a highly regular grid-structured drilling problem where the geometric heterogeneity Bande exploits is limited.

---

## Installation

### From source

```bash
git clone https://github.com/toprakbandeoglu/bande-mapping.git
cd bande-mapping
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+

---

## API reference

### Core functions

```python
from bande_mapping import bande_transform, anc_transform
from bande_mapping.core import bande_score, constraint_activation_rate
```

**`bande_transform(points, n, mc_samples=3000)`**
Apply Bande Mapping with a fixed neighborhood degree.
- `points`: `(N, d)` array of coordinates
- `n`: neighborhood degree, $1 \le n \le N-1$
- Returns: `(N, d)` virtual coordinates

**`anc_transform(points, heuristic_fn, n_range=None, return_all_costs=False)`**
Apply Bande Mapping with Adaptive Neighborhood Calibration.
- `heuristic_fn(D) -> tour`: callable accepting a distance matrix
- Returns: `(V, n_star)` or `(V, n_star, costs_dict)` if `return_all_costs=True`

**`bande_score(physical, virtual)`**
Total displacement $\mathcal{S}(P, V) = \sum_i \|P_i - V_i\|_2$.

**`constraint_activation_rate(points, n)`**
Fraction of nodes where the consensus constraint is binding (typically $>0.9$).

### Heuristics

```python
from bande_mapping.heuristics import (
    nn_tsp, two_opt, or_opt,
    nearest_insertion, farthest_insertion,
    tour_cost,
)
```

Standard TSP heuristics in pure Python. Useful for reproducing the paper's experiments, but compiled implementations (LKH, Concorde) should be preferred for $N > 200$.

### Loading TSPLIB instances

```python
from bande_mapping.instances import load_tsplib, load_embedded

# From a .tsp file (most instances)
points = load_tsplib("path/to/berlin52.tsp")

# Embedded instances (only berlin52 and eil51)
points, optimal = load_embedded("berlin52")
```

The repository embeds only `berlin52` and `eil51` for quickstart convenience. For the other 24+4 benchmark instances reported in the paper, obtain the official `.tsp` files from the [TSPLIB library](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and load them with `load_tsplib()`.

---

## Examples

Two runnable examples in `examples/`:

- **`quickstart.py`** — minimal usage on a TSPLIB instance.
- **`custom_instance.py`** — applying Bande to your own coordinates, with diagnostics (constraint activation, Bande Score).

Run them with:

```bash
PYTHONPATH=. python examples/quickstart.py
```

---

## Tests

```bash
python -m unittest tests/test_core.py
```

The test suite verifies the theorems stated in the paper:

- **Theorem 1 (Convexity)**: $C_i$ is convex when non-empty.
- **Theorem 2 (Non-emptiness)**: $P_i \in C_i$ for all $i$.
- **Theorem 3 (Boundary degeneration)**: $V_i = P_{j^*}$ (nearest neighbor) at $n = 1$.
- **ANC correctness**: ANC returns the exact minimizer over the search range.

---

## Limitations

- **Scale.** Current implementation handles up to $N \approx 300$ in reasonable time. For larger instances, the centroid Monte Carlo and 2-opt would benefit from compiled implementations.
- **Centroid approximation.** $\text{centroid}(C_i)$ is computed via Monte Carlo sampling. An analytical formula for ball intersections would be faster but more complex.
- **Reproducibility tolerance.** Per-cell tour-cost numbers may differ by a few percent across machines due to Monte Carlo sampling, multi-start ordering, and 2-opt tie-breaking; the aggregate statistics (win rate, asymmetry, p-values) are robust to these variations.
- **Not a complete solver.** Bande Mapping is a *preprocessing layer*, not a replacement for state-of-the-art TSP solvers like LKH or Concorde. It complements them by reshaping the input geometry.

---

## Citing

If you use Bande Mapping in academic work, please cite:

```bibtex
@article{bandeoglu2026bande,
  title   = {Bande Mapping: Heuristic-Agnostic Geometric Preprocessing
             for Combinatorial Spatial Optimization},
  author  = {Bandeo{\u{g}}lu, Toprak},
  year    = {2026},
  note    = {Preprint},
}
```

---

## License

MIT. See [LICENSE](LICENSE).

---

## Contact

**Toprak Bandeoğlu**
Department of Industrial Engineering, Özyeğin University
[toprak.bandeoglu@ozu.edu.tr](mailto:toprak.bandeoglu@ozu.edu.tr)

Issues and pull requests welcome.
