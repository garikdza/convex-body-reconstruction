# Convex Body Reconstruction

[![codecov](https://codecov.io/gh/garikdza/Convex_Body_Reconstruction/branch/main/graph/badge.svg)](https://codecov.io/gh/garikdza/Convex_Body_Reconstruction)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reconstruction of a convex body in 2D or 3D from noisy measurements of its support function.

<p align="center">
  <img src="examples/figures/animation_2d.gif" width="800"
       alt="A polygon estimate tightening around a rounded heptagon as the number of measurements grows from 5 to 400, beside its L2 and Hausdorff errors falling with k">
</p>
<p align="center"><sub>Reconstruction from k = 5 to 400 noisy measurements</sub></p>

The support function of a convex body K assigns to each unit vector u the value
h<sub>K</sub>(u) = max<sub>x∈K</sub> ⟨x, u⟩, how far K extends in the direction u. Given unit directions u<sub>1</sub>, …, u<sub>k</sub> and measurements
y<sub>i</sub> = h<sub>K</sub>(u<sub>i</sub>) + noise, the estimator finds the support values h<sub>1</sub>, …,
h<sub>k</sub> of a convex body closest to the measurements and returns the polytope
{x : ⟨x, u<sub>i</sub>⟩ ≤ h<sub>i</sub> for all i} ([Prince & Willsky, 1990](https://doi.org/10.1109/34.50623)). As
k grows the polytope converges to K ([Gardner, Kiderlen & Milanfar, 2006](https://doi.org/10.1214/009053606000000335)).

<p align="center">
  <img src="examples/figures/support_function.png" width="340"
       alt="A convex body, a direction u and its supporting line at distance h_K(u) from the origin">
</p>

## Features

**Fits**, chosen with `method`:

| `method`          | fit                                           | solved as         |
| ----------------- | --------------------------------------------- | ----------------- |
| `'lsq'` (default) | least squares                                 | quadratic program |
| `'lad'`           | least absolute deviations, robust to outliers | linear program    |

**Consistency constraints**, which make h<sub>1</sub>, …, h<sub>k</sub> the support values of a convex body:

- In 2D: with the directions sorted by angle, each line
  ⟨x, u<sub>i</sub>⟩ = h<sub>i</sub> touches the polygon cut out by its two neighbours, one linear constraint per
  direction ([Lele, Kulkarni & Willsky, 1992](https://doi.org/10.1364/JOSAA.9.001693)). k = 20,000 solves in under a
  second.
- In 3D: there are points x<sub>i</sub> with
  ⟨x<sub>i</sub>, u<sub>i</sub>⟩ = h<sub>i</sub> and ⟨x<sub>j</sub>, u<sub>i</sub>⟩ ≤ h<sub>i</sub> for all i, j. Only
  the constraints between nearby directions are active at the optimum, so they are added by constraint generation:
  start from each direction's nearest neighbours, solve, add the violated constraints and repeat. The result is
  exact, and k = 600 in 3D solves in under 2 s.

**Bodies**: `Polytope`, `Ellipsoid` and their Minkowski sums (`+`, which adds the support functions), compared with
`hausdorff_distance` and `l2_distance`.

## Install

```bash
git clone https://github.com/garikdza/Convex_Body_Reconstruction
cd Convex_Body_Reconstruction
uv sync
```

## Quickstart

```python
import numpy as np
from src import circle_directions, hausdorff_distance, random_polygon, reconstruct

body = random_polygon(7, rng=0)                                  # a test body
directions = circle_directions(200)                              # 200 unit directions
noise = 0.05 * np.random.default_rng(1).standard_normal(200)
y = body.support(directions) + noise                             # noisy support values

estimate = reconstruct(directions, y)                            # a Polytope; method='lad' is robust
estimate.vertices                                                # its vertices
hausdorff_distance(body, estimate)                               # its distance to the body
```

In 3D, take the directions from `sphere_directions(k)` and a body such as `random_polytope(n)` or `Ellipsoid(matrix, center)`.
`noisy_support(body, directions, sigma)` simulates the measurements in one call.

## Examples

<p align="center">
  <img src="examples/figures/reconstruction_3d.png" width="900"
       alt="A rounded 3D polytope beside its reconstructions from 60 and 600 measurements">
</p>
<p align="center"><sub>A rounded random polytope and its least-squares reconstructions from 60 and 600 directions with
noise σ = 0.03.</sub></p>

More in [`examples/`](examples):

- [`noise.py`](examples/noise.py): the error of least squares and least absolute deviations against the noise level.
- [`outliers.py`](examples/outliers.py): least squares pulled by outliers, least absolute deviations unaffected.

```bash
uv sync --extra examples
uv run --extra examples python examples/reconstruction_3d.py   # any script; the slow ones save their results,
                                                               # --recompute computes them again
```

## References

- J. L. Prince, A. S. Willsky, "Reconstructing convex sets from support line measurements", *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 12(4), 377–389, 1990. https://doi.org/10.1109/34.50623
- A. S. Lele, S. R. Kulkarni, A. S. Willsky, "Convex-polygon estimation from support-line measurements and
  applications to target reconstruction from laser-radar data", *Journal of the Optical Society of America A*, 9(10),
  1693–1714, 1992. https://doi.org/10.1364/JOSAA.9.001693
- R. J. Gardner, M. Kiderlen, P. Milanfar, "Convergence of algorithms for reconstructing convex bodies and
  directional measures", *Annals of Statistics*, 34(3), 1331–1374, 2006. https://doi.org/10.1214/009053606000000335
- R. J. Gardner, M. Kiderlen, "A new algorithm for 3D reconstruction from support functions", *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 31(3), 556–562, 2009. https://doi.org/10.1109/TPAMI.2008.190

## License

Released under the [MIT License](LICENSE), © 2020 [garikdza](https://github.com/garikdza).
