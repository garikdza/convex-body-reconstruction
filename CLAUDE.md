# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reconstruction of a convex body in 2D or 3D from noisy measurements y_i = h_K(u_i) + noise of its support function h_K(u) = max_{x∈K} ⟨x, u⟩. The estimator fits the consistent support values closest to the measurements, by least squares or least absolute deviations, and returns the polytope {x : ⟨x, u_i⟩ ≤ h_i}: Algorithm NoisySupportLSQ of Gardner, Kiderlen & Milanfar (2006), due to Prince & Willsky (1990). The references are in the README.

## Commands

The project is managed with uv (`pyproject.toml`, hatchling backend). The package directory is `src/`, imported as `src` (`from src import reconstruct`).

- `uv sync`: install the library plus the dev tools (pytest, pytest-cov, ruff).
- `uv run pytest`: run the tests. For a single test use `uv run pytest tests/test_solvers.py::test_local_and_pairwise_same_optimum`.
- `uv run pytest --cov`: run the tests with a coverage report for `src`. It fails below the 85% floor set in `pyproject.toml`, the same check CI runs. Add `--cov-report=html` for a browsable report.
- `uv run ruff check .`: lint (rules E, F, I, UP, B and N, line length 120). `src = [".", "examples"]` makes the shared example modules (`draw`, `results`, `shapes`, `style`) first-party imports.
- `uv run ruff format .`: format the code (single quotes; Markdown is excluded so README code blocks keep their aligned comments). CI runs `ruff format --check .` and fails on unformatted code.
- `uv sync --extra examples`: also install matplotlib, which only the example scripts need.
- `uv run --extra examples python examples/<script>.py`: run one example script, which writes its figure to `examples/figures/`. Only the three figures the README shows are tracked, and `.gitignore` keeps the rest out; `support_function.png` is a static illustration with no script. The README shows the animation, the support-function diagram and the 3D figure, and lists the other scripts. `reconstruction_3d.py` and `noise.py` save their raw results as `figures/<script>.npz` and redraw from that file on later runs; add `--recompute` to compute again, as needed after changing a script's settings (a changed setting is not detected).
- CI (`.github/workflows/ci.yml`) runs ruff, the format check and the tests with coverage on pushes to `main` and on PRs, and uploads the report to Codecov.
- Nothing here needs a GPU. Prefix Python runs with `CUDA_VISIBLE_DEVICES=""`, as the global GPU rule requires.

## Constants

No magic numbers in function bodies or default arguments: every tolerance, size, count, seed, colour and layout value is a named constant at the top of its module, with a comment saying what it does (the solver settings in `src/solvers.py`, the drawing sizes and z-orders in `examples/draw.py`, the colours in `examples/style.py`, each script's settings at its top). Docstrings name the constant rather than repeat its value. Mathematical constants stay inline: 2π, the golden angle, the dimensions 2 and 3, the quartiles 25/50/75.

## Naming

Variables and arguments are lowercase words, as PEP 8 asks and ruff's `N` rules enforce: `directions`, not `U`; `body` and `other`, not `K` and `L`; `polytope` or `estimate`, not `P`. The maths in docstrings, comments and the README keeps its notation (h_K(u), a body K, directions u_i). Short lowercase names from the formulas (`y`, `h`, `k`, `d`) are fine.

## Local folder

`local/` at the repository root is gitignored and never committed. Keep anything project-related that should not be tracked there: scratch scripts, drafts, notes, or code set aside from the package or the examples. Organise it in subfolders by topic. Ruff skips it through `.gitignore` and pytest only collects `tests/`, so nothing in it is linted or tested.

## Tests

Tests are unit tests of `src`, and `tests/` mirrors the package: `src/<module>.py` is tested in `tests/test_<module>.py`. The re-exporting `src/__init__.py` gets no test file.

- A test sits in the file of the module that holds the code it exercises, and errors are tested where they are raised: the halfspace errors (unbounded, empty, flat) in `test_geometry.py`, the local formulation's direction checks in `test_solvers.py`, the input checks of `reconstruct` in `test_reconstruct.py`.
- Test folders have no `__init__.py`: pytest runs with `--import-mode=importlib` (`addopts` in `pyproject.toml`), so test files can't import each other. Helpers shared across files are fixtures in `tests/conftest.py` (`square`, `cube`); a helper or constant used by one file stays in that file.
- Names say the behaviour checked, `test_<what>_<expected outcome>`, such as `test_lsq_projects_inconsistent_values_onto_sum_constraint`. Expected values come from hand calculation or a closed form, explained in a comment, not from rerunning the code under test. Bad inputs go through one parametrized test per module, each case with an `id`, matching the error message.
- Every test checks behaviour: no tests that only pin a default, only check an output's shape, or repeat a check made elsewhere.
- The solvers return optima to about 1e-5 on bodies of unit size, so tests that compare fitted values use `TOL = 1e-4`.
- Coverage measures `src` only. The `examples/` code has no tests: the scripts are run by hand.

## Architecture

The pipeline is `directions + measurements → consistent support values → polytope`. The public API is what `src/__init__.py` exports, kept to what a user needs: `reconstruct`, the bodies `Polytope` and `Ellipsoid`, the directions `circle_directions` and `sphere_directions`, the test bodies and `noisy_support`, and the two distances. Everything else (`ConvexBody`, `MinkowskiSum`, `evenly_spread`, `random_directions`, the solvers) is imported from its module.

- `bodies.py`: `ConvexBody`, the abstract base every body implements: `dim`, `support(directions)` (h_K at each row) and `support_points(directions)` (a point of K attaining it), plus `+` for the Minkowski sum, whose support function is the sum, and `to_polytope(n)`, which approximates the body by the hull of its support points in `n` evenly spread directions, for drawing and volumes (a `Polytope` returns itself). They are plain classes, not dataclasses: they check and transform their inputs, and `==` is identity. `Polytope(vertices)` keeps only the extreme points (counter-clockwise in 2D) and holds their `ConvexHull`, both read-only properties so they cannot go out of step; `Polytope.from_halfspaces(directions, h)` builds {x : ⟨x, u_i⟩ ≤ h_i}. `Ellipsoid(matrix, center)` is {c + Mz : |z| ≤ 1}. `MinkowskiSum(parts)` is what `+` returns, with nested sums flattened. Each body checks its shapes when built (a sum of a 2D and a 3D body raises, and so does a singular `matrix`).
- `geometry.py`: `chebyshev_center(directions, h)` (largest inscribed ball, a linear program; raises when the halfspaces leave the polyhedron unbounded) and `halfspace_vertices(directions, h)` (the vertices, from scipy's `HalfspaceIntersection` around that centre; raises when the interior is empty).
- `directions.py`: `circle_directions(k, offset)`, `sphere_directions(k)` (Fibonacci lattice), `evenly_spread(k, dim)` (one of the two) and `random_directions(k, dim, rng)`.
- `solvers.py`: the convex programs, both minimising Σ loss(y_i − h_i) over consistent h, with `loss` `'lsq'` (squared, a QP) or `'lad'` (absolute, an LP), solved with cvxpy's Clarabel (cvxpy's default QP solver, OSQP, stalls on them) using its single-threaded QDLDL factoriser (the default, faer, is about 6 times slower here because its threads contend). An unknown `loss` raises.
  - `fit_local_2d` writes consistency as one constraint per direction: with the directions sorted by angle t_i, h_{i−1} sin(t_{i+1} − t_i) − h_i sin(t_{i+1} − t_{i−1}) + h_{i+1} sin(t_i − t_{i−1}) ≥ 0, each row normalised so large `k` stays well conditioned. 2D only; the directions must be distinct and not in a closed half-plane.
  - `fit_pairwise` writes it with contact points x_i: ⟨x_i, u_i⟩ = h_i and ⟨x_j, u_i⟩ ≤ h_i for all i, j, as sparse constraint rows. It starts from each direction's `n_neighbors` nearest directions, adds every violated constraint after each solve, and stops when none is violated (exact) or raises after `max_rounds`.
- `reconstruct.py`: `reconstruct(directions, measurements, method='lsq')` validates the input (including, through `chebyshev_center`, that the directions are not in a closed hemisphere, before any solve), fits with `fit_local_2d` in 2D and `fit_pairwise` in 3D, and returns the estimate as a `Polytope`. Its support function at the directions is the fitted values, so `estimate.support(directions)` gives them and `y - estimate.support(directions)` the residuals; there is no separate result object.
- `simulate.py`: test bodies and the measurement model: `random_polygon(n, rng)`, `random_polytope(n, rng)` and `noisy_support(body, directions, sigma, rng)` (simulated measurements: h_K(u_i) plus independent N(0, σ²) noise). `rng` is an int, a `np.random.Generator` or `None` everywhere, so nothing uses numpy's global random state.
- `metrics.py`: `hausdorff_distance` (max over directions of |h_K − h_L|) and `l2_distance` (root mean square over the sphere, the uniform probability measure), both on 20,000 evenly spread directions in 2D and 50,000 in 3D.

`examples/` (not in the wheel, no `__init__.py`) holds one script per figure (except the static `support_function.png`), the four modules they share and `figures/`. The scripts import the shared modules as siblings, and `src` as the installed package.

- `style.py`: the minimal figure style every script draws in, `MINIMAL` (matplotlib rc settings for `plt.rc_context`), and the colours by role: `TRUTH`, `TRUTH_3D`, `MEASUREMENT`, `OUTLIER`, `LSQ`, `LAD`, `L2_ERROR`, `HAUSDORFF_ERROR`. No matplotlib.
- `draw.py`: helpers that draw on the axes they are given, so no matplotlib import: `setup_2d`, `draw_body`, `draw_estimate`, `draw_measurements` (each a short piece of its supporting line) and `draw_polytope_3d` (flat shading, back faces culled, optional facet edges).
- `results.py`: `save_results` / `load_results` (a dict of arrays as a `.npz` file, no pickle), `load_or_compute(path, compute, recompute)` and `parse_args` (the `--recompute` flag). No matplotlib.
- `shapes.py`: the bodies the scripts reconstruct, `rounded_heptagon()` and `rounded_polytope()`. No matplotlib.
- The scripts: `animation_2d.py` (the README's GIF, k = 5 to 400 beside the errors against k; it pauses, fades the measurements out through `draw_measurements(fade=...)` and holds the true body and the estimate), `outliers.py` (least squares and least absolute deviations overlaid on one panel with 10% scattered outliers), `reconstruction_3d.py` (k = 60 and 600) and `noise.py` (median L2 and Hausdorff errors of both methods against σ with k = 400). The last two have a `compute()` that fits and returns the raw arrays, a `plot(results)` that draws only from those, and a `main()` that joins them through `results.load_or_compute`.
