"""Distances between convex bodies, computed from support functions.

For convex bodies ``K, L``:

* Hausdorff distance:  ``delta(K, L) = max_u |h_K(u) - h_L(u)|``
* L2 distance:         ``delta_2(K, L) = ( mean_u (h_K(u) - h_L(u))^2 )^(1/2)``

where ``u`` ranges over the unit sphere (the mean is with respect to the uniform
probability measure, i.e. the L2 norm used in the literature divided by
``sqrt(surface area of the sphere)``). Both are evaluated on a dense set of evenly spread
directions, so the Hausdorff value is a (tight) lower bound.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .directions import evenly_spread


class HasSupport(Protocol):
    def support(self, U: np.ndarray) -> np.ndarray: ...


def _dim(body: HasSupport) -> int:
    return int(body.dim)  # type: ignore[attr-defined]


def _diffs(K: HasSupport, L: HasSupport, n_directions: int | None) -> np.ndarray:
    d = _dim(K)
    if n_directions is None:
        n_directions = 20_000 if d == 2 else 50_000
    U = evenly_spread(n_directions, d)
    return K.support(U) - L.support(U)


def hausdorff_distance(K: HasSupport, L: HasSupport, n_directions: int | None = None) -> float:
    """Hausdorff distance between two convex bodies."""
    return float(np.abs(_diffs(K, L, n_directions)).max())


def l2_distance(K: HasSupport, L: HasSupport, n_directions: int | None = None) -> float:
    """L2 distance between the support functions of two convex bodies."""
    return float(np.sqrt(np.mean(_diffs(K, L, n_directions) ** 2)))
