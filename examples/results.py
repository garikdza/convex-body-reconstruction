"""
Saved results of the example scripts that take a while, so their figures can be redrawn without computing again.

A script computes its raw results once and saves them as a ``.npz`` file next to its figures. Later runs plot from
that file, and ``--recompute`` computes again.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np


def save_results(path: Path, results: Mapping[str, np.ndarray]) -> None:
    """
    Save arrays to a ``.npz`` file, creating its folder.

    Parameters
    ----------
    path : Path
        file to write, by convention ending in ``.npz``
    results : Mapping[str, np.ndarray]
        arrays to save by name; object arrays are rejected, so the file loads without pickle
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # write beside it and swap in, so a failed or interrupted save leaves no half-written file for the next run
    part = path.with_suffix('.part.npz')
    try:
        np.savez(part, allow_pickle=False, **results)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def load_results(path: Path) -> dict[str, np.ndarray]:
    """Load the arrays that ``save_results`` wrote to ``path``."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_or_compute(path: Path, compute: Callable[[], dict[str, np.ndarray]], recompute: bool) -> dict[str, np.ndarray]:
    """
    Load the results saved at ``path``, computing and saving them first when the file is missing or asked to.

    The results are always read back from the file, so a plot shows the same values on every run.

    Parameters
    ----------
    path : Path
        results file, ending in ``.npz``
    compute : Callable[[], dict[str, np.ndarray]]
        computes and returns the raw results
    recompute : bool
        compute even when the file exists, overwriting it

    Returns
    -------
    results : dict[str, np.ndarray]
        the saved arrays by name
    """
    if recompute or not path.exists():
        save_results(path, compute())
        print(f'wrote {path}')
    else:
        print(f'loaded {path} (run with --recompute to compute again)')
    return load_results(path)


def parse_args(description: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of an example script that saves its results: only ``--recompute``."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recompute', action='store_true', help='compute again and overwrite the saved results')
    return parser.parse_args(argv)
