"""Helpers to construct independent mcpy GCMC chains safely."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble


def build_independent_gcmc_chains(
    *,
    initial_atoms: Atoms,
    cells: Sequence[Any],
    calculator: Any,
    move_selector_factory: Callable[[int, int], Any],
    units_type: str,
    mu: dict[str, float],
    species: list[str],
    temperature: float,
    n_chains: int,
    output_dir: str | Path = "batched_output",
    base_seed: int = 12345,
    molecules: Optional[dict[str, Atoms]] = None,
    trajectory_write_interval: int = 1000,
    outfile_write_interval: int = 100,
    minima: bool = False,
) -> list[GrandCanonicalEnsemble]:
    """Build chains with independent mutable state and one shared calculator.

    ``move_selector_factory(chain_index, chain_seed)`` must create a fresh
    MoveSelector and fresh move objects every time it is called.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chains: list[GrandCanonicalEnsemble] = []
    for i in range(n_chains):
        chain_seed = base_seed + 1000 * i
        atoms_i = initial_atoms.copy()

        # Cells carry calculated volumes, so do not share them between chains.
        cells_i = copy.deepcopy(list(cells))
        selector_i = move_selector_factory(i, chain_seed)

        chain_dir = output_dir / f"chain_{i:03d}"
        chain_dir.mkdir(parents=True, exist_ok=True)

        chain = GrandCanonicalEnsemble(
            atoms=atoms_i,
            cells=cells_i,
            units_type=units_type,
            calculator=calculator,
            mu=dict(mu),
            species=list(species),
            temperature=temperature,
            move_selector=selector_i,
            molecules=molecules,
            random_seed=chain_seed,
            traj_file=str(chain_dir / "trajectory.xyz"),
            trajectory_write_interval=trajectory_write_interval,
            outfile=str(chain_dir / "outfile.out"),
            outfile_write_interval=outfile_write_interval,
            minima_file=(str(chain_dir / "minima.xyz") if minima else None),
        )
        chains.append(chain)

    return chains
