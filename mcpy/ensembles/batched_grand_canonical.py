"""Batched coordinator for independent mcpy GrandCanonicalEnsemble chains.

This module intentionally does *not* patch mcpy itself.  Each chain is a normal
``GrandCanonicalEnsemble`` with its own Atoms, MoveSelector, cells, RNGs, and
statistics.  The coordinator splits each trial into:

    propose on CPU -> batch all viable structures -> one GPU model call
    -> independent Metropolis decisions

That preserves the serial mcpy acceptance logic while amortizing TorchSim/MACE
launch and state-processing overhead across independent chains.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Iterable, Optional

import numpy as np

try:
    from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
except ImportError as exc:  # clearer message when file is imported standalone
    raise ImportError(
        "batched_grand_canonical.py requires the mcpy package to be importable."
    ) from exc


logger = logging.getLogger(__name__)


@dataclass
class _TrialProposal:
    """Bookkeeping needed to finish one deferred GCMC trial."""

    chain: GrandCanonicalEnsemble
    saved_arrays: dict[str, np.ndarray]
    saved_constraints: list[Any]
    delta_particles: int
    species: str
    volume: float
    n_exchange: int


class BatchedGrandCanonicalRunner:
    """Run independent GCMC chains with batched energy evaluations.

    Notes
    -----
    * The chains must share the *same* calculator object.
    * Each chain must own an independent MoveSelector / move objects / RNGs.
    * Different atom counts across chains are supported if the calculator's
      ``get_potential_energies`` supports variable-size TorchSim batches.
    * ``MoveSelector.n_moves`` is preserved.  One batch is issued for each
      sub-move round, containing at most one proposal from each active chain.
    """

    def __init__(
        self,
        chains: Iterable[GrandCanonicalEnsemble],
        calculator: Optional[Any] = None,
    ) -> None:
        self.chains = list(chains)
        if not self.chains:
            raise ValueError("At least one GCMC chain is required.")

        if not all(isinstance(c, GrandCanonicalEnsemble) for c in self.chains):
            raise TypeError("All chains must be GrandCanonicalEnsemble instances.")

        if calculator is None:
            calculator = self.chains[0]._calculator
        self.calculator = calculator

        if not hasattr(self.calculator, "get_potential_energies"):
            raise TypeError(
                "The shared calculator must implement "
                "get_potential_energies(atoms_list)."
            )

        for i, chain in enumerate(self.chains):
            if chain._calculator is not self.calculator:
                raise ValueError(
                    f"Chain {i} does not reference the shared calculator object. "
                    "Use one calculator/model instance for all chains."
                )

        # Sharing move selectors would couple RNG/counter state between chains.
        selector_ids = [id(c.move_selector) for c in self.chains]
        if len(selector_ids) != len(set(selector_ids)):
            raise ValueError(
                "Each chain must have its own MoveSelector instance; at least "
                "two chains currently share one."
            )

        atom_ids = [id(c.atoms) for c in self.chains]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError(
                "Each chain must have its own ASE Atoms instance; use atoms.copy()."
            )

    @staticmethod
    def _restore(proposal: _TrialProposal) -> None:
        atoms = proposal.chain.atoms
        atoms.arrays = proposal.saved_arrays
        atoms.set_constraint(proposal.saved_constraints)

    @staticmethod
    def _propose(chain: GrandCanonicalEnsemble) -> Optional[_TrialProposal]:
        """Apply one mcpy trial move but defer the energy calculation."""
        atoms = chain.atoms

        # This mirrors current mcpy GCMC rollback semantics: snapshot all ASE
        # arrays and constraints because insert/delete/displacement mutate in place.
        saved_arrays = {k: v.copy() for k, v in atoms.arrays.items()}
        saved_constraints = [c.copy() for c in atoms.constraints]

        atoms_new, delta_particles, species = chain.move_selector.do_trial_move(atoms)

        if atoms_new is False or atoms_new is None:
            # A failed proposal is excluded from the acceptance denominator by
            # MoveSelector itself.  Restore defensively in case it mutated first.
            atoms.arrays = saved_arrays
            atoms.set_constraint(saved_constraints)
            return None

        if atoms_new is not atoms:
            atoms.arrays = saved_arrays
            atoms.set_constraint(saved_constraints)
            raise RuntimeError(
                f"move '{chain.move_selector.get_name()}' returned a different "
                "Atoms object; GCMC moves must mutate the passed object in place."
            )

        volume = chain.move_selector.get_volume()
        n_exchange = chain.move_selector.get_exchange_count()
        if n_exchange is None:
            # Important: use the pre-move accepted atom count, exactly as mcpy.
            n_exchange = chain.n_atoms

        return _TrialProposal(
            chain=chain,
            saved_arrays=saved_arrays,
            saved_constraints=saved_constraints,
            delta_particles=delta_particles,
            species=species,
            volume=volume,
            n_exchange=n_exchange,
        )

    @staticmethod
    def _finish(proposal: _TrialProposal, new_energy: float) -> bool:
        """Perform the original mcpy Metropolis decision for one proposal."""
        chain = proposal.chain
        atoms = chain.atoms
        delta_e = float(new_energy) - chain.E_old

        accepted = chain._acceptance_condition(
            delta_e,
            proposal.delta_particles,
            proposal.volume,
            proposal.species,
            proposal.n_exchange,
        )

        if accepted:
            if chain._wrap_on_accept:
                atoms.wrap()
            chain.n_atoms = len(atoms)
            chain.E_old = float(new_energy)
            chain.move_selector.acceptance_counter()
            chain.calculate_cells_volume(atoms)
            chain._record_minimum(atoms, chain.E_old)
            chain.logger.debug(
                "Volume: %.3f, Delta_particles: %d, Species: %s",
                proposal.volume,
                proposal.delta_particles,
                proposal.species,
            )
            return True

        BatchedGrandCanonicalRunner._restore(proposal)
        return False

    def _run_submove_round(self, active_chains: list[GrandCanonicalEnsemble]) -> None:
        proposals: list[_TrialProposal] = []
        for chain in active_chains:
            proposal = self._propose(chain)
            if proposal is not None:
                proposals.append(proposal)

        if not proposals:
            return

        try:
            energies = self.calculator.get_potential_energies(
                [proposal.chain.atoms for proposal in proposals]
            )
        except Exception:
            # Never leave chains in their mutated trial configurations if the
            # batched model call fails.
            for proposal in proposals:
                self._restore(proposal)
            raise

        if len(energies) != len(proposals):
            for proposal in proposals:
                self._restore(proposal)
            raise RuntimeError(
                f"Calculator returned {len(energies)} energies for "
                f"{len(proposals)} proposals."
            )

        for proposal, energy in zip(proposals, energies):
            self._finish(proposal, float(energy))

    def _run_one_outer_step(self) -> None:
        """Perform one mcpy GCMC step in every chain, synchronized by rounds."""
        t0 = time.perf_counter()
        max_n_moves = max(chain.move_selector.n_moves for chain in self.chains)

        for submove in range(max_n_moves):
            active = [
                chain
                for chain in self.chains
                if submove < chain.move_selector.n_moves
            ]
            self._run_submove_round(active)

        elapsed = time.perf_counter() - t0

        # Reproduce GrandCanonicalEnsemble._run() output/step behavior without
        # calling it (calling it would do serial energy evaluations again).
        for chain in self.chains:
            chain._step += 1
            chain._last_step_seconds = elapsed

            if chain._step % chain._outfile_write_interval == 0:
                chain.write_outfile()
                chain.logger.info(
                    "step=%d N=%d E=%.6f batched_step_wall=%.3fs",
                    chain._step,
                    chain.n_atoms,
                    chain.E_old,
                    elapsed,
                )

            if chain._step % chain._trajectory_write_interval == 0:
                chain.write_coordinates(chain.atoms, chain.E_old)

    def run(self, steps: int) -> None:
        """Run ``steps`` synchronized outer GCMC steps for all chains."""
        if steps < 0:
            raise ValueError("steps must be >= 0")

        initialized: list[GrandCanonicalEnsemble] = []
        try:
            for chain in self.chains:
                chain.initialize_run()
                initialized.append(chain)

            for _ in range(steps):
                self._run_one_outer_step()
        finally:
            # Normal mcpy finalization writes each chain's final row/frame and
            # reports cumulative move statistics.
            for chain in initialized:
                chain.finalize_run()
