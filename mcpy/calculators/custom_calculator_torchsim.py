# -*- coding: utf-8 -*-
"""TorchSim calculator adapter with batched energy evaluation for mcpy.

The key method for batched Monte Carlo is ``get_potential_energies``: a list
of ASE ``Atoms`` objects is packed into one TorchSim ``SimState`` and evaluated
with one model call.  ``get_potential_energy`` is kept for compatibility with
ordinary (serial) mcpy ensembles.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from torchsim_mace_d3 import AdditiveModel


class CustomCalculatorTorchSim:
    """Adapter for a TorchSim MACE + D3 model used by mcpy.

    Parameters
    ----------
    model_mace, model_d3
        TorchSim-compatible model objects combined by ``AdditiveModel``.
    device
        Torch device, typically ``torch.device('cuda')``.
    dtype
        Torch dtype, typically ``torch.float32`` or ``torch.float64``.
    """

    def __init__(self, model_mace, model_d3, device, dtype) -> None:
        self.model_mace = model_mace
        self.model_d3 = model_d3
        self.model = AdditiveModel(model_mace, model_d3)
        self.device = device
        self.dtype = dtype

    def get_potential_energies(self, atoms_list: Sequence[Atoms]) -> np.ndarray:
        """Return one potential energy (eV) per structure in ``atoms_list``.

        TorchSim supports variable-size systems in the same batch.  This is
        important for GCMC because different chains can have different atom
        counts after insertion/deletion moves.
        """
        atoms_list = list(atoms_list)
        if not atoms_list:
            return np.empty(0, dtype=float)

        state = ts.initialize_state(
            atoms_list,
            device=self.device,
            dtype=self.dtype,
        )

        # Energy-only Monte Carlo: no forces, stress, or backward derivatives
        # are needed here.  Whether the wrapped models internally compute extra
        # quantities depends on their own configuration.
        with torch.inference_mode():
            model_outputs = self.model(state)
            energies = model_outputs["energy"]

        # TorchSim defines energy as a systemwise property.  Refuse to silently
        # sum if a wrapper unexpectedly returns a different shape, because that
        # would mix energies from independent MC chains.
        energies = torch.as_tensor(energies).reshape(-1)
        if energies.numel() != len(atoms_list):
            raise RuntimeError(
                "Expected one systemwise energy per structure, but got "
                f"{energies.numel()} values for {len(atoms_list)} structures. "
                "Check the MACE+D3 AdditiveModel output semantics."
            )

        # One device -> host synchronization for the whole batch.
        return energies.detach().cpu().numpy().astype(float, copy=False)

    def get_potential_energy(self, atoms: Atoms) -> float:
        """Serial compatibility method expected by mcpy ``BaseEnsemble``."""
        return float(self.get_potential_energies([atoms])[0])
