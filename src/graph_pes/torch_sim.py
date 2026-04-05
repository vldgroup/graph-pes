from __future__ import annotations

import traceback
import warnings
from pathlib import Path
from typing import Any

import torch

from graph_pes import AtomicGraph, GraphPESModel
from graph_pes.atomic_graph import PropertyKey
from graph_pes.models import load_model


try:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
    from torch_sim.state import SimState

except ImportError as exc:
    _torch_sim_import_error = exc
    warnings.warn(
        f"torch-sim import failed: {traceback.format_exc()}",
        stacklevel=2,
    )

    class GraphPESWrapper(torch.nn.Module):
        """Placeholder raised when torch-sim is unavailable."""

        def __init__(
            self,
            err: ImportError = _torch_sim_import_error,
            *_args: Any,
            **_kwargs: Any,
        ) -> None:
            super().__init__()
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError

else:

    def _state_to_atomic_graph(state: SimState, cutoff: torch.Tensor) -> AtomicGraph:
        # graph-pes models internally trim the neighbor list to the model cutoff.
        # Bump it slightly here to avoid exact-cutoff inclusion edge cases.
        nl, _system_mapping, shifts = torchsim_nl(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            cutoff + 1e-5,
            state.system_idx,
        )
        n_atoms_per_system = torch.bincount(state.system_idx)
        ptr = torch.zeros(state.n_systems + 1, dtype=torch.long, device=state.device)
        ptr[1:] = n_atoms_per_system.cumsum(dim=0)
        n_sys = state.n_systems
        total_charge = torch.zeros(n_sys, device=state.device)
        total_spin = torch.zeros(n_sys, device=state.device)
        return AtomicGraph(
            Z=state.atomic_numbers.long(),
            R=state.positions,
            cell=state.row_vector_cell,
            neighbour_list=nl.long(),
            neighbour_cell_offsets=shifts,
            properties={},
            cutoff=cutoff.item(),
            other={
                "total_charge": total_charge,
                "total_spin": total_spin,
            },
            batch=state.system_idx,
            ptr=ptr,
        )

    class GraphPESWrapper(ModelInterface):
        """Wrap a GraphPES model for use with torch-sim."""

        def __init__(
            self,
            model: GraphPESModel | str | Path,
            device: torch.device | None = None,
            dtype: torch.dtype = torch.float64,
            *,
            compute_forces: bool = True,
            compute_stress: bool = True,
        ) -> None:
            super().__init__()
            self._device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._dtype = dtype

            _model = model if isinstance(model, GraphPESModel) else load_model(model)
            self._gp_model = _model.to(device=self.device, dtype=self.dtype)

            self._compute_forces = compute_forces
            self._compute_stress = compute_stress

            self._properties: list[PropertyKey] = ["energy"]
            if self.compute_forces:
                self._properties.append("forces")
            if self.compute_stress:
                self._properties.append("stress")

            cutoff_val = self._gp_model.cutoff
            if isinstance(cutoff_val, torch.Tensor) and cutoff_val.item() < 0.5:
                self._memory_scales_with = "n_atoms"

        def forward(
            self, state: SimState, **_kwargs: object
        ) -> dict[str, torch.Tensor]:
            cutoff = self._gp_model.cutoff
            if not isinstance(cutoff, torch.Tensor):
                raise TypeError("GraphPES model cutoff must be a tensor")

            atomic_graph = _state_to_atomic_graph(state, cutoff)
            preds = self._gp_model.predict(atomic_graph, self._properties)
            return {k: v.detach() for k, v in preds.items()}


__all__ = ["GraphPESWrapper"]
