from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from graph_pes.atomic_graph import AtomicGraph, PropertyKey
from graph_pes.graph_pes_model import GraphPESModel

try:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
    from torch_sim.state import SimState

except ImportError as exc:
    _torch_sim_import_error = exc

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

    def _state_to_atomic_graph(
            state: SimState, 
            cutoff: torch.Tensor) -> AtomicGraph:
        # graph-pes models internally trim the neighbor list to the model cutoff
        # Bump it slightly here to avoid exact-cutoff inclusion edge cases.
        neighbour_list, _system_mapping, neighbour_cell_offsets = torchsim_nl(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            cutoff + 1e-5,
            state.system_idx,
        )
        n_atoms_per_system = torch.bincount(state.system_idx)
        ptr = torch.zeros(state.n_systems + 1, 
                          dtype=torch.long, 
                          device=state.device)
        ptr[1:] = n_atoms_per_system.cumsum(dim=0)
        n_systems = state.n_systems
        # TorchSim does not track per-system charge or spin, but AtomicGraph
        # reserves these slots for downstream interfaces that may expect them.
        total_charge = torch.zeros(n_systems, device=state.device)
        total_spin = torch.zeros(n_systems, device=state.device)
        return AtomicGraph(
            Z=state.atomic_numbers.long(),
            R=state.positions,
            cell=state.row_vector_cell,
            neighbour_list=neighbour_list.long(),
            neighbour_cell_offsets=neighbour_cell_offsets,
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

            if isinstance(model, GraphPESModel):
                _model = model
            else:
                from graph_pes.models import load_model

                _model = load_model(model)
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
