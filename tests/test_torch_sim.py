import pytest
import torch
from ase.build import molecule

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.models import SchNet


torch_sim = pytest.importorskip("torch_sim")
from graph_pes.torch_sim import GraphPESWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def test_torch_sim_model_matches_direct_wrapper():
    atoms = molecule("H2O")
    atoms.center(vacuum=10.0)

    model = SchNet(cutoff=5.5)
    graph = AtomicGraph.from_ase(atoms, cutoff=5.5)
    direct_wrapper = GraphPESWrapper(
        model,
        device=DEVICE,
        dtype=DTYPE,
        compute_stress=False,
    )
    method_wrapper = model.torch_sim_model(
        device=DEVICE,
        dtype=DTYPE,
        compute_stress=False,
    )

    state = torch_sim.io.atoms_to_state([atoms], DEVICE, DTYPE)
    direct_output = direct_wrapper(state)
    method_output = method_wrapper(state)

    assert isinstance(method_wrapper, GraphPESWrapper)
    assert direct_output.keys() == method_output.keys()
    for key in direct_output:
        torch.testing.assert_close(direct_output[key], method_output[key])

    torch.testing.assert_close(
        method_output["energy"].cpu(),
        model.predict_energy(graph).reshape(1).cpu(),
    )
    torch.testing.assert_close(
        method_output["forces"].cpu(),
        model.predict_forces(graph).cpu(),
    )
