from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.models import GRACE
from graph_pes.models.components.distances import Chebyshev
from graph_pes.models.e3nn.grace import (
    ChannelMixer,
    ChannelwiseProduct,
    FunctionReduce,
    natural_parity_irreps,
)

CUTOFF = 3.0

SMALL = dict(
    n_rad_max=6,
    embedding_size=8,
    n_mlp_dens=4,
    readout_width=8,
    radial_hidden_layers=[16],
    indicator_channels=4,
)


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def _rotation(improper: bool = False) -> torch.Tensor:
    R = o3.rand_matrix()
    return -R if improper else R


def _D(irreps: list[o3.Irrep], R: torch.Tensor) -> torch.Tensor:
    return o3.Irreps([(1, ir) for ir in irreps]).D_from_matrix(R)


def _rotate(x: torch.Tensor, irreps, R: torch.Tensor) -> torch.Tensor:
    # x: (N, C, D)
    return x @ _D(irreps, R).T


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def test_chebyshev():
    cheb = Chebyshev(n_features=6, cutoff=4.0)
    r = torch.linspace(0, 4.0, 17)
    actual = cheb(r)
    x = (2 * r / 4.0 - 1).numpy()
    expected = np.polynomial.chebyshev.chebvander(x, 6)[:, 1:]
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10)


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_channelwise_product_equivariance(symmetric: bool, improper: bool):
    left = natural_parity_irreps(3)
    right = left if symmetric else natural_parity_irreps(2)
    product = ChannelwiseProduct(
        left, right, lmax=3, Lmax=3, symmetric=symmetric
    )

    # all outputs have natural parity
    assert all(ir.p == (-1) ** ir.l for ir in product.irreps_out)

    x = torch.randn(5, 4, sum(ir.dim for ir in left))
    y = x if symmetric else torch.randn(5, 4, sum(ir.dim for ir in right))
    R = _rotation(improper)

    out = product(x, y)
    rotated_out = product(_rotate(x, left, R), _rotate(y, right, R))
    torch.testing.assert_close(rotated_out, _rotate(out, product.irreps_out, R))


def test_symmetric_product_has_no_duplicate_paths():
    irreps = natural_parity_irreps(1)  # 0e + 1o
    full = ChannelwiseProduct(irreps, irreps, lmax=1, Lmax=2)
    sym = ChannelwiseProduct(irreps, irreps, lmax=1, Lmax=2, symmetric=True)
    # 0x0->0, 0x1->1, 1x0->1, 1x1->0, 1x1->2
    assert len(full.irreps_out) == 5
    # (1x0 and 0x1 are equivalent)
    assert len(sym.irreps_out) == 4


def test_outer_coupling_matches_direct_product():
    left = natural_parity_irreps(2)
    right = natural_parity_irreps(1)
    product = ChannelwiseProduct(left, right, lmax=2, Lmax=2)
    x = torch.randn(3, 2, 9)
    y = torch.randn(3, 2, 4)
    outer = (x.unsqueeze(-1) * y.unsqueeze(-2)).reshape(3, 2, -1)
    torch.testing.assert_close(product.couple_outer(outer), product(x, y))


def test_channel_mixer_equivariance():
    left = natural_parity_irreps(2) * 2
    right = natural_parity_irreps(1)
    mixer = ChannelMixer(left, 4, right, 3, 5)
    x = torch.randn(7, 4, sum(ir.dim for ir in left))
    y = torch.randn(7, 3, sum(ir.dim for ir in right))
    R = _rotation()

    out = mixer(x, y)
    assert out.shape == (7, 5, x.shape[-1])
    rotated = mixer(_rotate(x, left, R), _rotate(y, right, R))
    torch.testing.assert_close(rotated, _rotate(out, left, R))


@pytest.mark.parametrize("element_dependent", [False, True])
def test_function_reduce(element_dependent: bool):
    a = natural_parity_irreps(3)
    b = natural_parity_irreps(2) * 3
    reduce = FunctionReduce(
        [(a, 4, 1), (b, 2, 2)],
        natural_parity_irreps(2),
        out_channels=3,
        n_elements=2 if element_dependent else None,
    )
    # 2e only comes from b
    assert reduce.irreps_out == natural_parity_irreps(2)

    xa = torch.randn(6, 4, 16)
    xb = torch.randn(6, 2, 27)
    elements = torch.tensor([0, 1, 0, 1, 1, 0])
    R = _rotation()

    out = reduce([xa, xb], elements)
    assert out.shape == (6, 3, 9)
    rotated = reduce([_rotate(xa, a, R), _rotate(xb, b, R)], elements)
    torch.testing.assert_close(rotated, _rotate(out, reduce.irreps_out, R))

    if element_dependent:
        # swapping element labels should change the output
        assert not torch.allclose(out, reduce([xa, xb], 1 - elements))


# ---------------------------------------------------------------------------
# the full model
# ---------------------------------------------------------------------------


def _model(**kwargs) -> GRACE:
    return GRACE(elements=["H", "C", "O"], cutoff=CUTOFF, **{**SMALL, **kwargs})


MODEL_CONFIGS = {
    "1L-order1": dict(layers=1, max_order=1),
    "1L-order4": dict(layers=1, max_order=4, lmax=2),
    "2L": dict(layers=2, lmax=[2, 2], max_order=3),
    "2L-wide-products": dict(
        layers=2, lmax=[3, 1], n_rad_max=[6, 4], prod_func_n_max=[5, 3]
    ),
}


@pytest.mark.parametrize("config", MODEL_CONFIGS.values(), ids=MODEL_CONFIGS)
@pytest.mark.parametrize("improper", [False, True])
def test_model_invariance(config: dict, improper: bool):
    model = _model(**config)
    atoms = molecule("CH3CH2OH")
    graph = AtomicGraph.from_ase(atoms, cutoff=CUTOFF)
    preds = model.predict(graph, properties=["energy", "forces"])

    R = _rotation(improper).numpy()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ R.T
    graph_r = AtomicGraph.from_ase(rotated, cutoff=CUTOFF)
    preds_r = model.predict(graph_r, properties=["energy", "forces"])

    torch.testing.assert_close(preds["energy"], preds_r["energy"])
    torch.testing.assert_close(
        preds["forces"] @ torch.tensor(R).T, preds_r["forces"]
    )


def test_smooth_at_cutoff():
    model = _model(layers=2, max_order=3)

    def dimer_energy(r: float) -> float:
        atoms = Atoms("CO", positions=[[0, 0, 0], [r, 0, 0]])
        graph = AtomicGraph.from_ase(atoms, cutoff=CUTOFF)
        return model.predict(graph, properties=["energy"])["energy"].item()

    isolated = dimer_energy(CUTOFF + 0.5)
    for r in [CUTOFF - 1e-4, CUTOFF - 1e-3]:
        assert dimer_energy(r) == pytest.approx(isolated, abs=1e-6)
    # and the dimer energy should differ from the isolated atoms inside
    # the cutoff
    assert dimer_energy(CUTOFF * 0.5) != pytest.approx(isolated, abs=1e-6)


@pytest.mark.parametrize("layers", [1, 2])
def test_receptive_field(layers: int):
    # a linear chain: atom 0 and atom 2 are within 2 * cutoff, but not within
    # cutoff of each other. GRACE-1L is local, so moving atom 2 should
    # not change the local energy of atom 0. GRACE-2L is semi-local,
    # so it should.
    model = _model(layers=layers, max_order=3)

    def local_energy_0(x2: float) -> float:
        atoms = Atoms(
            "CCC", positions=[[0, 0, 0], [0.8 * CUTOFF, 0, 0], [x2, 0, 0.3]]
        )
        graph = AtomicGraph.from_ase(atoms, cutoff=CUTOFF)
        return model.predict(graph, properties=["local_energies"])[
            "local_energies"
        ][0].item()

    a, b = local_energy_0(1.6 * CUTOFF), local_energy_0(1.5 * CUTOFF)
    if layers == 1:
        assert a == pytest.approx(b, abs=1e-10)
    else:
        assert a != pytest.approx(b, abs=1e-8)


def test_bad_arguments():
    with pytest.raises(ValueError, match="1 or 2 layers"):
        _model(layers=3)
    with pytest.raises(ValueError, match="Expected 2 values for lmax"):
        _model(layers=2, lmax=[2, 2, 2])
    with pytest.raises(ValueError, match="max_order"):
        _model(max_order=5)
    with pytest.raises(ValueError, match="indicator_lmax"):
        _model(layers=2, lmax=[1, 1], indicator_lmax=2)

    model = _model()
    graph = AtomicGraph.from_ase(molecule("NH3"), cutoff=CUTOFF)
    with pytest.raises(ValueError, match="Unknown elements"):
        model.predict(graph, properties=["energy"])
