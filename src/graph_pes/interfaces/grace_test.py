"""
Tests for the GRACE foundation-model interface.

The spherical harmonic tests are self-contained. The model tests need the
``<model>-kokkos.npz`` files: they are looked for in the directory given by
the ``GRAPH_PES_GRACE_FM_DIR`` environment variable (defaulting to the
``graph-pes`` cache), and skipped if not present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from scipy.spatial.transform import Rotation

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.interfaces._grace import (
    GRACEFoundationModel,
    TPSphericalHarmonics,
    _cache_dir,
    grace_fm,
)
from graph_pes.models import ScriptedModel

LMAX = 4
MODELS = ["GRACE-2L-SMAX-OMAT-medium", "GRACE-2L-OMAT-large-ft-E"]


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# ---------------------------------------------------------------------------
# spherical harmonics
# ---------------------------------------------------------------------------


def _sphere_quadrature(n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre (in cos theta) x uniform (in phi) quadrature."""
    x, w = np.polynomial.legendre.leggauss(n)
    phi = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    cos_t, ph = np.meshgrid(x, phi, indexing="ij")
    sin_t = np.sqrt(1 - cos_t**2)
    points = np.stack(
        [sin_t * np.cos(ph), sin_t * np.sin(ph), cos_t], axis=-1
    ).reshape(-1, 3)
    weights = np.repeat(w, 2 * n) * (2 * np.pi / (2 * n)) / (4 * np.pi)
    return torch.tensor(points), torch.tensor(weights)


def test_spherical_harmonics_are_orthonormal():
    points, weights = _sphere_quadrature()
    Y = TPSphericalHarmonics(LMAX)(points)  # (P, 25)
    gram = Y.T @ (Y * weights.unsqueeze(-1))
    torch.testing.assert_close(gram, torch.eye((LMAX + 1) ** 2))


def test_spherical_harmonics_rotate_within_each_l():
    points, _ = _sphere_quadrature()
    R = torch.tensor(Rotation.random(random_state=1).as_matrix())
    sh = TPSphericalHarmonics(LMAX)
    Y, Y_rot = sh(points), sh(points @ R.T)
    for l in range(LMAX + 1):
        block = slice(l * l, (l + 1) ** 2)
        # Y(R r) = D(R) Y(r) for an orthogonal D within each l block
        D = torch.linalg.lstsq(Y[:, block], Y_rot[:, block]).solution
        torch.testing.assert_close(Y[:, block] @ D, Y_rot[:, block])
        torch.testing.assert_close(D @ D.T, torch.eye(2 * l + 1))


def test_spherical_harmonics_known_values():
    # sqrt(4 pi) * real Y_lm: Y_00 = 1, (Y_1,-1, Y_10, Y_11) = sqrt(3) (y, z, x)
    r = torch.tensor([[0.48, -0.6, 0.64]])
    Y = TPSphericalHarmonics(1)(r)[0]
    s3 = 3**0.5
    torch.testing.assert_close(
        Y, torch.tensor([1.0, s3 * -0.6, s3 * 0.64, s3 * 0.48])
    )


# ---------------------------------------------------------------------------
# the foundation models
# ---------------------------------------------------------------------------


def _npz_path(name: str) -> Path | None:
    root = os.environ.get("GRAPH_PES_GRACE_FM_DIR")
    path = (Path(root) if root else _cache_dir()) / f"{name}-kokkos.npz"
    return path if path.exists() else None


@pytest.fixture(params=MODELS, scope="module")
def model(request) -> GRACEFoundationModel:
    path = _npz_path(request.param)
    if path is None:
        pytest.skip(f"{request.param}-kokkos.npz not available")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return grace_fm(request.param, path=path)
    finally:
        torch.set_default_dtype(old)


def _graph(atoms: Atoms, model: GRACEFoundationModel) -> AtomicGraph:
    return AtomicGraph.from_ase(atoms, model.cutoff.item())


def _alloy() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.7, cubic=True).repeat(2)
    atoms.symbols[::3] = "Au"
    atoms.symbols[1] = "O"
    atoms.rattle(0.1, seed=0)
    return atoms


def test_is_eval_and_double(model):
    assert not model.training
    assert model.embeddings.dtype == torch.float64


@pytest.mark.parametrize("improper", [False, True])
def test_invariance(model, improper):
    atoms = _alloy()
    p = model.predict(_graph(atoms, model), properties=["energy", "forces"])

    R = Rotation.random(random_state=2).as_matrix() * (-1 if improper else 1)
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ R.T
    rotated.cell = atoms.cell @ R.T
    p_rot = model.predict(_graph(rotated, model), ["energy", "forces"])

    torch.testing.assert_close(p["energy"], p_rot["energy"])
    torch.testing.assert_close(p["forces"] @ torch.tensor(R).T, p_rot["forces"])


def test_forces_are_energy_gradients(model):
    atoms = molecule("CH3CH2OH")
    forces = model.predict(_graph(atoms, model), ["forces"])["forces"]

    h, i, k = 1e-4, 2, 1
    energies = []
    for sign in (1, -1):
        displaced = atoms.copy()
        displaced.positions[i, k] += sign * h
        g = _graph(displaced, model)
        energies.append(model.predict(g, ["energy"])["energy"].item())
    fd = -(energies[0] - energies[1]) / (2 * h)
    assert forces[i, k].item() == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize(
    "atoms, reference",
    [
        (bulk("Si", "diamond", a=5.47), -5.42),
        (bulk("Al", "fcc", a=4.04), -3.74),
        (bulk("MgO", "rocksalt", a=4.25), -5.97),
    ],
    ids=["Si", "Al", "MgO"],
)
def test_energies_close_to_pbe(model, atoms, reference):
    # approximate VASP-PBE energies per atom (Materials Project)
    energy = model.predict(_graph(atoms, model), ["energy"])["energy"]
    assert energy.item() / len(atoms) == pytest.approx(reference, abs=0.06)


def _reference_structure(label: str) -> Atoms:
    if label == "Si-rattled":
        atoms = bulk("Si", "diamond", a=5.47, cubic=True)
        atoms.rattle(0.05, seed=1)
    elif label == "Pt111-CO":
        atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
        co = molecule("CO")
        co.positions += atoms.positions[-1] + [0, 0, 1.9]
        atoms += co
        atoms.rattle(0.03, seed=5)
    else:
        atoms = Atoms(
            "CuO", positions=[[0, 0, 0], [3.5, 0, 0]], cell=[20] * 3, pbc=True
        )
    return atoms


# energies (eV) and forces on atom 0 (eV/Å) computed with the original
# tensorpotential 0.6.2.1 implementation (TPCalculator, float64)
TENSORPOTENTIAL_REFERENCE = {
    "GRACE-2L-SMAX-OMAT-medium": {
        "Si-rattled": (
            -42.881485794237285,
            [-1.51419419, 0.4613765, -0.28620128],
        ),
        "Pt111-CO": (
            -67.29042977638886,
            [0.04689835, -0.16047503, -1.53382924],
        ),
        "CuO-dimer": (-2.708698898465101, [0.8305743, 0.0, 0.0]),
    },
    "GRACE-2L-OMAT-large-ft-E": {
        "Si-rattled": (
            -42.873561127158,
            [-1.48778399, 0.44890227, -0.29597297],
        ),
        "Pt111-CO": (-67.20325503613877, [0.03231032, -0.1582352, -1.43083114]),
        "CuO-dimer": (-2.4769316065842055, [0.73167333, 0.0, 0.0]),
    },
}


@pytest.mark.parametrize("label", ["Si-rattled", "Pt111-CO", "CuO-dimer"])
def test_matches_tensorpotential(model, label):
    energy, force = TENSORPOTENTIAL_REFERENCE[model.model_name][label]
    atoms = _reference_structure(label)
    p = model.predict(_graph(atoms, model), ["energy", "forces"])
    assert p["energy"].item() == pytest.approx(energy, abs=1e-6)
    np.testing.assert_allclose(p["forces"][0].numpy(), force, atol=1e-6)


def test_torchscript(model):
    g = _graph(_alloy(), model)
    eager = model.predict(g, ["energy", "forces", "stress"])
    scripted = ScriptedModel(torch.jit.script(model)).predict(
        g, ["energy", "forces", "stress"]
    )
    for key in eager:
        torch.testing.assert_close(eager[key], scripted[key])


def test_bond_specific_cutoffs_are_smooth(model):
    if not model.radial_basis.bond_specific:
        pytest.skip("model has a single, global cutoff")
    elements = model.elements
    i, j = elements.index("Cu"), elements.index("O")
    rc = model.radial_basis.bond_cutoff_map[i, j].item()
    assert rc < model.cutoff.item()

    def energy(r: float) -> float:
        atoms = Atoms("CuO", positions=[[0, 0, 0], [r, 0, 0]])
        g = _graph(atoms, model)
        return model.predict(g, ["energy"])["energy"].item()

    # the pair's contribution vanishes continuously at its own cutoff...
    outside = energy(rc + 1e-4)
    gaps = [abs(energy(rc - dr) - outside) for dr in (1e-3, 1e-4, 1e-5)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-8
    # ...and is exactly zero beyond it, despite being inside model.cutoff
    assert energy(rc + 0.5) == pytest.approx(outside, abs=1e-12)


def test_unknown_element(model):
    atoms = Atoms("CuOg", positions=[[0, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="does not support elements"):
        model.predict(_graph(atoms, model), ["energy"])


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown GRACE foundation model"):
        grace_fm("GRACE-7L-imaginary")
