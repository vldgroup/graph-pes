"""
Pre-trained GRACE foundation models in ``graph-pes``.

The two-layer GRACE foundation models released by the ICAMS group (see
`gracemaker <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__)
are distributed, amongst other formats, as a ``kokkos.npz`` file for their
TensorFlow-free LAMMPS-Kokkos pair style. That file contains every learnable
weight together with all Clebsch-Gordan coupling tables, so we can rebuild the
model exactly in PyTorch without needing TensorFlow or ``tensorpotential``.

Every module below mirrors one ``tensorpotential`` instruction, keeps the
reference ``[atoms, channels, lm]`` layout, and copies its index tables
verbatim from the ``.npz`` file. The only convention that is *not* stored in
that file, the real spherical harmonics, is re-implemented here to exactly
match ``tensorpotential.functions.spherical_harmonics``.
"""

from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import torch

from graph_pes.atomic_graph import (
    AtomicGraph,
    PropertyKey,
    neighbour_distances,
    neighbour_vectors,
    number_of_atoms,
)
from graph_pes.graph_pes_model import GraphPESModel
from graph_pes.utils.logger import logger
from graph_pes.utils.misc import MAX_Z

_HF_KOKKOS_URL = (
    "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve/"
    "kk-v3-uq-v6/kokkos/{name}-kokkos.npz"
)

GRACE_2L_FOUNDATION_MODELS: Final[list[str]] = [
    "GRACE-2L-SMAX-OMAT-medium",
    "GRACE-2L-SMAX-OMAT-large",
    "GRACE-2L-SMAX-medium",
    "GRACE-2L-SMAX-large",
    "GRACE-2L-OMAT-medium-base",
    "GRACE-2L-OMAT-medium-ft-E",
    "GRACE-2L-OMAT-medium-ft-AM",
    "GRACE-2L-OMAT-large-base",
    "GRACE-2L-OMAT-large-ft-E",
    "GRACE-2L-OMAT-large-ft-AM",
]
"""Two-layer GRACE foundation models with the supported architecture."""

# the (fixed) topology of the GRACE-2L foundation models, as defined by
# the GRACE_2LAYER presets of tensorpotential and hard-coded in the
# reference LAMMPS-Kokkos pair style
_EXPECTED_NAMES: Final[dict[str, list[str]]] = {
    "mlp_rad_names": ["R", "R1"],
    "spb_names": ["A", "B0"],
    "fc_names": ["A1", "AA1", "AA2", "B1", "BB1", "BB2"],
    "prod_names": ["AA", "AAA", "AAAA", "BB", "BBB", "BBBB"],
    "reduce_names": ["I1", "I", "rho", "B", "I2"],
    "rms_names": ["I_nl_LN", "I_0_LN"],
}

Arrays = Mapping[str, np.ndarray]


def _t(x: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    elif t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    return t


def _long(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x).astype(np.int64).reshape(-1))


def _float(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _int(x: np.ndarray) -> int:
    return int(np.asarray(x).reshape(-1)[0])


def _param(x: np.ndarray) -> torch.nn.Parameter:
    return torch.nn.Parameter(_t(x))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class TPSphericalHarmonics(torch.nn.Module):
    r"""
    Real spherical harmonics, :math:`\sqrt{4\pi} Z_{lm}(\hat{r})`, following
    the exact conventions (ordering, phases and normalisation) of
    ``tensorpotential.functions.spherical_harmonics.SphericalHarmonics``.
    Output components are ordered as :math:`(l, m)` with
    :math:`m = -l, \dots, l` for :math:`l = 0, \dots, l_\text{max}`.
    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax: Final[int] = lmax
        alm: list[float] = [0.0]
        blm: list[float] = [0.0]
        for l in range(1, lmax + 1):
            lsq, ld = l * l, 2 * l
            l1, l2 = 4 * lsq - 1, lsq - ld + 1
            for m in range(0, l + 1):
                msq = m * m
                if l == m:
                    alm.append(-math.sqrt(1.0 + 0.5 / m))
                    blm.append(0.0)
                else:
                    alm.append(math.sqrt(l1 / (lsq - msq)))
                    blm.append(-math.sqrt((l2 - msq) / (4 * l2 - 1)))
        self.alm: list[float] = alm
        self.blm: list[float] = blm

    def _lm1d(self, l: int, m: int) -> int:
        return m + l * (l + 1) // 2

    def _lmsh(self, l: int, m: int) -> int:
        m = abs(m)
        return l + m * self.lmax - m * (m - 1) // 2

    def forward(self, rhat: torch.Tensor) -> torch.Tensor:
        x, y, z = rhat[:, 0], rhat[:, 1], rhat[:, 2]
        pi = math.pi

        # associated Legendre polynomials, indexed by lm1d(l, m)
        plm: list[torch.Tensor] = [torch.full_like(z, 1 / math.sqrt(4 * pi))]
        if self.lmax > 0:
            plm.append(math.sqrt(3 / (4 * pi)) * z)
            plm.append(torch.full_like(z, -math.sqrt(3 / (8 * pi))))
            for l in range(2, self.lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = math.sqrt(2.0 * m + 3.0)
                        plm.append(z * dl * plm[self._lm1d(l - 1, l - 1)])
                    elif m == l:
                        a = self.alm[self._lm1d(l, l)]
                        plm.append(a * plm[self._lm1d(l - 1, l - 1)])
                    else:
                        a = self.alm[self._lm1d(l, m)]
                        b = self.blm[self._lm1d(l, m)]
                        plm.append(
                            a
                            * (
                                z * plm[self._lm1d(l - 1, m)]
                                + b * plm[self._lm1d(l - 2, m)]
                            )
                        )

        # complex harmonics (real and imaginary parts), indexed by lmsh(l, m)
        ylm_r: list[torch.Tensor] = []
        ylm_i: list[torch.Tensor] = []
        for l in range(0, self.lmax + 1):
            ylm_r.append(plm[self._lm1d(l, 0)])
            ylm_i.append(torch.zeros_like(z))
        for l in range(1, self.lmax + 1):
            ylm_r.append(x * plm[self._lm1d(l, 1)])
            ylm_i.append(y * plm[self._lm1d(l, 1)])
        phase_r, phase_i = x, y
        for m in range(2, self.lmax + 1):
            new_r = phase_r * x - phase_i * y
            new_i = phase_r * y + phase_i * x
            phase_r, phase_i = new_r, new_i
            for l in range(m, self.lmax + 1):
                ylm_r.append(phase_r * plm[self._lm1d(l, m)])
                ylm_i.append(phase_i * plm[self._lm1d(l, m)])

        # real harmonics
        sqrt2 = math.sqrt(2.0)
        zlm: list[torch.Tensor] = []
        for l in range(0, self.lmax + 1):
            for m in range(-l, 0):
                ph = 1.0 - 2.0 * (m % 2)
                zlm.append(sqrt2 * ph * ylm_i[self._lmsh(l, m)])
            zlm.append(ylm_r[self._lmsh(l, 0)])
            for m in range(1, l + 1):
                ph = 1.0 - 2.0 * (m % 2)
                zlm.append(sqrt2 * ph * ylm_r[self._lmsh(l, m)])

        return torch.stack(zlm, dim=1) * math.sqrt(4 * pi)


def _cutoff_poly(x: torch.Tensor, p: int) -> torch.Tensor:
    return (
        1
        - (p + 1) * (p + 2) / 2 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2 * x ** (p + 2)
    )


class ChebyshevRadialBasis(torch.nn.Module):
    r"""
    The ``Cheb`` radial basis of ``tensorpotential``:
    :math:`T_k(2r/r_c - 1) \, f_p(r/r_c)` for :math:`k = 1, \dots, n`, with
    either a single cutoff or a per-element-pair cutoff,
    :math:`r_c(\mu_i, \mu_j)`.
    """

    def __init__(
        self,
        nfunc: int,
        p: int,
        rcut: float,
        bond_cutoff_map: torch.Tensor | None,
    ):
        super().__init__()
        self.nfunc: Final[int] = nfunc
        self.p: Final[int] = p
        self.rcut: Final[float] = rcut
        self.bond_specific: Final[bool] = bond_cutoff_map is not None
        if bond_cutoff_map is None:
            bond_cutoff_map = torch.zeros(1, 1)
        self.bond_cutoff_map: torch.Tensor
        self.register_buffer("bond_cutoff_map", bond_cutoff_map)

    def forward(
        self, d: torch.Tensor, mu_i: torch.Tensor, mu_j: torch.Tensor
    ) -> torch.Tensor:
        d = d.unsqueeze(-1)  # (E, 1)
        if self.bond_specific:
            rc = self.bond_cutoff_map[mu_i, mu_j].unsqueeze(-1)
            outside = d >= rc
        else:
            d = torch.where(d == 0.0, d + 1e-10, d)
            rc = torch.full_like(d, self.rcut)
            outside = d > rc

        x = d / rc
        t = 2.0 * (1.0 - torch.abs(1.0 - x)) - 1.0
        terms: list[torch.Tensor] = []
        t_prev = torch.ones_like(t)
        t_curr = t
        terms.append(t_curr)
        for _ in range(self.nfunc - 1):
            t_next = 2 * t * t_curr - t_prev
            terms.append(t_next)
            t_prev, t_curr = t_curr, t_next
        basis = torch.cat(terms, dim=-1) * _cutoff_poly(x, self.p)
        return torch.where(outside, torch.zeros_like(basis), basis)


# ---------------------------------------------------------------------------
# tensorpotential instructions
# ---------------------------------------------------------------------------


class MLPRadialFunction(torch.nn.Module):
    """``MLPRadialFunction_v2``: bias-free MLP onto ``R_{nl}``, tiled over m."""

    def __init__(self, arrays: Arrays, name: str, activation: str = "silu"):
        super().__init__()
        prefix = f"mlp_rad_{name}"
        if _int(arrays[f"{prefix}_has_chem_emb"]):
            raise NotImplementedError(
                "Chemically-embedded radial functions are not supported"
            )
        if activation != "silu":
            raise NotImplementedError(f"Unsupported activation {activation}")
        n_layers = _int(arrays[f"{prefix}_n_layers"])
        self.weights = torch.nn.ParameterList(
            [_param(arrays[f"{prefix}_W{i}"]) for i in range(n_layers)]
        )
        self.norms: list[float] = [
            _float(arrays[f"{prefix}_norm{i}"]) for i in range(n_layers)
        ]
        self.n_rad_max: Final[int] = _int(arrays[f"{prefix}_n_rad_max"])
        self.lmax: Final[int] = _int(arrays[f"{prefix}_lmax"])
        tile = [l for l in range(self.lmax + 1) for _ in range(2 * l + 1)]
        self.l_tile: torch.Tensor
        self.register_buffer("l_tile", torch.tensor(tile, dtype=torch.long))

    def forward(self, basis: torch.Tensor) -> torch.Tensor:
        x = basis
        n = len(self.norms)
        for i, w in enumerate(self.weights):
            x = x @ (w * self.norms[i])
            if i < n - 1:
                x = torch.nn.functional.silu(x)
        x = x.view(-1, self.n_rad_max, self.lmax + 1)
        return x.index_select(-1, self.l_tile)  # (E, n, (lmax+1)^2)


class ScalarSingleParticleBasis(torch.nn.Module):
    """``SingleParticleBasisFunctionScalarInd`` with a chemical indicator."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        if _int(arrays[f"{name}_indicator_l_depend"]):
            raise NotImplementedError("l-dependent indicators not supported")
        self.lin_w = _param(arrays[f"{name}_lin_transform_W"])
        self.lin_norm: Final[float] = _float(
            arrays[f"{name}_lin_transform_norm"]
        )
        self.inv_avg_n_neigh: Final[float] = _float(
            arrays[f"{name}_inv_avg_n_neigh"]
        )

    def forward(
        self,
        radial: torch.Tensor,  # (E, n, D)
        sph: torch.Tensor,  # (E, D_Y)
        embeddings: torch.Tensor,  # (n_elements, embedding_size)
        mu_j: torch.Tensor,
        ind_i: torch.Tensor,
        n_atoms: int,
    ) -> torch.Tensor:
        D = radial.shape[-1]
        chem = (embeddings @ (self.lin_w * self.lin_norm))[mu_j]  # (E, n)
        a = radial * sph[:, :D].unsqueeze(1) * chem.unsqueeze(-1)
        out = torch.zeros(
            (n_atoms, a.shape[1], D), dtype=a.dtype, device=a.device
        )
        return out.index_add(0, ind_i, a) * self.inv_avg_n_neigh


class EquivariantSingleParticleBasis(torch.nn.Module):
    """``SingleParticleBasisFunctionEquivariantInd``."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        lr = np.asarray(arrays[f"{name}_lr_inds"]).astype(np.int64)
        self.left: torch.Tensor
        self.right: torch.Tensor
        self.m_sum_ind: torch.Tensor
        self.cg: torch.Tensor
        self.register_buffer("left", torch.from_numpy(lr[:, 0].copy()))
        self.register_buffer("right", torch.from_numpy(lr[:, 1].copy()))
        self.register_buffer("m_sum_ind", _long(arrays[f"{name}_m_sum_ind"]))
        self.register_buffer("cg", _t(arrays[f"{name}_cg_coeff"]))
        self.n_out: Final[int] = _int(arrays[f"{name}_n_output_funcs"])
        self.inv_avg_n_neigh: Final[float] = _float(
            arrays[f"{name}_inv_avg_n_neigh"]
        )
        self.max_chunk_elements: int = 2**24

    def forward(
        self,
        radial: torch.Tensor,  # (E, n, D_Y)
        sph: torch.Tensor,  # (E, D_Y)
        indicator: torch.Tensor,  # (N, n, D_I)
        ind_i: torch.Tensor,
        ind_j: torch.Tensor,
        n_atoms: int,
    ) -> torch.Tensor:
        E, n, D_Y = radial.shape
        D_I = indicator.shape[-1]
        ry = radial * sph[:, :D_Y].unsqueeze(1)  # (E, n, D_Y)
        neighbour_indicator = indicator[ind_j]  # (E, n, D_I)

        # aggregate the per-edge outer products over neighbours, in chunks
        # of edges, so that the full (E, n, D_Y * D_I) tensor (and its
        # gradient) is never held in memory at once
        agg = torch.zeros(
            (n_atoms, n, D_Y * D_I), dtype=ry.dtype, device=ry.device
        )
        chunk = max(1, self.max_chunk_elements // max(1, n * D_Y * D_I))
        ry_chunks = torch.split(ry, chunk)
        ind_chunks = torch.split(neighbour_indicator, chunk)
        i_chunks = torch.split(ind_i, chunk)
        for c in range(len(ry_chunks)):
            outer = ry_chunks[c].unsqueeze(-1) * ind_chunks[c].unsqueeze(-2)
            outer = outer.reshape(outer.shape[0], n, D_Y * D_I)
            agg = agg.index_add(0, i_chunks[c], outer)
        agg = agg * self.inv_avg_n_neigh
        prod = agg.index_select(-1, self.left * D_I + self.right) * self.cg
        out = torch.zeros(
            (n_atoms, prod.shape[1], self.n_out),
            dtype=prod.dtype,
            device=prod.device,
        )
        return out.index_add(-1, self.m_sum_ind, prod)


class FCRight2Left(torch.nn.Module):
    """``FCRight2Left`` channel mixing."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        p = f"fc_{name}"
        if not _int(arrays[f"{p}_left_coefs"]):
            raise NotImplementedError("FCRight2Left without left coefficients")
        if np.asarray(arrays[f"{p}_w_left"]).ndim != 3:
            raise NotImplementedError("element-dependent FCRight2Left")
        self.w_left = _param(arrays[f"{p}_w_left"])  # (K, N_in, W)
        self.w_right = _param(arrays[f"{p}_w_right"])
        self.norm_left: Final[float] = _float(arrays[f"{p}_norm_left"])
        self.norm_right: Final[float] = _float(arrays[f"{p}_norm_right"])
        self.tile_left: torch.Tensor
        self.tile_right: torch.Tensor
        self.collect_to: torch.Tensor
        self.collect_from: torch.Tensor
        self.out_norm: torch.Tensor
        self.register_buffer("tile_left", _long(arrays[f"{p}_w_tile_left"]))
        self.register_buffer("tile_right", _long(arrays[f"{p}_w_tile_right"]))
        self.register_buffer("collect_to", _long(arrays[f"{p}_collect_to"]))
        self.register_buffer("collect_from", _long(arrays[f"{p}_collect_from"]))
        self.register_buffer(
            "out_norm",
            _t(np.asarray(arrays[f"{p}_norm_out_factor"]).reshape(-1)),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        w_left = self.w_left.index_select(-1, self.tile_left)
        out = torch.einsum("knd,and->akd", w_left, left) * self.norm_left
        r = right.index_select(-1, self.collect_from)
        w_right = self.w_right.index_select(-1, self.tile_right)
        r = torch.einsum("knd,and->akd", w_right, r) * self.norm_right
        out = out.index_add(-1, self.collect_to, r)
        return out * self.out_norm


class ProductFunction(torch.nn.Module):
    """``ProductFunction``: channel-wise Clebsch-Gordan coupled product."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        p = f"prod_{name}"
        self.left_ind: torch.Tensor
        self.right_ind: torch.Tensor
        self.m_sum_ind: torch.Tensor
        self.cg: torch.Tensor
        self.register_buffer("left_ind", _long(arrays[f"{p}_left_ind"]))
        self.register_buffer("right_ind", _long(arrays[f"{p}_right_ind"]))
        self.register_buffer("m_sum_ind", _long(arrays[f"{p}_m_sum_ind"]))
        self.register_buffer("cg", _t(arrays[f"{p}_cg_coeff"]))
        self.n_out: Final[int] = _int(arrays[f"{p}_n_output_funcs"])

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        prod = (
            left.index_select(-1, self.left_ind)
            * right.index_select(-1, self.right_ind)
            * self.cg
        )
        out = torch.zeros(
            (prod.shape[0], prod.shape[1], self.n_out),
            dtype=prod.dtype,
            device=prod.device,
        )
        return out.index_add(-1, self.m_sum_ind, prod)


class _ReduceInput(torch.nn.Module):
    def __init__(self, arrays: Arrays, name: str, source: str):
        super().__init__()
        self.weights = _param(arrays[f"{name}_reduce_{source}_W"])
        self.element_dependent: Final[bool] = self.weights.ndim == 4
        self.norm: Final[float] = _float(arrays[f"{name}_reduce_{source}_norm"])
        self.collect_ind: torch.Tensor
        self.tile: torch.Tensor
        self.target: torch.Tensor
        self.register_buffer(
            "collect_ind", _long(arrays[f"{name}_collect_ind_{source}"])
        )
        self.register_buffer("tile", _long(arrays[f"{name}_w_l_tile_{source}"]))
        self.register_buffer(
            "target", _long(arrays[f"{name}_total_sum_ind_{source}"])
        )

    def forward(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        x = x.index_select(-1, self.collect_ind)  # (N, C, D)
        if not self.element_dependent:
            w = self.weights.index_select(-1, self.tile)  # (K, C, D)
            return torch.einsum("kcd,acd->akd", w, x) * self.norm

        # element-dependent weights: contract each element's atoms separately
        # to avoid materialising a (N, K, C, D) weight tensor
        out = torch.zeros(
            (x.shape[0], self.weights.shape[1], x.shape[-1]),
            dtype=x.dtype,
            device=x.device,
        )
        elements: list[int] = torch.unique(mu).tolist()
        for e in elements:
            idx = torch.nonzero(mu == e).squeeze(-1)
            w = self.weights[e].index_select(-1, self.tile)  # (K, C, D)
            contrib = torch.einsum("kcd,acd->akd", w, x.index_select(0, idx))
            out = out.index_copy(0, idx, contrib)
        return out * self.norm


class FunctionReduce(torch.nn.Module):
    """``FunctionReduceN``: linear reduction of several features."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        sources = [s.decode() for s in arrays[f"{name}_instruction_names"]]
        self.sources: list[str] = sources
        self.inputs = torch.nn.ModuleList(
            [_ReduceInput(arrays, name, s) for s in sources]
        )
        self.n_out: Final[int] = _int(arrays[f"{name}_n_out"])
        self.n_funcs: Final[int] = _int(arrays[f"{name}_n_funcs"])
        self.norm_map: Final[float] = _float(arrays[f"{name}_norm_map"])

    def forward(
        self, features: list[torch.Tensor], mu: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros(
            (mu.shape[0], self.n_out, self.n_funcs),
            dtype=features[0].dtype,
            device=features[0].device,
        )
        for i, reducer in enumerate(self.inputs):
            out = out.index_add(-1, reducer.target, reducer(features[i], mu))
        return out * self.norm_map


class InvariantLayerRMSNorm(torch.nn.Module):
    """``InvariantLayerRMSNorm`` (``full`` or ``only_nonlin``)."""

    def __init__(self, arrays: Arrays, name: str):
        super().__init__()
        kind = _int(arrays[f"{name}_type"])
        if kind not in (0, 1):
            raise NotImplementedError(f"RMSNorm type {kind} not supported")
        self.full: Final[bool] = kind == 0
        self.scale = _param(np.asarray(arrays[f"{name}_scale"]).reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, n, 1)
        scale = self.scale.view(1, -1, 1)
        if self.full:
            rms = torch.rsqrt((x**2).mean(dim=1, keepdim=True) + 1e-10)
            return x * rms * scale
        lin = x[:, :1, :]
        nonlin = x[:, 1:, :]
        rms = torch.rsqrt((nonlin**2).mean(dim=1, keepdim=True) + 1e-10)
        return torch.cat([lin, nonlin * rms * scale], dim=1)


class EnergyReadout(torch.nn.Module):
    r"""
    ``LinMLPOut2ScalarTarget`` plus the collapsed per-element shifts:

    .. math::

        \varepsilon_i = s \left( \sum_o x^o_{i,0} + \text{MLP}\left(
        \sum_o x^o_{i,1:} \right) \right) + \delta_{\mu_i}
    """

    def __init__(self, arrays: Arrays, n_elements: int):
        super().__init__()
        n_layers = _int(arrays["energy_mlp_n_layers"])
        self.weights = torch.nn.ParameterList(
            [_param(arrays[f"energy_mlp_W{i}"]) for i in range(n_layers)]
        )
        self.norms: list[float] = [
            _float(arrays[f"energy_mlp_norm{i}"]) for i in range(n_layers)
        ]
        act = _int(arrays["energy_mlp_activation"])
        if act not in (0, 1):
            raise NotImplementedError(f"Unsupported activation code {act}")
        self.tanh: Final[bool] = act == 1
        shifts = arrays.get("shift_values")
        if shifts is None:
            shifts = np.zeros(n_elements)
        self.shifts = _param(np.asarray(shifts).reshape(-1))
        scale = arrays.get("output_scale")
        self.output_scale: Final[float] = (
            1.0 if scale is None else _float(scale)
        )

    def forward(
        self, origins: list[torch.Tensor], mu: torch.Tensor
    ) -> torch.Tensor:
        lin = torch.zeros_like(origins[0][:, 0, 0])
        x = torch.zeros_like(origins[0][:, 1:, 0])
        for o in origins:
            lin = lin + o[:, 0, 0]
            x = x + o[:, 1:, 0]
        n = len(self.norms)
        for i, w in enumerate(self.weights):
            x = x @ (w * self.norms[i])
            if i < n - 1:
                x = torch.tanh(x) if self.tanh else torch.nn.functional.silu(x)
        energy = (x.squeeze(-1) + lin) * self.output_scale
        return energy + self.shifts[mu]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class GRACEFoundationModel(GraphPESModel):
    """
    A two-layer GRACE foundation model, re-implemented in PyTorch from the
    weights and coupling tables of its ``kokkos.npz`` export.

    Use :func:`~graph_pes.interfaces.grace_fm` to create one of these.
    """

    def __init__(self, arrays: Arrays, name: str = "GRACE-2L"):
        for key, expected in _EXPECTED_NAMES.items():
            actual = [s.decode() for s in arrays[key]]
            if actual != expected:
                raise ValueError(
                    f"Unsupported GRACE architecture: expected {key}="
                    f"{expected}, got {actual}. Only the two-layer GRACE "
                    "foundation models are currently supported."
                )

        rcut = _float(arrays["rcut"])
        super().__init__(cutoff=rcut, implemented_properties=["local_energies"])
        self.model_name = name

        # chemistry
        self.elements: list[str] = [s.decode() for s in arrays["element_names"]]
        from ase.data import atomic_numbers, chemical_symbols

        self.chemical_symbols: list[str] = chemical_symbols
        self.Z_to_idx: torch.Tensor
        self.register_buffer(
            "Z_to_idx", torch.full((MAX_Z + 1,), -1, dtype=torch.long)
        )
        for i, symbol in enumerate(self.elements):
            self.Z_to_idx[atomic_numbers[symbol]] = i
        self.embeddings = _param(arrays["chem_embedding"])

        # geometry
        bcm = None
        if _int(arrays["has_bond_specific_cutoff"]):
            bcm = _t(arrays["bond_cutoff_map"])
        self.radial_basis = ChebyshevRadialBasis(
            nfunc=_int(arrays["nradbase"]),
            p=_int(arrays["radial_basis_p"]),
            rcut=rcut,
            bond_cutoff_map=bcm,
        )
        self.spherical_harmonics = TPSphericalHarmonics(_int(arrays["lmax"]))

        # first layer
        self.R = MLPRadialFunction(arrays, "R")
        self.A = ScalarSingleParticleBasis(arrays, "A")
        self.A1 = FCRight2Left(arrays, "A1")
        self.AA = ProductFunction(arrays, "AA")
        self.AA1 = FCRight2Left(arrays, "AA1")
        self.AAA = ProductFunction(arrays, "AAA")
        self.AA2 = FCRight2Left(arrays, "AA2")
        self.AAAA = ProductFunction(arrays, "AAAA")
        self.rho = FunctionReduce(arrays, "rho")
        self.I_nl_LN = InvariantLayerRMSNorm(arrays, "I_nl_LN")

        # indicators passed to the second layer
        self.I1 = FunctionReduce(arrays, "I1")
        self.I = FunctionReduce(arrays, "I")

        # second layer
        self.R1 = MLPRadialFunction(arrays, "R1")
        self.B0 = ScalarSingleParticleBasis(arrays, "B0")
        self.YI = EquivariantSingleParticleBasis(arrays, "YI")
        self.B = FunctionReduce(arrays, "B")
        self.B1 = FCRight2Left(arrays, "B1")
        self.BB = ProductFunction(arrays, "BB")
        self.BB1 = FCRight2Left(arrays, "BB1")
        self.BBB = ProductFunction(arrays, "BBB")
        self.BB2 = FCRight2Left(arrays, "BB2")
        self.BBBB = ProductFunction(arrays, "BBBB")
        self.I2 = FunctionReduce(arrays, "I2")
        self.I_0_LN = InvariantLayerRMSNorm(arrays, "I_0_LN")

        # readout
        self.readout = EnergyReadout(arrays, len(self.elements))

    def element_indices(self, Z: torch.Tensor) -> torch.Tensor:
        idx = self.Z_to_idx[Z]
        with torch.no_grad():
            if (idx < 0).any():
                unknown_Z = torch.unique(Z[idx < 0])
                unknown = [self.chemical_symbols[z] for z in unknown_Z]
                raise ValueError(
                    f"{self.model_name} does not support elements {unknown}"
                )
        return idx

    def forward(self, graph: AtomicGraph) -> dict[PropertyKey, torch.Tensor]:
        N = number_of_atoms(graph)
        mu = self.element_indices(graph.Z)
        ind_i, ind_j = graph.neighbour_list[0], graph.neighbour_list[1]
        mu_i, mu_j = mu[ind_i], mu[ind_j]

        vectors = neighbour_vectors(graph)
        d = neighbour_distances(graph)
        basis = self.radial_basis(d, mu_i, mu_j)
        Y = self.spherical_harmonics(vectors / d.unsqueeze(-1))

        # first layer
        A = self.A(self.R(basis), Y, self.embeddings, mu_j, ind_i, N)
        A1 = self.A1(A, A)
        AA = self.AA(A1, A1)
        AAA = self.AAA(self.AA1(AA, A), A)
        AA2 = self.AA2(AA, A)
        AAAA = self.AAAA(AA2, AA2)
        layer_1 = [A, AA, AAA, AAAA]
        rho = self.I_nl_LN(self.rho(layer_1, mu))

        # second layer
        indicator = self.I([self.I1(layer_1, mu)], mu)
        R1 = self.R1(basis)
        B0 = self.B0(R1, Y, self.embeddings, mu_j, ind_i, N)
        YI = self.YI(R1, Y, indicator, ind_i, ind_j, N)
        B = self.B([YI, B0], mu)
        B1 = self.B1(B, B)
        BB = self.BB(B1, B1)
        BBB = self.BBB(self.BB1(BB, B), B)
        BB2 = self.BB2(BB, B)
        BBBB = self.BBBB(BB2, BB2)
        I2 = self.I_0_LN(self.I2([B, BB, BBB, BBBB], mu))

        return {"local_energies": self.readout([rho, I2], mu)}

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (
            f"GRACEFoundationModel(name={self.model_name!r}, "
            f"elements={len(self.elements)}, cutoff={self.cutoff.item():.2f}, "
            f"parameters={n:,})"
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    root = os.environ.get("GRAPH_PES_CACHE")
    if root is None:
        root = Path.home() / ".cache" / "graph-pes"
    return Path(root) / "grace"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.info(f"Downloading {url} to {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def grace_fm(
    model: str = "GRACE-2L-SMAX-OMAT-medium",
    path: str | Path | None = None,
) -> GRACEFoundationModel:
    """
    Load a pre-trained, two-layer
    `GRACE foundation model <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__
    as a :class:`~graph_pes.GraphPESModel`.

    The model's ``kokkos.npz`` export is downloaded from
    `Hugging Face <https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models>`__
    on first use and cached in ``~/.cache/graph-pes/grace`` (set the
    ``GRAPH_PES_CACHE`` environment variable to change this). The model is then
    re-built in pure PyTorch: neither TensorFlow nor ``tensorpotential`` is
    required.

    .. note::

        The GRACE foundation models are released under the
        `Academic Software License <https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models>`__.
        By downloading them you agree to its terms.

    Parameters
    ----------
    model
        The full name of the foundation model, e.g.
        ``"GRACE-2L-SMAX-OMAT-medium"`` or ``"GRACE-2L-OMAT-large-ft-E"``.
        See ``GRACE_2L_FOUNDATION_MODELS`` for all supported names.
    path
        Optional path to an already-downloaded ``<model>-kokkos.npz`` file.

    Examples
    --------
    Use the model in ASE:

    >>> from graph_pes.interfaces import grace_fm
    >>> model = grace_fm("GRACE-2L-SMAX-OMAT-medium")
    >>> calc = model.ase_calculator()

    Fine-tune the model with ``graph-pes-train``:

    .. code-block:: yaml

        model:
            +grace_fm:
                model: GRACE-2L-SMAX-OMAT-medium
    """
    if path is None:
        if model not in GRACE_2L_FOUNDATION_MODELS:
            raise ValueError(
                f"Unknown GRACE foundation model {model!r}. Expected one of "
                f"{GRACE_2L_FOUNDATION_MODELS}"
            )
        path = _cache_dir() / f"{model}-kokkos.npz"
        if not path.exists():
            _download(_HF_KOKKOS_URL.format(name=model), path)

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("uq_")}
    return GRACEFoundationModel(arrays, name=model).eval()
