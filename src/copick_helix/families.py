"""Filament families: the structural priors (inductive bias) each analysis uses.

A family states where its signal can be (helical symmetry -> layer lines and allowed orders, radius band), how to
recentre a straightened filament (radial profile), how long a segment is, which Fourier terms carry polarity
(phase invariants), and whether the structure is polar at all. Parameters that vary between filaments (MT monomer repeat;
actin and IF rise and twist) are measured per filament and passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .helix import HelicalSymmetry, Support, Term


def stretch_axis(st, factor: float):
    """Resample a Straightened filament along s so a period P becomes P / factor (factor = measured / reference).
    Returns (vol, beam_local, tilt_local) on the new s grid."""
    from scipy.ndimage import map_coordinates

    vol = st.vol
    pos = np.arange(0, vol.shape[0] - 1, factor)
    grid = np.meshgrid(pos, np.arange(vol.shape[1]), np.arange(vol.shape[2]), indexing="ij")
    v = map_coordinates(vol, grid, order=1).astype(np.float32)
    idx = np.clip(np.round(pos).astype(int), 0, len(st.beam_local) - 1)
    return v, st.beam_local[idx], st.tilt_local[idx]


@dataclass
class Family:
    name: str
    polar: bool
    symmetry: Callable[..., HelicalSymmetry]  # from the filament's measured parameters
    reference_params: dict  # parameters of the common grid used by the iterative reference
    terms: list[Term]  # phase-invariant terms
    triples: list[tuple[Term, Term, Term]]  # phase-invariant triple products (A + B = C)
    r_out: float  # outer radius of the structure (A): Bessel order cap
    r_mask: float  # soft in-plane mask radius for the invariants (A)
    cyl_band: tuple[float, float]  # radial band for the cylindrical resampling (iterative), A
    segment_length: float  # A
    in_plane_half_width: float  # straightening box half width, A
    recentre_profile: Callable[[np.ndarray], np.ndarray]  # radial template for recentring, as a function of r (A)
    profile_band: tuple[float, float]  # radius band of the expected density (protein sign test), A
    measure: Callable = None  # Straightened -> (params dict, quality)
    measure_min_quality: float = 3.0
    to_reference_grid: Callable = None  # (Straightened, params) -> (vol, beam_local, tilt_local) on the common grid
    label_model: Callable = None  # Straightened -> (model volume on its grid in the 'plus' orientation, params) or None
    notes: dict = field(default_factory=dict)

    def support(self, z_max: float, n_max: int, **params) -> Support:
        sym = self.symmetry(**(params or self.reference_params))
        return Support(sym, r_out=self.r_out, z_max=z_max, n_max=n_max)


def _ring(radius: float, width: float):
    return lambda r: np.exp(-0.5 * ((r - radius) / width) ** 2)


def _rod(radius: float, edge: float):
    return lambda r: 0.5 - 0.5 * np.tanh((r - radius) / edge)


def microtubule(N: int = 13, S: int = 3) -> Family:
    """N_S microtubule. The monomer lattice is a 1-start helix: rise S*a/N, twist 360/N (a = monomer repeat, measured
    per filament from the 4 nm layer line). Phase-invariant terms: equator n = N and monomer line n = S, S - N, S + N."""

    def symmetry(monomer_repeat: float) -> HelicalSymmetry:
        return HelicalSymmetry(rise=S * monomer_repeat / N, twist=360.0 / N)

    eq, s0, sm, sp = Term(N, -1), Term(S, 0), Term(S - N, 1), Term(S + N, -1)
    a_ref = 40.836

    def measure(st):
        from .lattice import microtubule_repeat

        a, snr = microtubule_repeat(st.vol, st.step)
        return {"monomer_repeat": a}, snr

    def to_reference_grid(st, params):
        return stretch_axis(st, params["monomer_repeat"] / a_ref)

    def label_model(st):
        """13_3-type lattice from 6DPV on the filament's grid; plus end at -z (towards the first trace point)."""
        try:
            from .models import MicrotubuleLattice, on_grid
        except ImportError:
            return None
        lat = MicrotubuleLattice()
        xyz, w = lat.atoms(N, S, 1250.0)
        return on_grid(xyz, w, st.step, st.vol.shape[1], 1250.0), {"monomer_repeat": lat.a}

    return Family(
        name=f"microtubule_{N}_{S}", polar=True, symmetry=symmetry,
        reference_params={"monomer_repeat": a_ref}, terms=[eq, s0, sm, sp],
        triples=[(eq, s0, sp), (eq, sm, s0)], r_out=150.0, r_mask=165.0, cyl_band=(60.0, 165.0),
        segment_length=1250.0, in_plane_half_width=200.0, recentre_profile=_ring(112.0, 20.0),
        profile_band=(85.0, 135.0), measure=measure, measure_min_quality=3.0, to_reference_grid=to_reference_grid,
        label_model=label_model,
        notes={"lattice_readout": "protofilament number from the real-space equator count (lattice.equator_count)",
               "polarity_convention": "points ordered minus -> plus when polarity_known",
               "model": "PDB 6DPV (undecorated GDP MT); plus end = side of each monomer's nucleotide"})


def get_family(name: str) -> Family:
    """'microtubule' (13_3), 'microtubule_N_S', 'actin', 'intermediate_filament'."""
    key = name.lower().replace("-", "_")
    if key.startswith("microtubule"):
        parts = key.split("_")[1:]
        return microtubule(int(parts[0]), int(parts[1])) if len(parts) == 2 else microtubule()
    if key in FAMILIES:
        return FAMILIES[key]()
    raise ValueError(f"unknown family {name!r}; known: microtubule[_N_S], {', '.join(FAMILIES)}")


FAMILIES: dict[str, Callable[[], Family]] = {}
