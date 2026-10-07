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
    pos = np.arange(0, vol.shape[0] - 1 + 1e-6, factor)
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
    plane_tol_bins: float = 0.5  # invariants: orders whose line lies within this many Z bins are fitted together
    support_z_bins: float = 1.0  # iterative: half-width of a layer line in the support mask (Z bins)
    support_drop_n0: bool = False
    support_eq_nmax: int = -1
    support_eq_band: float = 0.0
    recentre_window: float = 300.0  # A
    recentre_max_shift: float = 100.0  # A, cap on a window's correction
    screw: Callable = None  # reference params -> (P A, omega deg): the lattice's symmetry step along the axis
    plus_at_minus_z: bool = False  # the labelling model (and so the oriented reference) has its plus end at -z
    notes: dict = field(default_factory=dict)

    def support(self, z_max: float, n_max: int, **params) -> Support:
        sym = self.symmetry(**(params or self.reference_params))
        return Support(sym, r_out=self.r_out, z_max=z_max, n_max=n_max, drop_n0=self.support_drop_n0,
                       eq_nmax=self.support_eq_nmax, eq_band=self.support_eq_band)


def _ring(radius: float, width: float):
    return lambda r: np.exp(-0.5 * ((r - radius) / width) ** 2)


def _rod(radius: float, edge: float):
    return lambda r: 0.5 - 0.5 * np.tanh((r - radius) / edge)


def microtubule(N: int = 13, S: int = 3) -> Family:
    """N_S microtubule. The monomer lattice is a 1-start helix: rise S*a/N, twist -360/N (a = monomer repeat, measured
    per filament from the 4 nm layer line). Phase-invariant terms: equator n = N and monomer line n = S, S - N, S + N."""

    def symmetry(monomer_repeat: float) -> HelicalSymmetry:
        return HelicalSymmetry(rise=S * monomer_repeat / N, twist=-360.0 / N)

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
        label_model=label_model, screw=lambda p: (2.0 * p["monomer_repeat"], 0.0), plus_at_minus_z=True,
        notes={"lattice_readout": "protofilament number from the real-space equator count (lattice.equator_count)",
               "polarity_convention": "points ordered minus -> plus when polarity_known",
               "model": "PDB 6DPV (undecorated GDP MT); plus end = side of each monomer's nucleotide"})


def helical_family(name: str, polar: bool, rise: float, twist: float, terms, triples, fit_lines, model_pdb: str,
                   r_out: float, r_mask: float, cyl_band, profile_band, recentre_profile, segment_length: float = 1250.0,
                   in_plane_half_width: float = 200.0, plane_tol_bins: float = 0.5, support_z_bins: float = 1.0,
                   support_drop_n0: bool = True, support_eq_nmax: int = -1, support_eq_band: float = 0.0,
                   recentre_window: float = 300.0, recentre_max_shift: float = 100.0, min_line_snr: float = 3.0,
                   plus_at_minus_z: bool = False, notes: dict | None = None) -> Family:
    """A 1-start helical family (rise, physical twist) whose per-filament rise and twist come from two measured layer
    lines (``fit_lines``: two Terms with different n). Filaments are stretched to the reference rise for the iterative
    reference (a twist mismatch moves layer lines by << 1 Z bin per segment)."""

    def symmetry(rise: float, twist: float) -> HelicalSymmetry:
        return HelicalSymmetry(rise=rise, twist=twist)

    ref = {"rise": rise, "twist": twist}

    def measure(st):
        from .lattice import rise_twist_from_lines

        d = rise_twist_from_lines(st.vol, st.step, symmetry(**ref), fit_lines[0], fit_lines[1], r_max=r_mask)
        q = min(v for k, v in d.items() if k.startswith("snr_"))
        return {"rise": d["rise"], "twist": d["twist"]}, q

    def to_reference_grid(st, params):
        return stretch_axis(st, params["rise"] / rise)

    def label_model(st):
        from .models import HelicalModel, on_grid

        m = HelicalModel(model_pdb, rise, twist)
        return on_grid(*m.atoms(segment_length), st.step, st.vol.shape[1], segment_length), dict(ref)

    return Family(name=name, polar=polar, symmetry=symmetry, reference_params=ref, terms=list(terms),
                  triples=list(triples), r_out=r_out, r_mask=r_mask, cyl_band=cyl_band, segment_length=segment_length,
                  in_plane_half_width=in_plane_half_width, recentre_profile=recentre_profile, profile_band=profile_band,
                  measure=measure, measure_min_quality=min_line_snr, to_reference_grid=to_reference_grid,
                  label_model=label_model, plane_tol_bins=plane_tol_bins, support_z_bins=support_z_bins,
                  support_drop_n0=support_drop_n0, support_eq_nmax=support_eq_nmax, support_eq_band=support_eq_band,
                  recentre_window=recentre_window, recentre_max_shift=recentre_max_shift,
                  screw=lambda p: (p["rise"], p["twist"]), plus_at_minus_z=plus_at_minus_z, notes=notes or {})


def intermediate_filament() -> Family:
    """Vimentin intermediate filament (PDB 8RVE / EMD-16844): rise 42.461 A, twist +73.73 deg, five protofibrils
    (about 21 nm protofibril repeat) around a luminal fibre of head domains.

    The deposited model and map are not symmetric under a polarity flip, so the family is treated as polar. In 10 A
    tomograms its layer lines were not detectable (pooled enrichment at decoy level), so a polarity call needs the
    lattice gate first. Closely spaced lines ((4, 1) is 0.7 Z bins from (-1, 0) at 125 nm) are fitted together as
    nuisance orders within 2.5 bins; without that, an exactly apolar control showed a spurious eigenvalue gap."""
    T = Term
    return helical_family(
        "intermediate_filament", polar=True, rise=42.461, twist=73.7308,
        terms=[T(-5, -1), T(-1, 0), T(4, 1), T(-6, -1), T(9, 2), T(8, 2), T(-10, -2)],
        triples=[(T(-5, -1), T(4, 1), T(-1, 0)), (T(-1, 0), T(-5, -1), T(-6, -1)), (T(4, 1), T(4, 1), T(8, 2)),
                 (T(-5, -1), T(-5, -1), T(-10, -2))],
        fit_lines=(T(-1, 0), T(4, 1)), model_pdb="8RVE", r_out=65.0, r_mask=80.0, cyl_band=(4.0, 80.0),
        profile_band=(30.0, 55.0), recentre_profile=_ring(42.0, 12.0), plane_tol_bins=2.5, support_z_bins=1.5,
        support_drop_n0=True, support_eq_nmax=4, support_eq_band=1 / 500.0, recentre_window=600.0,
        recentre_max_shift=30.0,
        notes={"lattice_gate": "required before interpreting polarity", "model": "PDB 8RVE (vimentin), EMD-16844"})


def get_family(name: str) -> Family:
    """'microtubule' (13_3), 'microtubule_N_S', 'actin', 'intermediate_filament'."""
    key = name.lower().replace("-", "_")
    if key.startswith("microtubule"):
        parts = key.split("_")[1:]
        return microtubule(int(parts[0]), int(parts[1])) if len(parts) == 2 else microtubule()
    if key in FAMILIES:
        return FAMILIES[key]()
    raise ValueError(f"unknown family {name!r}; known: microtubule[_N_S], {', '.join(FAMILIES)}")


FAMILIES: dict[str, Callable[[], Family]] = {"intermediate_filament": intermediate_filament}
