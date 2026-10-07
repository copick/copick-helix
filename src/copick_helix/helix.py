"""Helical symmetry and the Fourier support it implies.

A helical assembly with axial rise ``h`` and twist ``omega`` per subunit (cylindrical coordinates (r, phi, z) about the
filament axis) has Fourier–Bessel terms only at

    Z(n, m) = (m + n * omega / 360) / h,

one term per pair of integers (n, m): ``n`` is the angular (Bessel) order, ``m`` the layer-line index of the subunit
lattice. A term of order n carries signal only where J_n(2 pi R r) is non-negligible for radii r inside the
filament, i.e. roughly |n| <= 2 pi R r_out + 2.

Every filament family is described this way:

- **Microtubule N_S.** The monomer lattice is a 1-start helix with rise ``S * a / N`` (a = monomer repeat) and twist
  ``360 / N`` (plus a small supertwist for lattices other than 13_3). This gives the equator orders n = +-N k and the
  monomer layer line Z = 1/a with orders n = S + N k. The seam breaks only the dimer-level symmetry.
- **Actin.** Rise about 27.5 A, twist about -166.6 deg: the 59 A (n = -1), 51 A (n = +1), about 36 nm crossover
  (n = +2) and 27.5 A meridional (n = 0) layer lines.
- **Vimentin intermediate filament.** Rise 42.5 A, twist 73.7 deg (PDB 8RVE).

Sign convention: phi increases from x (= e1, about the beam) towards y (= e2), and the (n, Z) coefficients follow the
forward FFT. A positive twist then puts the MT 3-start term at n = +3, Z = +1/a, which is what the 6DPV-derived
13_3 lattice shows in this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Term:
    """One Fourier–Bessel term (Bessel order n, subunit layer-line index m)."""

    n: int
    m: int

    def __add__(self, other: "Term") -> "Term":
        return Term(self.n + other.n, self.m + other.m)


@dataclass(frozen=True)
class HelicalSymmetry:
    """Rise (A) and twist (deg) per subunit."""

    rise: float
    twist: float

    def Z(self, term: Term) -> float:
        """Axial frequency (1/A) of a term."""
        return (term.m + term.n * self.twist / 360.0) / self.rise

    def terms(self, n_max: int, z_max: float, z_min: float = -np.inf) -> list[Term]:
        """All terms with |n| <= n_max and z_min <= Z <= z_max."""
        out = []
        m_lo = int(np.floor(z_min * self.rise - n_max * abs(self.twist) / 360.0)) - 1 if np.isfinite(z_min) else None
        m_hi = int(np.ceil(z_max * self.rise + n_max * abs(self.twist) / 360.0)) + 1
        if m_lo is None:
            m_lo = -m_hi
        for n in range(-n_max, n_max + 1):
            for m in range(m_lo, m_hi + 1):
                z = self.Z(Term(n, m))
                if z_min - 1e-12 <= z <= z_max + 1e-12:
                    out.append(Term(n, m))
        return out

    def plane_orders(self, term: Term, n_max: int, tol: float) -> list[int]:
        """Bessel orders sharing ``term``'s layer line (|Z' - Z| <= tol): everything a fit on that plane must model."""
        z0 = self.Z(term)
        return sorted({t.n for t in self.terms(n_max, z0 + tol, z0 - tol)})


def n_cap(R: float, r_out: float, extra: int = 2) -> int:
    """Highest Bessel order that can carry signal at Fourier radius R (1/A) for a filament of outer radius r_out (A)."""
    return int(np.ceil(2 * np.pi * R * r_out)) + extra


@dataclass
class Support:
    """Where a family's signal is: the helical symmetry, the outer radius, the axial frequency band and the terms."""

    symmetry: HelicalSymmetry
    r_out: float
    z_max: float
    n_max: int
    exclude: list[tuple[int, float]] = field(default_factory=lambda: [(0, 0.0)])  # (n, Z): radial profile etc.

    def nz_mask(self, n: np.ndarray, Z: np.ndarray, z_tol: float) -> np.ndarray:
        """Boolean (n, Z) mask over FFT grids ``n`` (orders) and ``Z`` (1/A): True on the family's layer lines."""
        mask = np.zeros((len(n), len(Z)), bool)
        for t in self.symmetry.terms(self.n_max, self.z_max, -self.z_max):
            zi = np.abs(Z - self.symmetry.Z(t)) <= z_tol
            ni = n == t.n
            mask[np.ix_(ni, zi)] = True
        for ne, ze in self.exclude:
            mask[np.ix_(n == ne, np.abs(Z - ze) <= z_tol)] = False
        return mask
