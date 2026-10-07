"""Atomic models as densities on the straightened-filament grid (optional; needs gemmi).

Used only to name which polarity group is 'plus' and for synthetic tests; the analyses themselves are data-driven.

Microtubule lattices N_S are rolled from the 6DPV surface lattice (undecorated GDP microtubule, a 3-protofilament x
2-dimer patch of a 14_3 lattice, axis along z). Nucleotides sit on the -z face of every monomer in that frame, so the
plus end is at -z for ``flip=False``; ``flip=True`` turns the lattice 180 deg about x.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

Z_OF = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "MG": 12}
CACHE = os.environ.get("COPICK_HELIX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "copick-helix"))


def fetch_pdb(pdb_id: str, path: str | None = None) -> str:
    """mmCIF of a PDB entry, cached."""
    if path and os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    out = path or os.path.join(CACHE, f"{pdb_id.upper()}.cif")
    if not os.path.exists(out):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif", out)
    return out


def _chain_atoms(model, name, ligands=False):
    xyz, w = [], []
    for r in model[name]:
        if ligands and r.het_flag != "H":
            continue
        for a in r:
            xyz.append(a.pos.tolist())
            w.append(Z_OF.get(a.element.name.upper(), 6))
    return np.array(xyz).reshape(-1, 3), np.array(w, float)


class MicrotubuleLattice:
    """Intrinsic surface lattice of 6DPV and monomer atoms in local cylindrical offsets; rolls any N_S lattice."""

    def __init__(self, cif: str | None = None):
        import gemmi

        st = gemmi.read_structure(fetch_pdb("6DPV", cif))
        m = st[0]
        cen = {ch.name: _chain_atoms(m, ch.name)[0].mean(0) for ch in m}
        self.axis_xy = self._circle([cen[k] for k in ("E", "A", "C")])
        phi = {k: np.arctan2(v[1] - self.axis_xy[1], v[0] - self.axis_xy[0]) for k, v in cen.items()}
        rad = {k: np.hypot(v[0] - self.axis_xy[0], v[1] - self.axis_xy[1]) for k, v in cen.items()}
        self.r0 = float(np.mean([rad[k] for k in "ABCDEFGHIJKL"]))

        def step(p, q):
            dphi = (phi[q] - phi[p] + np.pi) % (2 * np.pi) - np.pi
            return np.array([-self.r0 * dphi, cen[q][2] - cen[p][2]])

        b14 = np.mean([step("A", "C"), step("E", "A"), step("B", "D"), step("F", "B")], axis=0)
        a14 = np.mean([step(p, q) for p, q in (("A", "K"), ("H", "B"), ("C", "L"), ("I", "D"), ("E", "J"), ("G", "F"))],
                      axis=0) / 2.0
        th = float(np.arctan2(a14[0], a14[1]))
        c, s = np.cos(th), np.sin(th)
        to_pf = np.array([[c, -s], [s, c]])
        a_pf, b_pf = to_pf @ a14, to_pf @ b14
        self.a, self.d, self.delta0 = float(a_pf[1]), float(b_pf[0]), float(b_pf[1])
        self.monomers = {}
        for label, ch in (("alpha", "A"), ("beta", "B")):
            xyz, w = _chain_atoms(m, ch)
            lx, lw = _chain_atoms(m, ch, ligands=True)
            xyz, w = np.vstack([xyz, lx]), np.concatenate([w, lw])
            r = np.hypot(xyz[:, 0] - self.axis_xy[0], xyz[:, 1] - self.axis_xy[1])
            ph = np.arctan2(xyz[:, 1] - self.axis_xy[1], xyz[:, 0] - self.axis_xy[0])
            dphi = (ph - phi[ch] + np.pi) % (2 * np.pi) - np.pi
            du, dz = -self.r0 * dphi, xyz[:, 2] - cen[ch][2]
            du, dz = c * du - s * dz, s * du + c * dz
            self.monomers[label] = (np.stack([r - rad[ch], du, dz], 1), w)

    @staticmethod
    def _circle(pts):
        (x1, y1), (x2, y2), (x3, y3) = [(p[0], p[1]) for p in pts]
        A = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]]) * 2
        B = np.array([x2**2 - x1**2 + y2**2 - y1**2, x3**2 - x1**2 + y3**2 - y1**2])
        return np.linalg.solve(A, B)

    def geometry(self, N: int, S: int) -> dict:
        Cu, Cv = N * self.d, N * self.delta0 - S * self.a
        theta = np.arctan2(Cv, Cu)
        L = np.hypot(Cu, Cv)
        return {"N": N, "S": S, "theta_deg": float(np.degrees(theta)), "radius": float(L / (2 * np.pi)),
                "seam": bool(S % 2)}

    def atoms(self, N: int, S: int, length: float, flip: bool = False, z0: float = 0.0):
        g = self.geometry(N, S)
        theta, R = np.radians(g["theta_deg"]), g["radius"]
        c, s = np.cos(theta), np.sin(theta)
        out, wts = [], []
        for i in range(N):
            for j in range(-4, int(np.ceil(length / self.a)) + 5):
                u, v = i * self.d, i * self.delta0 + j * self.a
                uc, zc = c * u + s * v, -s * u + c * v
                if not (z0 - 30 <= zc <= z0 + length + 30):
                    continue
                off, w = self.monomers["alpha" if j % 2 == 0 else "beta"]
                du, dz = c * off[:, 1] + s * off[:, 2], -s * off[:, 1] + c * off[:, 2]
                r = R + off[:, 0]
                phi = -(uc + du) / R
                out.append(np.stack([r * np.cos(phi), r * np.sin(phi), zc + dz], 1))
                wts.append(w)
        xyz, w = np.vstack(out), np.concatenate(wts)
        keep = (xyz[:, 2] >= z0) & (xyz[:, 2] < z0 + length)
        xyz, w = xyz[keep], w[keep]
        if flip:
            xyz = xyz * np.array([1, -1, -1]) + np.array([0, 0, 2 * z0 + length])
        return xyz, w


def density(xyz, w, apix, box_xy, length, sigma=1.5):
    """Gaussian-splatted density vol[z, y, x]; xy centred on the axis (index n/2), z from 0."""
    from scipy.ndimage import gaussian_filter

    nxy = int(round(box_xy / apix))
    nz = int(np.ceil(length / apix))
    idx = np.empty_like(xyz)
    idx[:, 0] = xyz[:, 0] / apix + nxy / 2
    idx[:, 1] = xyz[:, 1] / apix + nxy / 2
    idx[:, 2] = xyz[:, 2] / apix
    vol = np.zeros((nz, nxy, nxy), np.float32)
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ww = w * np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * \
                    np.where(dz, f[:, 2], 1 - f[:, 2])
                zi, yi, xi = i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx
                ok = (zi >= 0) & (zi < nz) & (yi >= 0) & (yi < nxy) & (xi >= 0) & (xi < nxy)
                np.add.at(vol, (zi[ok], yi[ok], xi[ok]), ww[ok])
    return gaussian_filter(vol, sigma / apix)


def on_grid(xyz, w, step: float, n_inplane: int, length: float, res: float = 10.0) -> np.ndarray:
    """Model on the straightened grid: ``step`` sampling, n_inplane pixels with the axis at the centre index,
    band-limited to ``res`` A (built at step/2 and decimated)."""
    from .fourier import lowpass

    vol = density(xyz, w, apix=step / 2, box_xy=(2 * n_inplane) * step / 2, length=length)
    vol = lowpass(vol, step / 2, res)[::2, 1::2, 1::2]
    return vol[: int(round(length / step)), :n_inplane, :n_inplane]
