"""Atomic models as densities on the straightened-filament grid (optional; needs gemmi).

Used only to name which polarity group is 'plus' and for synthetic tests; the analyzes themselves are data-driven.

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
    """Gaussian-splatted density vol[z, y, x]; xy centered on the axis (index n/2), z from 0."""
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
    """Model on the straightened grid: ``step`` sampling, n_inplane pixels with the axis at the center index,
    band-limited to ``res`` A (built at step/2 and decimated)."""
    from .fourier import lowpass

    vol = density(xyz, w, apix=step / 2, box_xy=(2 * n_inplane) * step / 2, length=length)
    vol = lowpass(vol, step / 2, res)[::2, 1::2, 1::2]
    return vol[: int(round(length / step)), :n_inplane, :n_inplane]


class HelicalModel:
    """A long helical filament built from a deposited model with helical symmetry (rise A, twist deg).

    One axial slab of height ``rise`` from the middle of the deposited model holds one asymmetric unit; copies of it
    by k helical steps build the filament. The axis and the twist sign are fitted by symmetry self-consistency, so the
    deposited convention does not matter."""

    def __init__(self, pdb_id: str, rise: float, twist: float, cif: str | None = None):
        import gemmi
        from scipy.optimize import minimize
        from scipy.spatial import cKDTree

        st = gemmi.read_structure(fetch_pdb(pdb_id, cif))
        xyz, w = [], []
        for ch in st[0]:
            for r in ch:
                for a in r:
                    xyz.append(a.pos.tolist())
                    w.append(Z_OF.get(a.element.name.upper(), 6))
        xyz, w = np.array(xyz), np.array(w, float)
        zmid = np.median(xyz[:, 2])
        z_lo, z_hi = zmid - 2 * rise, zmid + 2 * rise
        tree = cKDTree(xyz)
        rng = np.random.default_rng(0)
        sel = np.where((xyz[:, 2] >= z_lo) & (xyz[:, 2] < z_hi))[0]
        sel = rng.choice(sel, min(4000, len(sel)), replace=False)

        def mismatch(p, tw):
            c = np.array([p[0], p[1], 0.0])
            moved = (xyz[sel] - c) @ _rotz(np.radians(tw)).T + c + np.array([0, 0, rise])
            return float(np.median(tree.query(moved)[0]))

        best = None
        for sign in (1, -1):
            res = minimize(lambda p: mismatch(p, sign * abs(twist)), xyz[:, :2].mean(0), method="Nelder-Mead",
                           options={"xatol": 0.05, "fatol": 1e-3})
            if best is None or res.fun < best[2]:
                best = (res.x, sign * abs(twist), float(res.fun))
        self.axis_xy, self.twist_deposited, self.mismatch = best
        self.rise = rise
        keep = (xyz[:, 2] >= zmid) & (xyz[:, 2] < zmid + rise)
        self.unit = xyz[keep] - np.array([self.axis_xy[0], self.axis_xy[1], zmid])
        self.unit_w = w[keep]

    def atoms(self, length: float, flip: bool = False):
        """Filament along z in [0, length), axis through x = y = 0; flip: 180 deg about x (the other polarity)."""
        n = int(np.ceil(length / self.rise)) + 2
        xyz = np.vstack([self.unit @ _rotz(np.radians(self.twist_deposited * k)).T + np.array([0, 0, self.rise * k])
                         for k in range(-1, n)])
        w = np.tile(self.unit_w, n + 1)
        keep = (xyz[:, 2] >= 0) & (xyz[:, 2] < length)
        xyz, w = xyz[keep], w[keep]
        if flip:
            xyz = xyz * np.array([1, -1, -1]) + np.array([0, 0, length])
        return xyz, w


def _rotz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _kabsch(P, Q):
    """R, t with Q ~ P R^T + t."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


class HelicalProtomer:
    """One subunit of a 1-start helical filament in the filament frame (axis = z through the origin) and the helix
    relating consecutive subunits: subunit k is the protomer rotated by k * twist about z and shifted by k * rise.

    Built from two consecutive chains of a deposited model (``chain_b`` the next subunit along the 1-start helix from
    ``chain_a``): their CA superposition is a screw whose axis, angle and rise are the filament's, so the helical
    parameters and the axis come from the model itself. Residue numbers are kept, so structural landmarks (actin's
    subdomain 2, which points to the pointed end) can name the ends."""

    def __init__(self, xyz, w, res, rise, twist):
        self.xyz, self.w, self.res, self.rise, self.twist = xyz, w, res, rise, twist

    @classmethod
    def from_model(cls, pdb_id: str, chain_a: str, chain_b: str, cif: str | None = None) -> "HelicalProtomer":
        import gemmi

        m = gemmi.read_structure(fetch_pdb(pdb_id, cif))[0]

        def ca(name):
            out = {}
            for r in m[name]:
                a = r.find_atom("CA", "*")
                if a is not None and r.het_flag != "H":
                    out[r.seqid.num] = np.array(a.pos.tolist())
            return out

        A, B = ca(chain_a), ca(chain_b)
        common = sorted(set(A) & set(B))
        R, t = _kabsch(np.array([A[k] for k in common]), np.array([B[k] for k in common]))
        w_, v = np.linalg.eig(R)
        ax = np.real(v[:, np.argmin(abs(w_ - 1))])
        ax /= np.linalg.norm(ax)
        ang = np.degrees(np.arctan2(ax @ np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]),
                                    np.trace(R) - 1))
        rise = float(t @ ax)
        p = np.linalg.lstsq(np.eye(3) - R, t - rise * ax, rcond=None)[0]
        if rise < 0:  # orient the axis so that the step chain_a -> chain_b rises
            ax, ang, rise = -ax, -ang, -rise
        x = np.cross([0, 1.0, 0], ax)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([1.0, 0, 0], ax)
        x /= np.linalg.norm(x)
        Rot = np.stack([x, np.cross(ax, x), ax])
        xyz, w, res = [], [], []
        for r in m[chain_a]:
            if r.het_flag == "H" and r.name == "HOH":
                continue
            for a in r:
                xyz.append(a.pos.tolist())
                w.append(Z_OF.get(a.element.name.upper(), 6))
                res.append(r.seqid.num if r.het_flag != "H" else -1)
        return cls((np.array(xyz) - p) @ Rot.T, np.array(w, float), np.array(res), rise, float(ang))

    def offset_along_axis(self, residues) -> float:
        """z of the centroid of ``residues`` (iterable of residue numbers) relative to the protomer centroid (A)."""
        sel = np.isin(self.res, list(residues))
        prot = self.res > 0
        return float(self.xyz[sel, 2].mean() - self.xyz[prot, 2].mean())

    def atoms(self, length: float, flip: bool = False):
        """Filament along z in [0, length), axis through x = y = 0; flip: 180 deg about x (the other polarity)."""
        zc = self.xyz[:, 2].mean()
        ks = np.arange(int(np.floor((-60 - zc) / self.rise)) - 1, int(np.ceil((length + 60 - zc) / self.rise)) + 2)
        xyz = np.vstack([self.xyz @ _rotz(np.radians(k * self.twist)).T + [0, 0, k * self.rise] for k in ks])
        w = np.tile(self.w, len(ks))
        keep = (xyz[:, 2] >= 0) & (xyz[:, 2] < length)
        xyz, w = xyz[keep], w[keep]
        if flip:
            xyz = xyz * np.array([1, -1, -1]) + np.array([0, 0, length])
        return xyz, w
