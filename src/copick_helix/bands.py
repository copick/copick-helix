"""Band-limited helical terms: polarity and registration of thin helical filaments (actin) from per-segment
reconstructions (the "term route").

Each segment is a straight box ``vol[z = t, y = e2, x = e1]`` with the filament axis near the in-plane centre index
and the segment centre at index ``n // 2`` along z (a local reconstruction from the tilt series, or a slice of a
straightened tomogram). It is reduced to the coefficients of the family's strongest helical terms T = (n, m):

    c_T(R) = mean over the measured Phi of F(R, Phi, Z_T) exp(-i n Phi),   Z_T = (m - n twist / 360) / rise,

each term only inside the in-plane resolution band it was selected for. The terms per band come from the family's
atomic model on the same grid (where the signal is expected: an inductive bias, not a template). The axis offset is
refined per segment on the low band only, so the higher bands take no part in choosing it. Terms whose layer line
lies within ``zmin`` / L of the equator are dropped (in a segment of length L they share the equator's main lobe).

Transformations, with the segment centre as origin: rotation by phi0 about z: c -> c exp(-i n phi0); shift by z0:
c -> c exp(-2 pi i Z z0); polarity flip (180 deg about x through the centre): c -> (-1)^n conj(c).

Polarity comes from a reference built from the data in term space (no model). From random polarity starts, every
segment is aligned (phi0 over 360 deg, z0 over one rise, since the helix makes (phi0, z0) and (phi0 + twist, z0 + rise)
equivalent) to the leave-one-filament-out reference in both polarities. Each filament takes the polarity with the
larger summed score, the reference becomes the mean of the aligned segments, and this repeats until no filament
changes. Per segment, D = S(as is) - S(flipped) against the final leave-one-out reference gives each filament's call,
consistency, odd / even halves and z score.

Everything is repeated on phase-scrambled decoys of the same segments. The decoys' statistics are the false-positive
baseline: polarity counts as detected only when the data beat the decoys, in odd / even halves agreement or in the
number of confident filaments. The family's model only names the consensus orientation (plus / minus).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .fourier import tukey

BANDS = {"lo": (300.0, 40.0), "mid": (40.0, 28.0), "hi": (28.0, 20.0)}


@dataclass
class TermConfig:
    use: tuple = ("lo", "mid")  # bands whose terms carry the polarity score
    bands: dict = field(default_factory=lambda: dict(BANDS))
    r_out: float = 60.0  # A, soft radial mask of the segment cross-section
    half_wedge: float = 44.0  # deg, symmetric measured region about the specimen plane
    zmin: float = 2.5  # drop terms with |Z| L < zmin
    max_shift: float = 12.0  # A, axis offset search (+-)
    shift_step: float = 2.0
    starts: int = 5  # random polarity starts
    n_iter: int = 10
    nphi: int = 180
    nz: int = 12  # z0 steps over one rise
    min_count: int = 8  # measured Phi samples for a coefficient
    cover: float = 0.85  # term selection: share of a band's model power covered
    max_terms: int = 12
    nmax: int = 15  # term selection: largest Bessel order considered
    z_seed: float = 3.0  # seed: filament z score
    min_stability: float = 0.8  # seed: fraction of starts agreeing
    decoys: bool = True
    fit_sample: int = 200  # segments used to fit the dataset's rise and twist
    fit_range: float = 0.05  # +- relative Z range scanned per term
    seed: int = 0


@dataclass
class SegmentRef:
    """One segment: ``path`` (stem of NPY + JSON with step_A, beam, tilt_axis) or an in-memory volume."""

    filament: str
    index: int
    path: str | None = None
    vol: np.ndarray | None = None
    step: float | None = None
    beam: np.ndarray | None = None
    tilt: np.ndarray | None = None

    def load(self):
        if self.path is None:
            return self.vol, self.step, np.asarray(self.beam, float), np.asarray(self.tilt, float)
        m = json.load(open(self.path + ".json"))
        v = np.load(self.path + ".npy").astype(np.float32) * float(m.get("protein_sign", 1.0))
        return v, float(m["step_A"]), np.asarray(m["beam"], float), np.asarray(m["tilt_axis"], float)


def phase_randomise(v: np.ndarray, rng) -> np.ndarray:
    """Same power spectrum, random phases: a segment with no helical order (the decoy)."""
    F = np.fft.rfftn(v)
    ph = np.exp(1j * np.angle(np.fft.rfftn(rng.normal(size=v.shape))))
    return np.fft.irfftn(np.abs(F) * ph, s=v.shape).astype(np.float32)


class Window:
    """Per-slice polar Fourier samples ``G[s, R, Phi]`` of one segment, and its layer-line planes at any Z."""

    def __init__(self, vol, step, r_out, beam, tilt, half_wedge=44.0, r_band=(1 / 300.0, 1 / 12.0), npad=None,
                 nphi=180):
        n, ny, nx = vol.shape
        self.step, self.n, self.L = step, n, n * step
        self.beam, self.tilt, self.half_wedge = np.asarray(beam, float), np.asarray(tilt, float), half_wedge
        npad = npad or max(162, int(np.ceil(2.6 * ny / 2)) * 2)
        self.phi = np.arange(nphi) * 2 * np.pi / nphi
        c = (np.arange(ny) - ny // 2) * step
        rr = np.hypot(c[:, None], c[None, :])
        w = np.clip((r_out - rr) / 10.0, 0, 1)
        x = (vol - vol.mean()) * (0.5 - 0.5 * np.cos(np.pi * w))[None] * tukey(n)[:, None, None]
        o = (npad - ny) // 2
        pad = np.zeros((n, npad, npad), np.float32)
        pad[:, o:o + ny, o:o + nx] = x
        pad = np.roll(pad, (-(o + ny // 2), -(o + nx // 2)), axis=(1, 2))
        F = np.fft.fftshift(np.fft.fft2(pad, axes=(1, 2)), axes=(1, 2))
        kf = np.fft.fftshift(np.fft.fftfreq(npad, step))
        dk = 1.0 / (npad * step)
        self.R = np.arange(np.ceil(r_band[0] / dk), np.floor(r_band[1] / dk) + 1) * dk
        self.kx = self.R[:, None] * np.cos(self.phi)[None, :]
        self.ky = self.R[:, None] * np.sin(self.phi)[None, :]
        d = kf[1] - kf[0]
        iy, ix = (self.ky - kf[0]) / d, (self.kx - kf[0]) / d
        self.G = np.stack([map_coordinates(f.real, [iy, ix], order=1) + 1j * map_coordinates(f.imag, [iy, ix], order=1)
                           for f in F]).astype(np.complex64)
        self.s = (np.arange(n) - n // 2) * step  # segment centre at index n // 2 (the box centre)
        self._masks = {}

    def mask(self, Z):
        k = round(Z * 1e7)
        if k not in self._masks:
            b = self.beam / np.linalg.norm(self.beam)
            y = self.tilt - b * (self.tilt @ b)
            y /= np.linalg.norm(y)
            x = np.cross(y, b)
            kb = self.kx * b[0] + self.ky * b[1] + Z * b[2]
            kxx = self.kx * x[0] + self.ky * x[1] + Z * x[2]
            ang = (np.degrees(np.arctan2(kb, kxx)) + 90) % 180 - 90
            self._masks[k] = np.abs(ang) <= self.half_wedge
        return self._masks[k]

    def plane(self, Z, rows=slice(None)):
        return np.tensordot(np.exp(-2j * np.pi * Z * self.s).astype(np.complex64), self.G[:, rows], axes=(0, 0))

    def shift_phase(self, dx, dy, rows=slice(None)):
        """Moves the analysis axis to (dx along e1, dy along e2) A."""
        return np.exp(2j * np.pi * (self.kx[rows] * dx + self.ky[rows] * dy))

    def band_rows(self, band):
        lo, hi = band
        idx = np.where((self.R >= 1 / lo) & (self.R < 1 / hi))[0]
        return slice(idx[0], idx[-1] + 1) if len(idx) else slice(0, 0)


def _order(P, M, n, phi, ph=1.0):
    """Single-order projection over the measured Phi: (c(R), count(R))."""
    cnt = M.sum(1)
    c = (P * ph * np.exp(-1j * n * phi)[None, :] * M).sum(1) / np.maximum(cnt, 1)
    return c, cnt


def z_of(n, m, rise, twist):
    return (m - n * twist / 360.0) / rise


def refine_offset(win: Window, lo_terms, max_shift, shift_step=2.0, band=(300.0, 40.0)):
    """Axis offset (dx along e1, dy along e2) A maximising the low-band power of the low-band terms' own orders."""
    rows = win.band_rows(band)
    planes = [(n, win.plane(Z, rows), win.mask(Z)[rows]) for n, Z in lo_terms]
    if not planes:
        return 0.0, 0.0
    g = np.arange(-max_shift, max_shift + 1e-6, shift_step)
    best, arg = -1.0, (0.0, 0.0)
    for dx in g:
        for dy in g:
            ph = win.shift_phase(dx, dy, rows)
            tot = 0.0
            for n, P, M in planes:
                c, cnt = _order(P, M, n, win.phi, ph)
                tot += float(np.sum(np.abs(c) ** 2 * cnt))
            if tot > best:
                best, arg = tot, (float(dx), float(dy))
    return arg


def select_terms(model_vol, step, r_out, rise, twist, cfg: TermConfig) -> dict:
    """Per band, the (n, m) terms carrying ``cfg.cover`` of the model's power in that band (at most cfg.max_terms),
    measured with the same single-order projection as the data (no wedge). Only terms the segment can resolve count
    (|Z| L >= cfg.zmin, L the model's length): the actin crossover term (-2, 1) holds 30 % of the low band's power, and
    counting it towards the cover pushed out (3, -1), which carries much of the polarity. n = 0 is excluded (no
    polarity)."""
    win = Window(model_vol, step, r_out, (1.0, 0, 0), (0, 1.0, 0), half_wedge=90.0)
    rows = {b: win.band_rows(v) for b, v in cfg.bands.items()}
    zmax = 1.0 / min(v[1] for v in cfg.bands.values())
    cand = []
    for m in range(-60, 61):
        for n in range(-cfg.nmax, cfg.nmax + 1):
            Z = z_of(n, m, rise, twist)
            if n == 0 or Z <= 1e-9 or Z > zmax or Z * win.L < cfg.zmin:
                continue
            c, cnt = _order(win.plane(Z), win.mask(Z), n, win.phi)
            p = np.abs(c) ** 2 * cnt
            cand.append((n, m, {b: float(p[r].sum()) for b, r in rows.items()}))
    out = {}
    for b in cfg.bands:
        tot = sum(c[2][b] for c in cand)
        chosen, acc = [], 0.0
        for c in sorted(cand, key=lambda c: -c[2][b]):
            if acc >= cfg.cover * tot or len(chosen) >= cfg.max_terms:
                break
            chosen.append((c[0], c[1]))
            acc += c[2][b]
        out[b] = chosen
    return out


def _keys(terms: dict, use, rise, twist, L, zmin):
    keys = [(b, n, m) for b in use for n, m in terms[b] if abs(z_of(n, m, rise, twist)) * L >= zmin]
    return keys, np.array([k[1] for k in keys]), np.array([z_of(k[1], k[2], rise, twist) for k in keys])


def coefficients(win: Window, keys, Zs, bands, ph=1.0, min_count=8):
    """(K, nR) complex: each key's single-order coefficient over its own band's R (nan elsewhere / unmeasured)."""
    C = np.full((len(keys), len(win.R)), np.nan + 0j, np.complex128)
    for i, ((b, n, _), Z) in enumerate(zip(keys, Zs)):
        rows = win.band_rows(bands[b])
        c, cnt = _order(win.plane(Z, rows), win.mask(Z)[rows], n, win.phi, ph if np.isscalar(ph) else ph[rows])
        C[i, rows] = np.where(cnt >= min_count, c, np.nan)
    return C


def enrichment_sums(win: Window, terms_nz, bands, ph=1.0):
    """Per band: (on, off) power of each term's own order on its plane against the planes Z +- k / L (k = 2, 3)."""
    out = {}
    for b, band in bands.items():
        rows = win.band_rows(band)
        p = ph if np.isscalar(ph) else ph[rows]
        on = off = 0.0
        for n, Z in terms_nz.get(b, []):
            def pw(z):
                c, cnt = _order(win.plane(z, rows), win.mask(z)[rows], n, win.phi, p)
                return np.abs(c) ** 2 * cnt
            on += float(pw(Z).sum())
            off += float(np.mean([np.sqrt(np.maximum(pw(Z - k / win.L), 1e-30) * np.maximum(pw(Z + k / win.L), 1e-30))
                                  for k in (2.0, 3.0)], axis=0).sum())
        out[b] = (on, off)
    return out


# --------------------------------------------------------------------------------------------------- alignment


def flip(C, n):
    """180 deg about x through the segment centre."""
    return np.where((n % 2)[:, None] == 0, 1.0, -1.0) * np.conj(C) if C.ndim == 2 else \
        np.where((n % 2)[None, :, None] == 0, 1.0, -1.0) * np.conj(C)


class Aligner:
    def __init__(self, n, Z, rise, nphi=180, nz=12):
        self.n, self.Z = n, Z
        self.phis = np.arange(nphi) * 2 * np.pi / nphi
        self.zs = np.arange(nz) * rise / nz
        self.Ephi = np.exp(1j * np.outer(self.phis, n))  # (nphi, K)
        self.Ez = np.exp(2j * np.pi * np.outer(Z, self.zs))  # (K, nz)

    def best(self, ref, C):
        """ref (P, K, R) or (K, R); C (P, K, R): best normalised score, phi0, z0 per particle."""
        ref = np.broadcast_to(ref, C.shape)
        ok = np.isfinite(ref) & np.isfinite(C)
        cross = np.where(ok, np.conj(ref) * C, 0).sum(-1)  # (P, K)
        na = np.where(ok, np.abs(C) ** 2, 0).sum((1, 2))
        nb = np.where(ok, np.abs(ref) ** 2, 0).sum((1, 2))
        tot = np.real((cross[:, None, :] * self.Ephi[None]) @ self.Ez)  # (P, nphi, nz)
        flat = tot.reshape(len(C), -1)
        i = np.argmax(flat, 1)
        s = flat[np.arange(len(C)), i] / np.sqrt(np.maximum(na * nb, 1e-300))
        s = np.where((na > 0) & (nb > 0), s, np.nan)
        ip, iz = np.unravel_index(i, tot.shape[1:])
        return s, self.phis[ip], self.zs[iz]

    def apply(self, C, phi0, z0):
        return C * np.exp(1j * (self.n[None, :, None] * phi0[:, None, None] + 2 * np.pi * self.Z[None, :, None] * z0[:, None, None]))


def _loo_refs(A, fil, nfil):
    """Leave-one-filament-out mean of the aligned coefficients A (P, K, R); nan where nothing is measured."""
    ok = np.isfinite(A)
    S = np.zeros((nfil,) + A.shape[1:], np.complex128)
    N = np.zeros((nfil,) + A.shape[1:])
    np.add.at(S, fil, np.where(ok, A, 0))
    np.add.at(N, fil, ok)
    St, Nt = S.sum(0), N.sum(0)
    cnt = Nt[None] - N
    return np.where(cnt > 0, (St[None] - S) / np.maximum(cnt, 1), np.nan)


def iterate(C, fil, nfil, al: Aligner, rng, n_iter=10):
    """Data-built reference from a random polarity start. Returns (pol (nfil,), objective, margin (nfil,))."""
    pol = rng.choice([-1, 1], nfil)
    Cf = flip(C, al.n)
    A = np.where((pol[fil] < 0)[:, None, None], Cf, C)
    margin = np.zeros(nfil)
    obj = 0.0
    for it in range(n_iter):
        refs = _loo_refs(A, fil, nfil)[fil]
        su, pu, zu = al.best(refs, C)
        sd, pd_, zd = al.best(refs, Cf)
        SU, SD, cnt = np.zeros(nfil), np.zeros(nfil), np.zeros(nfil)
        np.add.at(SU, fil, np.nan_to_num(su))
        np.add.at(SD, fil, np.nan_to_num(sd))
        np.add.at(cnt, fil, 1)
        new = np.where(SU >= SD, 1, -1)
        margin = (SU - SD) / np.maximum(cnt, 1)
        obj = float(np.maximum(SU, SD).sum())
        up = (new[fil] > 0)
        A = np.where(up[:, None, None], al.apply(C, pu, zu), al.apply(Cf, pd_, zd))
        changed = int((new != pol).sum())
        pol = new
        if it > 0 and changed == 0:
            break
    return pol, obj, margin, A


@dataclass
class TermResult:
    table: pd.DataFrame  # per filament
    segments: pd.DataFrame  # per segment: registration and score difference
    summary: dict
    decoy_table: pd.DataFrame | None = None


def _filament_stats(D, fil, names):
    rows = []
    for f, name in enumerate(names):
        v = D[fil == f]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            rows.append({"filament": name, "segments": 0})
            continue
        s = np.sign(v.sum()) or 1.0
        z = abs(v.mean()) / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) >= 3 and v.std(ddof=1) > 0 else np.nan
        rows.append({"filament": name, "segments": len(v), "sign": int(s), "consistency": float(np.mean(np.sign(v) == s)),
                     "halves_agree": bool(np.sign(v[0::2].sum()) == np.sign(v[1::2].sum())) if len(v) >= 4 else None,
                     "z": float(z)})
    return pd.DataFrame(rows)


def assign(C, fil, names, al: Aligner, cfg: TermConfig, model_C=None, rng_seed=0):
    """All starts, consensus, final per-segment scores against the leave-one-out reference, orientation by the model.
    Returns (table, per-segment arrays dict)."""
    nfil = len(names)
    rng = np.random.default_rng(rng_seed)
    runs = [iterate(C, fil, nfil, al, rng, cfg.n_iter) for _ in range(max(cfg.starts, 1))]
    best = int(np.argmax([r[1] for r in runs]))
    pol, _, margin, A = runs[best]
    stab = []
    for r in runs:
        a = np.mean(r[0] == pol)
        stab.append(r[0] * (1 if a >= 0.5 else -1) == pol)
    stability = np.mean(stab, axis=0)
    # final: every segment against its leave-one-out reference, both polarities
    refs = _loo_refs(A, fil, nfil)[fil]
    Cf = flip(C, al.n)
    su, pu, zu = al.best(refs, C)
    sd, pd_, zd = al.best(refs, Cf)
    D = su - sd
    orient = 1
    if model_C is not None:  # the model only names the consensus orientation
        ok = np.isfinite(A)
        mean = np.where(ok.sum(0) > 0, np.where(ok, A, 0).sum(0) / np.maximum(ok.sum(0), 1), np.nan)
        s_up = al.best(model_C[None], mean[None])[0][0]
        s_dn = al.best(flip(model_C, al.n)[None], mean[None])[0][0]
        orient = 1 if s_up >= s_dn else -1
    flip_seg = pol[fil] < 0
    phi0 = np.where(flip_seg, pd_, pu)
    z0 = np.where(flip_seg, zd, zu)
    score = np.where(flip_seg, sd, su)
    if orient < 0:  # flipping the reference: every filament's polarity inverts, (phi0, z0) -> (-phi0, -z0)
        pol, margin, D, flip_seg, phi0, z0 = -pol, -margin, -D, ~flip_seg, -phi0, -z0
    stats = _filament_stats(D, fil, names)
    stats["pol"] = pol
    stats["margin"] = margin
    stats["stability"] = stability
    seg = {"flip": flip_seg, "roll_deg": np.degrees(-phi0) % 360.0, "shift_A": -z0, "score": score, "D": D,
           "orient": orient}
    return stats, seg


# --------------------------------------------------------------------------------------------------- per filament

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _map(fn, jobs, workers):
    """Map over worker processes (spawned, single-threaded BLAS / FFT so N workers use N cores)."""
    if workers <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    import sys

    # spawn needs an importable __main__ (not a script on stdin); fork otherwise
    method = "spawn" if getattr(sys.modules.get("__main__"), "__file__", None) else "fork"
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context(method)) as ex:
            return list(ex.map(fn, jobs, chunksize=1))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v



def _filament_job(args):
    refs, keys, Zs, nvec, rise, nz_terms, lo_nz, mvol, cfg, decoy_seed = args
    rng = np.random.default_rng(decoy_seed)
    al = Aligner(nvec, Zs, rise, cfg.nphi, cfg.nz)
    mwin = None
    out = []
    for ref in refs:
        vol, step, beam, tilt = ref.load()
        if mvol is not None and mwin is None:
            mwin = Window(mvol, step, cfg.r_out, beam, tilt, cfg.half_wedge)
        res = {}
        MC = None
        if mwin is not None:  # the model through this segment's measured region (diagnostic: model-reference score)
            mwin.beam, mwin.tilt, mwin._masks = np.asarray(beam, float), np.asarray(tilt, float), {}
            MC = coefficients(mwin, keys, Zs, cfg.bands, 1.0, cfg.min_count)
        for decoy in ((False, True) if cfg.decoys else (False,)):
            v = phase_randomise(vol, rng) if decoy else vol
            win = Window(v, step, cfg.r_out, beam, tilt, cfg.half_wedge)
            lo = [(n, Z) for n, Z in lo_nz if abs(Z) * win.L >= cfg.zmin]
            dx, dy = refine_offset(win, lo, cfg.max_shift, cfg.shift_step, cfg.bands["lo"])
            ph = win.shift_phase(dx, dy)
            C = coefficients(win, keys, Zs, cfg.bands, ph, cfg.min_count)
            enr = enrichment_sums(win, nz_terms, cfg.bands, ph)
            dm = np.nan
            if MC is not None:
                su = al.best(MC[None], C[None])[0][0]
                sd = al.best(flip(MC, nvec)[None], C[None])[0][0]
                dm = float(su - sd)
            res[decoy] = (C, (dx, dy), enr, dm)
        out.append((ref.filament, ref.index, res))
    return out


def _fit_job(args):
    """Pooled power of each low-band term's order along Z around its predicted plane (offset refined first)."""
    refs, nz, rel, cfg = args
    acc = {t: np.zeros(len(rel)) for t in range(len(nz))}
    for ref in refs:
        vol, step, beam, tilt = ref.load()
        win = Window(vol, step, cfg.r_out, beam, tilt, cfg.half_wedge)
        dx, dy = refine_offset(win, nz, cfg.max_shift, cfg.shift_step, cfg.bands["lo"])
        rows = win.band_rows(cfg.bands["lo"])
        ph = win.shift_phase(dx, dy, rows)
        for t, (n, Z) in enumerate(nz):
            for i, r in enumerate(rel):
                z = Z * (1 + r)
                c, cnt = _order(win.plane(z, rows), win.mask(z)[rows], n, win.phi, ph)
                acc[t][i] += float(np.sum(np.abs(c) ** 2 * cnt))
    return acc


def fit_axial_scale(refs, lo_terms, rise0, twist0, cfg: TermConfig, workers=1):
    """The dataset's rise at the model's twist: every layer line moves by the same factor when only the axial scale
    differs (a pixel-size mismatch, or a uniformly stretched lattice), so each low-band term's pooled power peak along
    Z gives one estimate of 1 / scale; their power-weighted mean in log is the fit. Twist variation between filaments
    (+-0.5 deg in actin) moves a term by << 1 Z bin and is not fitted. Falls back to rise0 when no peak is resolved.
    Returns (rise, {term: relative Z offset at its peak})."""
    sample = refs[:: max(1, len(refs) // cfg.fit_sample)][: cfg.fit_sample]
    if not sample:
        return rise0, {}
    v0, s0, _, _ = sample[0].load()
    L = v0.shape[0] * s0
    terms = [t for t in lo_terms if abs(z_of(t[0], t[1], rise0, twist0)) * L >= 4.0]
    if not terms:
        return rise0, {}
    nz = [(n, z_of(n, m, rise0, twist0)) for n, m in terms]
    rel = np.linspace(-cfg.fit_range, cfg.fit_range, 41)
    chunks = [c for c in (sample[i::max(workers, 1)] for i in range(max(workers, 1))) if c]
    acc = np.zeros((len(terms), len(rel)))
    for part in _map(_fit_job, [(c, nz, rel, cfg) for c in chunks], workers):
        for t in range(len(terms)):
            acc[t] += part[t]
    est, w, peaks = [], [], {}
    for t, p in enumerate(acc):
        i = int(np.argmax(p))
        if i in (0, len(p) - 1):
            continue
        d = 0.5 * (p[i - 1] - p[i + 1]) / (p[i - 1] - 2 * p[i] + p[i + 1])
        r = rel[i] + d * (rel[1] - rel[0])
        peaks[terms[t]] = float(r)
        est.append(np.log1p(r))
        w.append(p[i] / max(np.median(p), 1e-30))
    if not est:
        return rise0, peaks
    scale = float(np.exp(np.average(est, weights=w)))  # Z_data / Z_model
    return float(rise0 / scale), peaks


def analyse(segments: dict, family, cfg: TermConfig | None = None, workers: int = 1, label: bool = True,
            rise: float | None = None, log=print) -> TermResult:
    """The term route over a dataset. ``segments``: {filament name: [SegmentRef, ...]}, one grid for all segments
    (step, box). ``rise``: skip the fit and use this rise (A)."""
    cfg = cfg or TermConfig(r_out=family.term_r_out)
    names = [k for k, v in segments.items() if v]
    refs = [r for k in names for r in segments[k]]
    v0, step, _, _ = refs[0].load()
    n_s, n_in = v0.shape[0], v0.shape[1]
    L = n_s * step
    mvol, mrise, mtwist = family.term_model(step, n_in, L)
    terms = select_terms(mvol, step, cfg.r_out, mrise, mtwist, cfg)
    log("terms per band (n, m): " + "; ".join(f"{b}: {v}" for b, v in terms.items()))
    peaks = {}
    if rise is None:
        rise, peaks = fit_axial_scale(refs, terms["lo"], mrise, mtwist, cfg, workers)
    twist = mtwist
    log(f"rise {rise:.3f} A (model {mrise:.3f}), twist {twist:.2f} deg")
    # the model with its rise matched to the data's (Z in data units), for orientation and the model diagnostic
    sc = mrise / rise
    mvol_s = family.term_model(step * sc, n_in, L * sc)[0][:n_s] if label else None
    keys, nvec, Zs = _keys(terms, cfg.use, rise, twist, L, cfg.zmin)
    nz_terms = {b: [(n, z_of(n, m, rise, twist)) for n, m in terms[b] if abs(z_of(n, m, rise, twist)) * L >= cfg.zmin]
                for b in cfg.bands}
    jobs = [(segments[k], keys, Zs, nvec, rise, nz_terms, nz_terms["lo"], mvol_s, cfg, cfg.seed + 1000 + i)
            for i, k in enumerate(names)]
    per = [x for part in _map(_filament_job, jobs, workers) for x in part]
    model_C = None
    if label:
        mwin = Window(mvol_s, step, cfg.r_out, (1.0, 0, 0), (0, 1.0, 0), half_wedge=90.0)
        model_C = coefficients(mwin, keys, Zs, cfg.bands)
    al = Aligner(nvec, Zs, rise, cfg.nphi, cfg.nz)
    fidx = {k: i for i, k in enumerate(names)}
    fil = np.array([fidx[p[0]] for p in per])
    out = {}
    for decoy in ((False, True) if cfg.decoys else (False,)):
        C = np.stack([p[2][decoy][0] for p in per])
        stats, seg = assign(C, fil, names, al, cfg, model_C if not decoy else None, cfg.seed)
        enr = {b: (sum(p[2][decoy][2][b][0] for p in per), sum(p[2][decoy][2][b][1] for p in per)) for b in cfg.bands}
        dm = np.array([p[2][decoy][3] for p in per])
        segdf = pd.DataFrame({"filament": [p[0] for p in per], "segment": [p[1] for p in per],
                              "flip": seg["flip"], "roll_deg": seg["roll_deg"], "shift_A": seg["shift_A"],
                              "offset_e1_A": [p[2][decoy][1][0] for p in per],
                              "offset_e2_A": [p[2][decoy][1][1] for p in per], "score": seg["score"], "D": seg["D"],
                              "D_model": dm})
        mstats = _filament_stats(dm, fil, names) if label else None
        out[decoy] = (stats, segdf, enr, seg["orient"], mstats)
    stats, segdf, enr, orient, mstats = out[False]
    summary = {"rise": rise, "twist": twist, "model_rise": mrise, "model_twist": mtwist,
               "fit_peaks_rel": {str(k): v for k, v in peaks.items()},
               "terms": {b: [list(t) for t in v] for b, v in terms.items()}, "keys_used": [list(k) for k in keys],
               "segment_length_A": L, "step_A": step,
               "enrichment": {b: {"data": enr[b][0] / max(enr[b][1], 1e-30)} for b in cfg.bands}}
    if cfg.decoys:
        dstats, _, denr, _, dmstats = out[True]
        for b in cfg.bands:
            summary["enrichment"][b]["decoy"] = denr[b][0] / max(denr[b][1], 1e-30)
        summary["data"] = _dataset_stats(stats, cfg)
        summary["decoy"] = _dataset_stats(dstats, cfg)
        h_d, n_d = summary["data"]["halves_agree"], summary["data"]["halves_n"]
        h_q, n_q = summary["decoy"]["halves_agree"], summary["decoy"]["halves_n"]
        p = (h_d + h_q) / max(n_d + n_q, 1)
        se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / max(n_d, 1) + 1 / max(n_q, 1)))
        diff = h_d / max(n_d, 1) - h_q / max(n_q, 1)
        lat = summary["enrichment"]["lo"]
        summary["lattice_detected"] = bool(lat["data"] > 1.25 * lat["decoy"])
        # two tests against the decoys: odd / even halves agreeing more often, or more confident filaments (z >=
        # z_seed; counts compared as Poisson); either one carries the detection
        z_d, z_q = summary["data"]["z_ge_seed"], summary["decoy"]["z_ge_seed"]
        z_ok = (z_d - z_q) > 2 * np.sqrt(max(z_d + z_q, 1))
        summary["polarity_detected"] = bool(summary["lattice_detected"] and (diff > 2 * se or z_ok))
        summary["halves_excess"] = {"diff": float(diff), "se": float(se)}
        summary["confident_excess"] = {"data": z_d, "decoy": z_q, "two_sigma": float(2 * np.sqrt(max(z_d + z_q, 1)))}
        if label:  # diagnostic only: the same statistics from the model-reference score
            summary["model_reference"] = {"data": _dataset_stats(mstats, cfg), "decoy": _dataset_stats(dmstats, cfg)}
    else:
        summary["lattice_detected"] = summary["polarity_detected"] = None
    if label and mstats is not None:
        both = stats.set_index("filament").pol * mstats.set_index("filament").sign
        summary["data_vs_model_calls"] = float(np.mean(both.dropna() > 0))
    stats["call"] = [("plus" if p > 0 else "minus") if label else ("A" if p > 0 else "B") for p in stats["pol"]]
    stats["seed"] = ((stats.z >= cfg.z_seed) & (stats.halves_agree == True) & (stats.stability >= cfg.min_stability)  # noqa: E712
                     & bool(summary["polarity_detected"] if summary["polarity_detected"] is not None else True))
    stats["rise"], stats["twist"], stats["segment_length_A"] = rise, twist, L
    if cfg.decoys:
        q = out[True][0]
        n_seed_decoy = int(((q.z >= cfg.z_seed) & (q.halves_agree == True) & (q.stability >= cfg.min_stability)).sum())  # noqa: E712
        summary["seeds"] = {"data": int(stats.seed.sum()), "decoy_at_same_rule": n_seed_decoy}
    return TermResult(stats, segdf, summary, out[True][0] if cfg.decoys else None)


def _dataset_stats(t, cfg):
    h = t.halves_agree.dropna() if "halves_agree" in t else pd.Series(dtype=bool)
    return {"filaments": int(len(t)), "halves_agree": int(h.astype(bool).sum()), "halves_n": int(len(h)),
            "z_ge_seed": int((t.z >= cfg.z_seed).sum()), "consistency_median": float(t.consistency.median())}


def bundle_pairs(geometry: dict, calls: dict, names, max_dist=150.0, min_cos=0.9, min_overlap=200.0):
    """Pairs of filaments in one run running side by side (within ``max_dist`` A over at least ``min_overlap`` A,
    |cos| >= min_cos): (a, b, same_polarity) with the pair's relative direction taken into account.
    ``geometry``: name -> (run, centres (n, 3), tangents (n, 3)) sampled about every 50 A; calls: name -> +-1."""
    from scipy.spatial import cKDTree

    out = []
    names = [n for n in names if n in geometry and n in calls]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if geometry[a][0] != geometry[b][0]:
                continue
            ca, cb = geometry[a][1], geometry[b][1]
            d, k = cKDTree(cb).query(ca)
            close = d <= max_dist
            if close.sum() * 50.0 < min_overlap:
                continue
            cosv = np.sum(geometry[a][2][close] * geometry[b][2][k[close]], 1)
            if np.median(np.abs(cosv)) < min_cos:
                continue
            g = np.sign(np.median(cosv))
            out.append((a, b, bool(g * calls[a] * calls[b] > 0)))
    return out


def segment_refs_from_dir(root: str) -> dict:
    """{filament: [SegmentRef]} from DIR/<filament>/segNN.npy + .json (as written by tiltseries.reconstruct)."""
    out = {}
    for f in sorted(os.listdir(root)):
        d = os.path.join(root, f)
        if not os.path.isdir(d):
            continue
        stems = sorted(p[:-4] for p in os.listdir(d) if p.startswith("seg") and p.endswith(".npy"))
        refs = []
        for s in stems:
            m = json.load(open(os.path.join(d, s + ".json")))
            refs.append(SegmentRef(f, int(m.get("segment", int(s[3:]))), path=os.path.join(d, s)))
        if refs:
            out[f] = refs
    return out
