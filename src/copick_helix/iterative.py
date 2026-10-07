"""Iterative, data-built reference: per-filament polarity with 2-parameter segment alignment, in Bessel space.

Each segment becomes g(r, n, Z) (symmetric measured-region mask, band limits, cylindrical resampling), restricted to
the family's Fourier support (its layer lines and allowed orders). Restricting to the support is what makes a
self-built reference converge from random starts: outside it the data hold only missing-wedge leakage (fixed to
the beam, not the lattice) and off-centring leakage, which a reference otherwise locks onto.

Alignment of a segment to a reference G is one inverse FFT over (n, Z) for rotation about and shift along the axis;
a polarity flip is conj(g). Polarity is a per-filament variable; rotation and shift are per segment. The reference is
the average of the aligned, polarity-applied segments, and each filament is scored against the reference without its
own segments. Starts: random assignments, single random segments, optionally a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fourier import bessel_coefficients, cylindrical, measured_mask_3d
from .helix import Support
from .invariants import SegmentGeometry


@dataclass
class IterativeConfig:
    rmin: float = 60.0
    rmax: float = 165.0
    dr: float = 2.5
    nphi: int = 256
    nmax: int = 56
    zmax: float = 1 / 20.0
    lp: float | None = 20.0
    hp: float | None = 400.0
    half_wedge: float = 45.0
    support: Support | None = None  # family support; None keeps everything except the prototype's equatorial rule
    support_z_bins: float = 1.0  # half-width (in Z bins) of a layer line in the support mask
    eq_band: float = 1 / 250.0  # prototype rule (support None): drop n = 0 and |n| <= eq_nmax for |Z| < eq_band
    eq_nmax: int = 8
    random_starts: int = 20
    seed_starts: int = 5
    max_iter: int = 40


def segment_g(seg: np.ndarray, geom: SegmentGeometry, step: float, cfg: IterativeConfig):
    """Full g(r, n, Z) of one segment (seg[z, y, x], protein positive)."""
    seg = seg - seg.mean()
    m = measured_mask_3d(seg.shape, step, geom.beam, geom.tilt_axis, cfg.half_wedge, cfg.lp, cfg.hp)
    v = np.fft.irfftn(np.fft.rfftn(seg) * m, s=seg.shape).astype(np.float32)
    c = seg.shape[1] // 2
    r, _, cyl = cylindrical(v, step, cfg.rmin, cfg.rmax, cfg.dr, cfg.nphi, centre=(c, c))
    n, Z, g = bessel_coefficients(cyl, step)
    return r, n, Z, g


class Space:
    """Cropped (n, Z) grid of one segment length, alignment primitives and the support mask."""

    def __init__(self, r, n_full, Z_full, cfg: IterativeConfig):
        self.cfg = cfg
        self.w = r.astype(np.float32)
        self.nphi, self.nz = len(n_full), len(Z_full)
        self.keep_n = np.where(np.abs(n_full) <= cfg.nmax)[0]
        self.keep_z = np.where(np.abs(Z_full) <= cfg.zmax + 1e-9)[0]
        self.NN = n_full[self.keep_n].astype(np.float32)
        self.ZZ = Z_full[self.keep_z].astype(np.float32)
        self.step = 1.0 / (len(Z_full) * (Z_full[1] - Z_full[0])) if len(Z_full) > 1 else 1.0
        if cfg.support is not None:
            tol = cfg.support_z_bins * abs(Z_full[1] - Z_full[0]) + 1e-9
            self.mask = cfg.support.nz_mask(self.NN.astype(int), self.ZZ, tol)
        else:
            self.mask = np.ones((len(self.NN), len(self.ZZ)), bool)
            self.mask[self.NN == 0, :] = False
            self.mask[(np.abs(self.NN) <= cfg.eq_nmax)[:, None] & (np.abs(self.ZZ) < cfg.eq_band)[None, :]] = False

    def crop(self, g):
        g = g[..., self.keep_n, :][..., self.keep_z]
        return (g * self.mask).astype(np.complex64)

    def norm(self, g):
        p = np.einsum("r,...rnz->...", self.w, (np.abs(g) ** 2).astype(np.float32))
        return g / np.sqrt(np.maximum(p, 1e-30))[(...,) + (None,) * 3]

    def cross(self, G, g):
        return np.einsum("r,rnz,...rnz->...nz", self.w, G, np.conj(g), optimize=True)

    def power(self, G):
        return float(np.einsum("r,rnz->", self.w, (np.abs(G) ** 2).astype(np.float32)))

    def ccmap(self, cross):
        full = np.zeros(cross.shape[:-2] + (self.nphi, self.nz), np.complex64)
        full[..., self.keep_n[:, None], self.keep_z[None, :]] = cross
        return np.real(np.fft.ifft2(full, axes=(-2, -1))) * (self.nphi * self.nz)

    def phase(self, k, l, dz):
        phi = 2 * np.pi * np.asarray(k, np.float32) / self.nphi
        z = np.asarray(l, np.float32) * dz
        return np.exp(-1j * (phi[:, None, None] * self.NN[None, :, None]
                             + 2 * np.pi * z[:, None, None] * self.ZZ[None, None, :])).astype(np.complex64)

    def best(self, G, g):
        cc = self.ccmap(self.cross(G, g)) / np.sqrt(max(self.power(G), 1e-30))
        flat = cc.reshape(cc.shape[0], -1)
        i = flat.argmax(1)
        return flat[np.arange(len(i)), i], i // self.nz, i % self.nz

    def cc_max(self, A, B):
        c = self.ccmap(self.cross(A, B[None]))[0]
        return float(c.max() / np.sqrt(max(self.power(A) * self.power(B), 1e-30)))


def _aligned(sp: Space, gs, flip, k, l, dz):
    g = np.where(flip[:, None, None, None], np.conj(gs), gs)
    return g * sp.phase(k, l, dz)[:, None]


def iterate(sp: Space, gs, fil, nfil, pol0, k0, l0, dz, G0=None, fixed_pol=False, max_iter=40):
    pol, k, l = pol0.copy(), k0.copy(), l0.copy()
    hist, stable = [], 0
    margin = np.zeros(nfil)
    seg_margin = np.zeros(len(gs))
    for it in range(max_iter):
        if it == 0 and G0 is not None:
            Gfull, contrib = G0, None
        else:
            a = _aligned(sp, gs, pol[fil] < 0, k, l, dz)
            Gfull = a.sum(0)
            contrib = np.stack([a[fil == f].sum(0) for f in range(nfil)])
        new_pol, new_k, new_l = pol.copy(), k.copy(), l.copy()
        score = np.zeros(nfil)
        for f in range(nfil):
            idx = np.where(fil == f)[0]
            G = Gfull if contrib is None else Gfull - contrib[f]
            cp, kp, lp = sp.best(G, gs[idx])
            cm, km, lm = sp.best(G, np.conj(gs[idx]))
            seg_margin[idx] = cp - cm
            if not fixed_pol:
                new_pol[f] = 1 if cp.sum() >= cm.sum() else -1
            use_p = new_pol[f] > 0
            new_k[idx] = kp if use_p else km
            new_l[idx] = lp if use_p else lm
            score[f] = (cp if use_p else cm).sum()
            margin[f] = cp.sum() - cm.sum()
        changed = int((new_pol != pol).sum())
        moved = float(np.mean((new_k != k) | (new_l != l)))
        pol, k, l = new_pol, new_k, new_l
        hist.append({"iter": it, "objective": float(score.sum() / len(gs)), "pol_changed": changed, "align_moved": moved})
        stable = stable + 1 if changed == 0 else 0
        if (it >= 2 and changed == 0 and moved < 0.02) or stable >= 5:
            break
    a = _aligned(sp, gs, pol[fil] < 0, k, l, dz)
    return {"pol": pol, "k": k, "l": l, "G": a.sum(0) / len(gs), "hist": hist, "margin": margin.copy(),
            "seg_margin": seg_margin.copy(), "objective": hist[-1]["objective"]}


def agree(p, q):
    s = float(np.mean(p == q))
    return (s, 1) if s >= 0.5 else (1 - s, -1)


@dataclass
class IterativeResult:
    calls: pd.DataFrame
    runs: pd.DataFrame
    reference: np.ndarray  # consensus reference g (cropped grid), oriented to 'plus' if a model was given
    strength: dict = field(default_factory=dict)


def assign(g_by_filament: dict[str, np.ndarray], r, n_full, Z_full, cfg: IterativeConfig,
           model_plus_g: np.ndarray | None = None, rng_seed: int = 7, bootstrap_draws: int = 2000) -> IterativeResult:
    """Run all starts and summarise. g_by_filament: name -> full g of its segments (S, R, n, Z), one segment length.
    model_plus_g: optional full g of a model in the 'plus' orientation, used only to orient the final global sign."""
    sp = Space(r, n_full, Z_full, cfg)
    dz = 1.0 / (len(Z_full) * abs(Z_full[1] - Z_full[0]))
    names = list(g_by_filament)
    gs = sp.norm(np.concatenate([sp.crop(g_by_filament[nm]) for nm in names]))
    fil = np.concatenate([np.full(len(g_by_filament[nm]), i) for i, nm in enumerate(names)])
    nfil = len(names)
    rng = np.random.default_rng(rng_seed)
    zeros = np.zeros(len(gs), int)
    runs = {}
    for s in range(cfg.random_starts):
        runs[f"random_{s:02d}"] = iterate(sp, gs, fil, nfil, rng.choice([-1, 1], nfil), zeros, zeros, dz,
                                          max_iter=cfg.max_iter)
    for s in range(cfg.seed_starts):
        i = int(rng.integers(len(gs)))
        G0 = gs[i] if rng.random() < 0.5 else np.conj(gs[i])
        runs[f"seed_{s:02d}"] = iterate(sp, gs, fil, nfil, np.ones(nfil, int), zeros, zeros, dz, G0=G0,
                                        max_iter=cfg.max_iter)
    gm = sp.norm(sp.crop(model_plus_g)[None])[0] if model_plus_g is not None else None
    if gm is not None:
        runs["model_plus"] = iterate(sp, gs, fil, nfil, np.ones(nfil, int), zeros, zeros, dz, G0=gm,
                                     max_iter=cfg.max_iter)
    for res in runs.values():
        if gm is not None:
            cp, cm = sp.cc_max(res["G"], gm), sp.cc_max(res["G"], np.conj(gm))
            res["orient"] = 1 if cp >= cm else -1
        else:
            res["orient"] = 1
    data_runs = [k for k in runs if k.startswith(("random", "seed"))]
    best = max(data_runs, key=lambda k: runs[k]["objective"])
    cons = runs[best]
    if gm is None:  # without a model, orient every run to the consensus
        for res in runs.values():
            res["orient"] = agree(cons["pol"], res["pol"])[1]
    for res in runs.values():
        res["pol_oriented"] = res["pol"] * res["orient"]
    run_rows = [{"start": k, "iterations": len(v["hist"]), "objective": v["objective"],
                 "agree_with_consensus": agree(cons["pol"], v["pol"])[0]} for k, v in runs.items()]

    def strength(G):
        return 1 - sp.cc_max(G, np.conj(G))

    s_rand = [strength(iterate(sp, gs, fil, nfil, rng.choice([-1, 1], nfil), cons["k"], cons["l"], dz,
                               fixed_pol=True, max_iter=cfg.max_iter)["G"]) for _ in range(8)]
    strengths = {"consensus": strength(cons["G"]), "random_fixed_mean": float(np.mean(s_rand)),
                 "random_fixed_std": float(np.std(s_rand))}
    if gm is not None:
        strengths["model"] = strength(gm)

    oriented = np.array([runs[k]["pol_oriented"] for k in data_runs])
    stability = (oriented == cons["pol_oriented"][None]).mean(0)
    seg_margin = cons["seg_margin"] * cons["orient"]
    rows = []
    for f, name in enumerate(names):
        m = seg_margin[fil == f] * cons["pol_oriented"][f]
        draws = rng.choice(m, (bootstrap_draws, len(m)), replace=True).sum(1)
        boot = float(np.mean(draws > 0))
        call = ("plus" if cons["pol_oriented"][f] > 0 else "minus") if gm is not None else \
            ("A" if cons["pol_oriented"][f] > 0 else "B")
        rows.append({"filament": name, "segments": int((fil == f).sum()), "call": call,
                     "loo_margin": float(cons["margin"][f] * cons["orient"] * cons["pol_oriented"][f]),
                     "bootstrap": boot, "start_stability": float(stability[f]),
                     "confident": bool(boot >= 0.95 and stability[f] >= 0.9)})
    G = cons["G"] if cons["orient"] > 0 else np.conj(cons["G"])
    return IterativeResult(pd.DataFrame(rows), pd.DataFrame(run_rows), G, strengths)
