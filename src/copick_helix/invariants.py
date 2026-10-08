"""Phase invariants: polarity from origin-free phase invariants of a wedge-aware Fourier–Bessel fit.

Per segment: the exact Fourier transform on each layer line the family's invariant terms live on, resampled on
(R, Phi); at each R a least-squares fit of every order allowed on that plane (``HelicalSymmetry.plane_orders``,
capped at the orders that can carry signal at R), using only Phi inside the measured region. The missing wedge is a
gap in the samples rather than a mask that mixes orders.

Invariants (independent of rotation about and shift along the axis; complex-conjugated by a polarity flip):
- radial: h_T(R) = c_T(R) conj(c_T(R*_T)) / |c_T(R*_T)| for each term T;
- triple: c_A(R*_A) c_B(R*_B) conj(c_C(R*_C)) for terms with A + B = C (orders and layer-line indices add).
They are averaged over a filament's segments without alignment. Filament features are their imaginary parts; the
leading eigenvector of the filament x filament product splits the filaments into two polarity groups.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fourier import PolarPlanes
from .helix import HelicalSymmetry, Term, n_cap


@dataclass
class SegmentGeometry:
    beam: np.ndarray  # beam direction in local (e1, e2, t) components
    tilt_axis: np.ndarray  # tilt axis in local (e1, e2, t) components


@dataclass
class InvariantConfig:
    terms: list[Term]
    triples: list[tuple[Term, Term, Term]]  # (A, B, C) with A + B = C
    r_out: float  # outer radius for the order cap (A)
    r_band: tuple[float, float] = (1 / 300.0, 1 / 24.0)
    r_mask: float = 165.0  # soft in-plane mask radius (A)
    half_wedge: float = 44.0
    n_max: int = 60
    ridge: float = 1e-3
    plane_tol_frac: float = 0.5  # orders on the same plane: |dZ| <= plane_tol_frac / segment length


def fit_plane(
    F: np.ndarray,
    M: np.ndarray,
    R: np.ndarray,
    orders: list[int],
    r_out: float,
    phi: np.ndarray,
    ridge: float,
) -> dict[int, np.ndarray]:
    """Least squares per R of F(R, Phi_measured) ~ sum_n c_n(R) exp(i n Phi), orders capped by n_cap(R)."""
    coefs: dict[int, np.ndarray] = {}
    for i, Ri in enumerate(R):
        use = [n for n in orders if abs(n) <= n_cap(Ri, r_out)]
        m = M[i]
        if not use or m.sum() < 2 * len(use) + 4:
            continue
        y = F[i, m]
        E = np.exp(1j * np.outer(phi[m], use))
        A = E.conj().T @ E
        c = np.linalg.solve(A + ridge * np.trace(A).real / len(use) * np.eye(len(use)), E.conj().T @ y)
        for n, v in zip(use, c):
            coefs.setdefault(n, np.full(len(R), np.nan + 0j))[i] = v
    return coefs


@dataclass
class SegmentFit:
    coefs: dict[Term, np.ndarray]  # c_T(R) for the configured terms
    coverage: dict[float, float] = field(default_factory=dict)  # measured fraction per plane Z


def fit_segment(
    seg: np.ndarray,
    geom: SegmentGeometry,
    step: float,
    sym: HelicalSymmetry,
    cfg: InvariantConfig,
    planes: PolarPlanes | None = None,
) -> SegmentFit:
    """Fit the configured terms of one segment (seg[z, y, x], protein positive, axis at the in-plane center)."""
    if planes is None:
        planes = PolarPlanes(step, seg.shape[1], cfg.r_band, cfg.r_mask)
    tol = cfg.plane_tol_frac / (seg.shape[0] * step)
    by_plane: dict[float, list[Term]] = {}
    for t in cfg.terms:
        z = round(sym.Z(t), 9)
        by_plane.setdefault(z, []).append(t)
    out, cov = {}, {}
    for z, ts in by_plane.items():
        F = planes.plane(seg, z)
        M = planes.measured(geom.beam, geom.tilt_axis, z, cfg.half_wedge)
        orders = sym.plane_orders(ts[0], cfg.n_max, tol)
        co = fit_plane(F, M, planes.R, orders, cfg.r_out, planes.phi, cfg.ridge)
        for t in ts:
            out[t] = co.get(t.n, np.full(len(planes.R), np.nan + 0j))
        cov[z] = float(M.mean())
    return SegmentFit(out, cov)


def _nanmean(a, axis=0):
    """nanmean without the empty-slice warning (radii no segment could fit stay nan)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def pooled_rstar(fits: list[SegmentFit], terms: list[Term]) -> dict[Term, int]:
    """R index of each term's pooled amplitude peak."""
    return {t: int(np.nanargmax(_nanmean([np.abs(f.coefs[t]) ** 2 for f in fits]))) for t in terms}


def pooled_weights(fits: list[SegmentFit], terms: list[Term]) -> dict[Term, np.ndarray]:
    w = {}
    for t in terms:
        amp = np.sqrt(_nanmean([np.abs(f.coefs[t]) ** 2 for f in fits]))
        w[t] = np.nan_to_num(amp / np.nanmax(amp))
    return w


def invariants(fits: list[SegmentFit], cfg: InvariantConfig, rstar: dict[Term, int]) -> dict:
    """Filament-averaged invariants (complex) from a filament's segment fits."""
    out = {}
    for t in cfg.terms:
        hs = []
        for f in fits:
            c = f.coefs[t]
            ref = c[rstar[t]]
            if np.isfinite(ref) and abs(ref) > 0:
                hs.append(c * np.conj(ref) / abs(ref))
        out[("radial", t)] = _nanmean(np.array(hs)) if hs else None
    for a, b, c in cfg.triples:
        vals = []
        for f in fits:
            v = f.coefs[a][rstar[a]] * f.coefs[b][rstar[b]] * np.conj(f.coefs[c][rstar[c]])
            if np.isfinite(v) and abs(v) > 0:
                vals.append(v / abs(v))
        out[("triple", a, b, c)] = np.array([np.mean(vals)]) if vals else None
    return out


def feature_vector(inv: dict, keys: list, weights: dict[Term, np.ndarray], n_r: int) -> np.ndarray:
    parts = []
    for k in keys:
        v = inv.get(k)
        size = n_r if k[0] == "radial" else 1
        if v is None:
            parts.append(np.zeros(size))
            continue
        w = weights[k[1]] if k[0] == "radial" else 1.0
        parts.append(np.nan_to_num(np.imag(v)) * w)
    return np.concatenate(parts)


def assign(
    fits_by_filament: dict[str, list[SegmentFit]],
    cfg: InvariantConfig,
    model_fits: list[SegmentFit] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Relative polarity of all filaments (and an absolute label if model fits are given).

    Returns a per-filament table (group, leave-one-out projection, per-segment and odd/even agreement, call) and a
    summary (eigenvalues, model projection)."""
    names = list(fits_by_filament)
    allfits = [f for n in names for f in fits_by_filament[n]]
    rstar = pooled_rstar(allfits, cfg.terms)
    weights = pooled_weights(allfits, cfg.terms)
    n_r = len(next(iter(allfits)).coefs[cfg.terms[0]])
    keys = sorted({k for n in names for k in invariants(fits_by_filament[n][:1], cfg, rstar)}, key=str)
    X = np.array([feature_vector(invariants(fits_by_filament[n], cfg, rstar), keys, weights, n_r) for n in names])
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    w, V = np.linalg.eigh(X @ X.T)
    v = V[:, -1]
    sign_model, cos_model = None, None
    u_all = X.T @ np.sign(v)
    if model_fits:
        xm = feature_vector(invariants(model_fits, cfg, rstar), keys, weights, n_r)
        sign_model = float(np.sign(xm @ u_all))
        cos_model = float(xm @ u_all / max(np.linalg.norm(xm) * np.linalg.norm(u_all), 1e-12))
    rows, seg_rows = [], []
    for i, n in enumerate(names):
        u = u_all - X[i] * np.sign(v[i])
        un = u / max(np.linalg.norm(u), 1e-12)
        proj = float(X[i] @ un)
        grp = int(np.sign(proj)) if proj != 0 else 0
        seg_signs = []
        for si, f in enumerate(fits_by_filament[n]):
            xs = feature_vector(invariants([f], cfg, rstar), keys, weights, n_r)
            ns = np.linalg.norm(xs)
            if ns > 0:
                p = float(xs @ un / ns)
                seg_signs.append(np.sign(p))
                seg_rows.append({"filament": n, "segment": si, "projection": p})
        seg_signs = np.array(seg_signs)
        halves = []
        for h in (0, 1):
            sub = fits_by_filament[n][h::2]
            if sub:
                xh = feature_vector(invariants(sub, cfg, rstar), keys, weights, n_r)
                halves.append(np.sign(xh @ un) if np.linalg.norm(xh) > 0 else 0.0)
        call = None if sign_model is None else ("plus" if grp * sign_model > 0 else "minus")
        rows.append(
            {
                "filament": n,
                "group": grp,
                "loo_projection": proj,
                "eigvec": float(v[i]),
                "segments": len(fits_by_filament[n]),
                "seg_agree": float((seg_signs == grp).mean()) if len(seg_signs) else np.nan,
                "halves_agree": bool(len(halves) == 2 and halves[0] == halves[1] == grp),
                "call": call,
            },
        )
    summary = {
        "eigenvalues": w[::-1][:5].tolist(),
        "eigen_gap": float(w[-1] / max(w[-2], 1e-12)),
        "model_sign": sign_model,
        "model_cos": cos_model,
        "segments": pd.DataFrame(seg_rows),
        "rstar_R": {str(t): float(1.0) for t in rstar},
    }
    return pd.DataFrame(rows), summary
