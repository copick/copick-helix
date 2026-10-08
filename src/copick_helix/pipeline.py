"""End-to-end analysis of a set of straightened filaments for one family: per-filament parameters, phase invariants,
iterative reference, combined call."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import invariants, iterative
from .families import Family
from .fourier import PolarPlanes
from .geometry import Straightened
from .invariants import SegmentGeometry


@dataclass
class Result:
    table: pd.DataFrame  # one row per filament
    invariants_summary: dict | None
    iterative: iterative.IterativeResult | None


def _segment_geoms(st: Straightened, sl: slice) -> SegmentGeometry:
    return SegmentGeometry(np.median(st.beam_local[sl], 0), np.median(st.tilt_local[sl], 0))


def measure_parameters(family: Family, filaments: dict[str, Straightened]) -> pd.DataFrame:
    """Per-filament helical parameters with the family's measurement; unreliable values fall back to the median."""
    rows = []
    for name, st in filaments.items():
        params, quality = family.measure(st)
        rows.append({"filament": name, **params, "quality": quality})
    t = pd.DataFrame(rows).set_index("filament")
    ok = t.quality >= family.measure_min_quality
    for col in [c for c in t.columns if c != "quality"]:
        fallback = float(t.loc[ok, col].median()) if ok.any() else float(family.reference_params[col])
        t.loc[~ok, col] = fallback
    t["parameter_source"] = np.where(ok, "measured", "median of reliable filaments")
    return t


def run_invariants(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame, model=None,
                half_wedge: float = 44.0):
    cfg = invariants.InvariantConfig(terms=family.terms, triples=family.triples, r_out=family.r_out, r_mask=family.r_mask,
                                     half_wedge=half_wedge, plane_tol_frac=family.plane_tol_bins)
    fits = {}
    planes = None
    for name, st in filaments.items():
        if planes is None:
            planes = PolarPlanes(st.step, st.vol.shape[1], cfg.r_band, cfg.r_mask)
        sym = family.symmetry(**{k: params.loc[name, k] for k in family.reference_params})
        fits[name] = [invariants.fit_segment(st.vol[sl] - st.vol[sl].mean(), _segment_geoms(st, sl), st.step, sym, cfg, planes)
                      for _, sl in st.segments(family.segment_length)]
    fits = {k: v for k, v in fits.items() if v}
    model_fits = None
    if model is not None:
        mvol, mparams = model
        mcfg = invariants.InvariantConfig(terms=family.terms, triples=family.triples, r_out=family.r_out,
                                          r_mask=family.r_mask, half_wedge=90.0, plane_tol_frac=family.plane_tol_bins)
        model_fits = [invariants.fit_segment(mvol - mvol.mean(), SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                                          next(iter(filaments.values())).step, family.symmetry(**mparams), mcfg, planes)]
    return invariants.assign(fits, cfg, model_fits)


def run_iterative(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame, model=None,
                  random_starts: int = 20, seed_starts: int = 5, strength_draws: int = 8):
    """Stretch each filament to the family's common reference geometry, then the data-built reference.

    The full polarity search (many random and seeded starts, plus the random-polarity strength baseline) is what makes
    a call trustworthy; for registration alone (e.g. a lattice too weak to call polarity) a few seeded starts and no
    strength baseline suffice: ``random_starts=0, seed_starts=3, strength_draws=0``."""
    cfg = iterative.IterativeConfig(rmin=family.cyl_band[0], rmax=family.cyl_band[1], random_starts=random_starts,
                                    seed_starts=seed_starts, support_z_bins=family.support_z_bins)
    cfg.support = family.support(z_max=cfg.zmax, n_max=cfg.nmax)
    gby = {}
    r = n = Z = None
    for name, st in filaments.items():
        v, beam, tilt = family.to_reference_grid(st, {k: params.loc[name, k] for k in family.reference_params})
        nseg = int(round(family.segment_length / st.step))
        gl = []
        for k in range(v.shape[0] // nseg):
            sl = slice(k * nseg, (k + 1) * nseg)
            r, n, Z, g = iterative.segment_g(v[sl], SegmentGeometry(np.median(beam[sl], 0), np.median(tilt[sl], 0)),
                                             st.step, cfg)
            gl.append(g)
        if gl:
            gby[name] = np.stack(gl)
    gm = None
    if model is not None:
        mvol, _ = model
        _, _, _, gm = iterative.segment_g(mvol, SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                                          next(iter(filaments.values())).step, iterative.IterativeConfig(
                                              rmin=family.cyl_band[0], rmax=family.cyl_band[1], half_wedge=90.0))
    return iterative.assign(gby, r, n, Z, cfg, model_plus_g=gm, strength_draws=strength_draws)


def lattice_gate(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame, rng_seed: int = 0,
                 min_ratio: float = 2.0) -> dict:
    """Is the family's helical lattice detectable at all? Pooled power at each phase-invariant term against the same
    for phase-scrambled copies of the segments (identical power spectra, no helical order).

    Returns {"terms": {(n, m): (data, decoy)}, "detected": bool}: detected when at least one term's enrichment exceeds
    ``min_ratio`` times its decoy value."""
    from .lattice import lattice_enrichment

    rng = np.random.default_rng(rng_seed)
    cfg = iterative.IterativeConfig(rmin=family.cyl_band[0], rmax=family.cyl_band[1])
    pooled = {"data": None, "decoy": None}
    n = Z = None
    for name, st in filaments.items():
        v, beam, tilt = family.to_reference_grid(st, {k: params.loc[name, k] for k in family.reference_params})
        nseg = int(round(family.segment_length / st.step))
        for k in range(v.shape[0] // nseg):
            sl = slice(k * nseg, (k + 1) * nseg)
            seg = v[sl] - v[sl].mean()
            geom = SegmentGeometry(np.median(beam[sl], 0), np.median(tilt[sl], 0))
            Fs = np.fft.rfftn(seg)
            decoy = np.fft.irfftn(np.abs(Fs) * np.exp(1j * np.angle(np.fft.rfftn(rng.normal(size=seg.shape)))),
                                  s=seg.shape).astype(np.float32)
            for key, vol in (("data", seg), ("decoy", decoy)):
                r, n, Z, g = iterative.segment_g(vol, geom, st.step, cfg)
                p = np.tensordot(r, np.abs(g) ** 2, axes=(0, 0))
                pooled[key] = p if pooled[key] is None else pooled[key] + p
    sym = family.symmetry(**family.reference_params)
    d = lattice_enrichment(pooled["data"], n, Z, sym, family.terms)
    q = lattice_enrichment(pooled["decoy"], n, Z, sym, family.terms)
    terms = {k: (d[k], q[k]) for k in d}
    return {"terms": terms, "detected": bool(any(a > min_ratio * max(b, 1.0) for a, b in terms.values()))}


def average_reference(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame,
                      res: iterative.IterativeResult, seeds: set | None = None) -> np.ndarray:
    """Fast average of the registration particles (one per segment), extracted on each filament's reference grid in
    the exported frame (+Z towards the plus end): an initial model consistent with the registration picks.
    ``seeds``: restrict to these filaments."""
    from .registration import Registration, extract, lattice_particles, local_frames

    L = family.segment_length
    screw = family.screw(family.reference_params)
    acc, n = None, 0
    for name, st in filaments.items():
        if seeds is not None and name not in seeds:
            continue
        v, _, _ = family.to_reference_grid(st, {k: params.loc[name, k] for k in family.reference_params})
        grid = local_frames(v.shape[0], st.step, v.shape[1])
        half = (L / 2, family.in_plane_half_width * 0.8, family.in_plane_half_width * 0.8)
        for row in res.segments[res.segments.filament == name].itertuples():
            reg = Registration(row.segment, row.flip, row.roll_deg, row.shift_A, row.score)
            pos, rots, _ = lattice_particles(grid, reg, L, 1.0, screw, family.plus_at_minus_z, centre_only=True)
            sub = extract(v, st.step, np.zeros(3), pos[0], rots[0], half)
            acc = sub if acc is None else acc + sub
            n += 1
    return acc / max(n, 1)


def registration_particles(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame,
                           res: iterative.IterativeResult, every: int | None = None) -> dict:
    """Per filament: (positions (n, 3) A, rotations (n, 3, 3), scores (n,), segment (n,)) in tomogram coordinates.
    One particle per segment (the lattice point nearest its centre), or with ``every`` every n-th lattice point of
    each segment (dense sampling for averaging)."""
    from .registration import Registration, lattice_particles

    screw = family.screw(family.reference_params)
    out = {}
    for name, st in filaments.items():
        p = {k: params.loc[name, k] for k in family.reference_params}
        factor = _axial_factor(family, p)
        P, R, S, G = [], [], [], []
        for row in res.segments[res.segments.filament == name].itertuples():
            reg = Registration(row.segment, row.flip, row.roll_deg, row.shift_A, row.score)
            pos, rots, _ = lattice_particles(st, reg, family.segment_length, factor, screw, family.plus_at_minus_z,
                                             centre_only=every is None, every=every or 1)
            P.append(pos)
            R.append(rots)
            S.append(np.full(len(pos), row.score))
            G.append(np.full(len(pos), row.segment))
        if P:
            out[name] = (np.concatenate(P), np.concatenate(R), np.concatenate(S), np.concatenate(G))
    return out


def _axial_factor(family: Family, params: dict) -> float:
    """original / reference axial scale of a filament (what to_reference_grid divided out)."""
    ref = family.reference_params
    key = "monomer_repeat" if "monomer_repeat" in ref else "rise"
    return float(params[key]) / float(ref[key])


def combine(mc: pd.DataFrame | None, it: iterative.IterativeResult | None, min_projection: float = 0.6) -> pd.DataFrame:
    """One row per filament: both calls, their agreement, and a seed flag (confident in both methods and agreeing)."""
    parts = []
    if mc is not None:
        parts.append(mc.set_index("filament").add_prefix("inv_"))
    if it is not None:
        parts.append(it.calls.set_index("filament").add_prefix("it_"))
    t = pd.concat(parts, axis=1)
    if mc is not None and it is not None:
        t["agree"] = t.inv_call == t.it_call
        t["seed"] = t.agree & t.it_confident & (t.inv_loo_projection.abs() >= min_projection) & t.inv_halves_agree
        t["call"] = np.where(t.agree, t.inv_call, "uncertain")
    return t.reset_index()


def save(result: Result, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    result.table.to_csv(os.path.join(out_dir, "calls.tsv"), sep="\t", index=False)
    if result.iterative is not None:
        result.iterative.runs.to_csv(os.path.join(out_dir, "iterative_runs.tsv"), sep="\t", index=False)
        np.save(os.path.join(out_dir, "iterative_reference_g.npy"), result.iterative.reference)


# --------------------------------------------------------------------------------------------- term route (actin)


def term_segments(filaments: dict[str, Straightened], segment_length: float) -> dict:
    """Segments of straightened tomogram volumes for the term route (centre at index k n + n // 2)."""
    from .bands import SegmentRef

    out = {}
    for name, st in filaments.items():
        refs = []
        for k, sl in st.segments(segment_length):
            refs.append(SegmentRef(name, k, vol=st.vol[sl], step=st.step, beam=np.median(st.beam_local[sl], 0),
                                   tilt=np.median(st.tilt_local[sl], 0)))
        if refs:
            out[name] = refs
    return out


def _term_regs(res, name):
    from .registration import Registration

    for row in res.segments[res.segments.filament == name].itertuples():
        yield (Registration(int(row.segment), bool(row.flip), float(row.roll_deg), float(row.shift_A), float(row.score)),
               (float(row.offset_e1_A), float(row.offset_e2_A)))


def term_registration_particles(family: Family, filaments: dict[str, Straightened], res, segment_length: float,
                                every: int | None = None) -> dict:
    """As ``registration_particles``, for the term route: per filament (positions, rotations, scores, segment)."""
    from .registration import term_particles

    rise, twist = float(res.summary["rise"]), float(res.summary["twist"])
    screw = family.screw({"rise": rise, "twist": twist})
    out = {}
    for name, st in filaments.items():
        n = int(round(segment_length / st.step))
        P, R, S, G = [], [], [], []
        for reg, off in _term_regs(res, name):
            s_centre = (reg.segment * n + n // 2) * st.step
            pos, rots, _ = term_particles(st, s_centre, reg, off, screw, family.plus_at_minus_z, segment_length / 2,
                                          every=every or 1, centre_only=every is None)
            P.append(pos)
            R.append(rots)
            S.append(np.full(len(pos), reg.score))
            G.append(np.full(len(pos), reg.segment))
        if P:
            out[name] = (np.concatenate(P), np.concatenate(R), np.concatenate(S), np.concatenate(G))
    return out


def term_average(family: Family, segments: dict, res, seeds: set | None = None, half_axial: float = 300.0) -> tuple:
    """Fast average of the registered segments themselves (tilt-series reconstructions or tomogram slices) in the
    exported frame (+Z towards the plus end): one particle per segment, its lattice point nearest the centre.
    Returns (map [z, y, x], step)."""
    from .registration import extract, local_frames, term_particles

    rise, twist = float(res.summary["rise"]), float(res.summary["twist"])
    screw = family.screw({"rise": rise, "twist": twist})
    acc, n, step = None, 0, None
    regs = {(f, int(k)): r for f in res.segments.filament.unique() for r in _term_regs(res, f) for k in [r[0].segment]}
    for name, refs in segments.items():
        if seeds is not None and name not in seeds:
            continue
        for ref in refs:
            if (name, ref.index) not in regs:
                continue
            reg, off = regs[(name, ref.index)]
            vol, step, _, _ = ref.load()
            n_s, n_in = vol.shape[0], vol.shape[1]
            grid = local_frames(n_s, step, n_in)
            half_w = 0.9 * (n_in // 2) * step
            pos, rots, _ = term_particles(grid, (n_s // 2) * step, reg, off, screw, family.plus_at_minus_z,
                                          n_s * step / 2, centre_only=True)
            sub = extract(vol - vol.mean(), step, np.zeros(3), pos[0], rots[0],
                          (min(half_axial, 0.45 * n_s * step), half_w, half_w))
            acc = sub if acc is None else acc + sub
            n += 1
    return (acc / max(n, 1) if acc is not None else None), step
