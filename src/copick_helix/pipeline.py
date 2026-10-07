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
                                half_wedge=half_wedge)
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
                                     r_mask=family.r_mask, half_wedge=90.0)
        model_fits = [invariants.fit_segment(mvol - mvol.mean(), SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                                          next(iter(filaments.values())).step, family.symmetry(**mparams), mcfg, planes)]
    return invariants.assign(fits, cfg, model_fits)


def run_iterative(family: Family, filaments: dict[str, Straightened], params: pd.DataFrame, model=None,
                  random_starts: int = 20, seed_starts: int = 5):
    """Stretch each filament to the family's common reference geometry, then the data-built reference."""
    cfg = iterative.IterativeConfig(rmin=family.cyl_band[0], rmax=family.cyl_band[1], random_starts=random_starts,
                                    seed_starts=seed_starts)
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
    return iterative.assign(gby, r, n, Z, cfg, model_plus_g=gm)


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
