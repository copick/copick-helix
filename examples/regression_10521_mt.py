"""Regression against the prototype scripts on the 54 straightened 10521 microtubules (wbp-filtered, 10 A).

    regression_10521_mt.py [--iterative]

The phase invariants must reproduce results/method_c_wbpf/polarity.tsv; the iterative reference must reproduce
results/polar_iter/calls.tsv (up to the documented uncertain filaments)."""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from copick_helix import invariants, iterative
from copick_helix.families import microtubule
from copick_helix.fourier import PolarPlanes, lowpass
from copick_helix.invariants import SegmentGeometry
from copick_helix.models import MicrotubuleLattice, on_grid

PROTO = "/hpc/projects/group.czii/utz.ermel/mt-polarity"
DATA = f"{PROTO}/data/10521/wbp-filtered-aretomo3V2.3.0-ctfdeconv"
fam = microtubule()
rp = pd.read_csv(f"{PROTO}/data/10521/axial_repeat_all_wbpf.tsv", sep="\t")
good = rp.peak_snr >= 3
med = float(rp.loc[good, "monomer_repeat_A"].median())
rep = {f"{r.run}_f{r.idx}": float(r.monomer_repeat_A) for r in rp[good].itertuples()}

t0 = time.time()
cfg = invariants.InvariantConfig(terms=fam.terms, triples=fam.triples, r_out=fam.r_out, r_mask=fam.r_mask)
planes = PolarPlanes(5.0, 81, cfg.r_band, cfg.r_mask)
fits, vols = {}, {}
for p in sorted(glob.glob(f"{DATA}/*.npy")):
    name = os.path.basename(p)[:-4]
    meta = json.load(open(p[:-4] + ".json"))
    vol = np.nan_to_num(np.load(p).astype(np.float32)) * float(meta["protein_sign"])
    beam = np.array(meta["beam_in_local_e1_e2_t"]["per_s"])
    tilt = np.array(meta["tilt_axis_in_local_e1_e2_t"]["per_s"])
    sym = fam.symmetry(monomer_repeat=rep.get(name, med))
    nseg = int(round(fam.segment_length / 5.0))
    segs = []
    for k in range(vol.shape[0] // nseg):
        sl = slice(k * nseg, (k + 1) * nseg)
        seg = vol[sl] - vol[sl].mean()
        geom = SegmentGeometry(np.median(beam[sl], 0), np.median(tilt[sl], 0))
        segs.append(invariants.fit_segment(seg, geom, 5.0, sym, cfg, planes))
    fits[name] = segs
    vols[name] = (vol, beam, tilt, rep.get(name, med))
lat = MicrotubuleLattice(f"{PROTO}/models/6DPV.cif")
xyz, w = lat.atoms(13, 3, fam.segment_length)
mv = on_grid(xyz, w, 5.0, 81, fam.segment_length)
model_fit = [invariants.fit_segment(mv - mv.mean(), SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0.25, 0.97])),
                                 5.0, fam.symmetry(monomer_repeat=lat.a),
                                 invariants.InvariantConfig(terms=fam.terms, triples=fam.triples, r_out=fam.r_out,
                                                       r_mask=fam.r_mask, half_wedge=90.0), planes)]
table, summ = invariants.assign(fits, cfg, model_fit)
proto = pd.read_csv(f"{PROTO}/results/method_c_wbpf/polarity.tsv", sep="\t").set_index("filament")
t = table.set_index("filament").join(proto[["call", "loo_projection"]], rsuffix="_proto")
print(f"phase invariants: {time.time() - t0:.0f} s; eigen gap {summ['eigen_gap']:.2f}; model cos {summ['model_cos']:+.2f}")
print(f"  calls equal to prototype: {(t.call == t.call_proto).sum()}/{len(t)}; |projection| correlation "
      f"{np.corrcoef(t.loo_projection.abs(), t.loo_projection_proto.abs())[0, 1]:.3f}; halves agree {t.halves_agree.sum()}/{len(t)}")
if (t.call != t.call_proto).any():
    print(t[t.call != t.call_proto][["call", "call_proto", "loo_projection", "loo_projection_proto"]])

if "--iterative" in sys.argv:
    from scipy.ndimage import map_coordinates

    t1 = time.time()
    a_ref = fam.reference_params["monomer_repeat"]
    for label, support in (("prototype rule", None), ("family support", fam.support(z_max=1 / 20.0, n_max=56))):
        icfg = iterative.IterativeConfig(support=support)
        gby = {}
        for name, (vol, beam, tilt, a) in vols.items():
            pos = np.arange(0, vol.shape[0] - 1, a / a_ref)  # stretch to the common repeat
            grid = np.meshgrid(pos, np.arange(vol.shape[1]), np.arange(vol.shape[2]), indexing="ij")
            v = map_coordinates(vol, grid, order=1).astype(np.float32)
            idx = np.clip(np.round(pos).astype(int), 0, len(beam) - 1)
            b, tl = beam[idx], tilt[idx]
            gl = []
            for k in range(v.shape[0] // 250):
                sl = slice(k * 250, (k + 1) * 250)
                r, n, Z, g = iterative.segment_g(v[sl], SegmentGeometry(np.median(b[sl], 0), np.median(tl[sl], 0)), 5.0, icfg)
                gl.append(g)
            gby[name] = np.stack(gl)
        _, _, _, gm = iterative.segment_g(mv, SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])), 5.0,
                                          iterative.IterativeConfig(half_wedge=90.0))
        res = iterative.assign(gby, r, n, Z, icfg, model_plus_g=gm)
        pi = pd.read_csv(f"{PROTO}/results/polar_iter/calls.tsv", sep="\t").set_index("filament")
        c = res.calls.set_index("filament").join(pi[["call"]], rsuffix="_proto")
        conv = (res.runs[res.runs.start.str.startswith(("random", "seed"))].agree_with_consensus >= 0.95).mean()
        print(f"iterative ({label}): {time.time() - t1:.0f} s; calls equal to prototype {(c.call == c.call_proto).sum()}/{len(c)}; "
              f"confident {c.confident.sum()}; starts reaching consensus {conv:.0%}; strength {res.strength}")
        if (c.call != c.call_proto).any():
            print(c[c.call != c.call_proto][["call", "call_proto", "bootstrap", "start_stability"]])
