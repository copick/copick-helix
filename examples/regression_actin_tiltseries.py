"""Regression of the term route against the prototype (mt-polarity/actin, template_polarity.py) on its
zarr-particle-tools segment reconstructions of EMPIAR-10426 (280 actin filaments) and EMPIAR-10521 (182).

    regression_actin_tiltseries.py TAG SEG_ROOT STRAIGHT_DIR PROTO_TPL_PREFIX OUT_DIR WORKERS

e.g. a21 .../actin/data/zpt_10521_bin3 .../actin/data/10521 .../actin/results/diag/tpl21_all760 out/ 8.
Expected (data / decoy), 10426 and 10521: halves 154/220 / 112/220 and 105/135 / 60/135; z >= 3 55 / 13 and 43 / 11;
bundle pairs 46/69 / 33/69 and 147/177 / 93/177; calls agree 0.93 with the prototype's data-built calls.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from copick_helix import bands
from copick_helix.families import get_family


def geometry(straight_dir, names):
    g = {}
    for n in names:
        p = os.path.join(straight_dir, n + ".json")
        if os.path.exists(p):
            m = json.load(open(p))
            c = m["centers_A"] if "centers_A" in m else m["centres_A"]  # the prototype's files spell it centres_A
            g[n] = (str(m["run"]), np.asarray(c)[::10], np.asarray(m["t"])[::10])
    return g


def pairs_summary(geo, table, label):
    calls = dict(zip(table.filament, table.pol))
    out = {}
    for name, sel in (("all", table), ("top50_margin", table[table.margin.abs() >= table.margin.abs().median()]),
                      ("z>=2", table[table.z >= 2]), ("z>=3", table[table.z >= 3])):
        pr = bands.bundle_pairs(geo, calls, list(sel.filament))
        out[name] = f"{sum(p[2] for p in pr)}/{len(pr)}"
    print(f"  bundle pairs same polarity ({label}): {out}")
    return out


def agree(a: pd.Series, b: pd.Series):
    common = a.index.intersection(b.index)
    m = float(np.mean(np.sign(a[common].values) == np.sign(b[common].values)))
    return max(m, 1 - m), len(common)


if __name__ == "__main__":
    tag, seg_root, straight_dir, fork, out_dir, workers = sys.argv[1:7]
    os.makedirs(out_dir, exist_ok=True)
    fam = get_family("actin")
    refs = bands.segment_refs_from_dir(seg_root)
    res = bands.analyze(refs, fam, workers=int(workers))
    res.table.to_csv(f"{out_dir}/{tag}_calls.tsv", sep="\t", index=False)
    res.segments.to_csv(f"{out_dir}/{tag}_segments.tsv", sep="\t", index=False)
    res.decoy_table.to_csv(f"{out_dir}/{tag}_decoy_calls.tsv", sep="\t", index=False)
    s = res.summary
    geo = geometry(straight_dir, list(res.table.filament))
    s["bundle_pairs"] = {"data": pairs_summary(geo, res.table, "data"),
                         "decoy": pairs_summary(geo, res.decoy_table, "decoy")}
    # model-reference calls (diagnostic) and bundle pairs on them
    mcalls = res.segments.groupby("filament").D_model.sum()
    mt = pd.DataFrame({"filament": mcalls.index, "pol": np.sign(mcalls.values), "margin": mcalls.values,
                       "z": res.table.set_index("filament").z.reindex(mcalls.index).values})
    s["bundle_pairs"]["model_reference"] = pairs_summary(geo, mt, "model reference")
    # against the fork
    fk_it = pd.read_csv(f"{fork}_iterative_data_lo+mid.tsv", sep="\t").set_index("filament").pol
    tpl = pd.read_csv(f"{fork}.tsv", sep="\t")
    fk_tpl = tpl[tpl.decoy == False].groupby("filament")["lo+mid"].sum()  # noqa: E712
    mine = res.table.set_index("filament").pol
    s["vs_fork"] = {"data_built_vs_fork_data_built": agree(mine, fk_it), "data_built_vs_fork_model_ref": agree(mine, fk_tpl),
                    "model_ref_vs_fork_model_ref": agree(mcalls, fk_tpl), "fork_data_built_vs_fork_model_ref": agree(fk_it, fk_tpl)}
    conf = res.table[(res.table.z >= 3)].set_index("filament").pol
    s["vs_fork"]["confident_data_built_vs_fork_model_ref"] = agree(conf, fk_tpl)
    json.dump(s, open(f"{out_dir}/{tag}_summary.json", "w"), indent=1, default=str)
    for k in ("rise", "fit_peaks_rel", "terms", "enrichment", "data", "decoy", "model_reference", "data_vs_model_calls",
              "lattice_detected", "polarity_detected", "halves_excess", "seeds", "vs_fork"):
        print(k, s.get(k))
