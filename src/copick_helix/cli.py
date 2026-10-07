"""copick CLI commands (registered through the ``copick.process.commands`` entry point).

    copick process helix-polarity -c config.json -i "microtubule:baseline/v1" -t "wbp-filtered@10.005" \\
        --family microtubule --tilt-range -45 63 --work-dir out/ -o "microtubule:helix/1"
"""

from __future__ import annotations

import json
import os

import click
import numpy as np
import pandas as pd
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option

from . import io
from .families import get_family
from .geometry import Straightened, TiltGeometry, straighten as straighten_one
from .pipeline import Result, combine, measure_parameters, run_iterative, run_invariants, save


def _filament_uri(uri: str):
    obj, rest = uri.split(":", 1)
    user, session = rest.split("/", 1)
    return obj, user, session


def _tomo_uri(uri: str):
    tomo_type, vs = uri.rsplit("@", 1)
    return tomo_type, float(vs)


def _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis, min_length, work_dir):
    root = io.open_project(config)
    obj, user, session = _filament_uri(input_uri)
    tomo_type, vs = _tomo_uri(tomogram)
    geom = TiltGeometry(tilt_axis=(0.0, 1.0, 0.0) if tilt_axis == "y" else (1.0, 0.0, 0.0), tilt_range=tuple(tilt_range))
    runs = list(run_names) if run_names else [r.name for r in root.runs]
    out = {}
    sdir = os.path.join(work_dir, "straightened")
    os.makedirs(sdir, exist_ok=True)
    for run in runs:
        fils = io.read_filaments(root, run, obj, user, session)
        fils = [f for f in fils if np.linalg.norm(np.diff(f[1], axis=0), axis=1).sum() >= min_length]
        if not fils:
            continue
        tomo = None
        for iid, pts, _ in fils:
            stem = os.path.join(sdir, f"{run}_f{iid}")
            if os.path.exists(stem + ".json"):
                out[f"{run}_f{iid}"] = Straightened.load(stem)
                continue
            if tomo is None:
                tomo = io.read_tomogram(root, run, vs, tomo_type)
                tomo = (tomo - tomo.mean()) / tomo.std()
            st = straighten_one(pts, tomo, vs, family, geom, normalise=False)
            st.meta.update({"run": run, "instance_id": int(iid), "source": input_uri, "tomogram": tomogram})
            st.save(stem)
            out[f"{run}_f{iid}"] = st
        click.echo(f"{run}: {len(fils)} filaments")
    return root, out


@click.command("helix-straighten", context_settings={"show_default": True})
@add_config_option
@add_run_names_option
@click.option("-i", "--input", "input_uri", required=True, help="Filaments: object:user/session.")
@click.option("-t", "--tomogram", required=True, help="Tomogram: type@voxel_spacing (CTF-corrected).")
@click.option("--family", "family_name", default="microtubule", help="microtubule[_N_S] | actin | intermediate_filament")
@click.option("--tilt-range", nargs=2, type=float, default=(-60.0, 60.0), help="Tilt range (deg).")
@click.option("--tilt-axis", type=click.Choice(["y", "x"]), default="y", help="Tomogram axis of the tilt axis.")
@click.option("--min-length", type=float, default=None, help="Minimum filament length (A); default two segments.")
@click.option("--work-dir", required=True, type=click.Path(), help="Straightened filaments and result tables.")
@add_debug_option
def straighten(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, work_dir, debug):
    """Straighten and recentre traced filaments (cached in WORK_DIR/straightened)."""
    family = get_family(family_name)
    _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                    min_length or 2 * family.segment_length, work_dir)


@click.command("helix-polarity", context_settings={"show_default": True})
@add_config_option
@add_run_names_option
@click.option("-i", "--input", "input_uri", required=True, help="Filaments: object:user/session.")
@click.option("-t", "--tomogram", required=True, help="Tomogram: type@voxel_spacing (CTF-corrected).")
@click.option("--family", "family_name", default="microtubule", help="microtubule[_N_S] | actin | intermediate_filament")
@click.option("--tilt-range", nargs=2, type=float, default=(-60.0, 60.0), help="Tilt range (deg).")
@click.option("--tilt-axis", type=click.Choice(["y", "x"]), default="y", help="Tomogram axis of the tilt axis.")
@click.option("--min-length", type=float, default=None, help="Minimum filament length (A); default two segments.")
@click.option("--methods", default="invariants,iterative", help="Comma-separated: invariants, iterative.")
@click.option("--label/--no-label", default=True, help="Name the groups plus/minus with the family's atomic model.")
@click.option("--work-dir", required=True, type=click.Path(), help="Straightened filaments and result tables.")
@click.option("-o", "--output", "output_uri", default=None, help="Write oriented filaments: object:user/session.")
@add_debug_option
def polarity(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, methods, label,
             work_dir, output_uri, debug):
    """Polarity of traced filaments from their own helical Fourier signal (phase invariants and/or the iterative,
    data-built reference). Writes WORK_DIR/calls.tsv; with -o, the filaments reoriented to the family's polarity
    convention, ``polarity_known`` set for seeds (confident and agreeing between methods)."""
    family = get_family(family_name)
    if not family.polar:
        click.echo(f"{family.name} is apolar: running the null test (expect no polarity groups).")
    root, fils = _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                                 min_length or 2 * family.segment_length, work_dir)
    params = measure_parameters(family, fils)
    model = family.label_model(next(iter(fils.values()))) if (label and family.label_model) else None
    which = {m.strip() for m in methods.split(",")}
    mc, mc_summary = run_invariants(family, fils, params, model) if "invariants" in which else (None, None)
    it = run_iterative(family, fils, params, model) if "iterative" in which else None
    table = combine(mc, it).merge(params.reset_index(), on="filament", how="left")
    res = Result(table, {k: v for k, v in (mc_summary or {}).items() if k != "segments"}, it)
    save(res, work_dir)
    json.dump({"family": family.name, "invariants": res.invariants_summary, "iterative_strength": it.strength if it else None},
              open(os.path.join(work_dir, "summary.json"), "w"), default=str)
    click.echo(table.to_string(index=False))
    if output_uri and family.polar:
        obj, user, session = _filament_uri(output_uri)
        by_run = {}
        call_col = "call" if "call" in table else ("inv_call" if mc is not None else "it_call")
        seed_col = "seed" if "seed" in table else None
        for row in table.itertuples():
            st = fils[row.filament]
            run, iid = st.meta["run"], st.meta["instance_id"]
            by_run.setdefault(run, []).append((iid, getattr(row, call_col), bool(getattr(row, seed_col)) if seed_col else False,
                                               {k: (v.item() if hasattr(v, "item") else v) for k, v in row._asdict().items()
                                                if k not in ("Index",)}))
        for run, items in by_run.items():
            src = {iid: f for iid, _, f in io.read_filaments(root, run, *_filament_uri(input_uri))}
            io.write_oriented(root, run, obj, user, session,
                              [{"source": src[iid], "first_is_plus": call == "plus", "known": known and call in ("plus", "minus"),
                                "info": info} for iid, call, known, info in items], family.name.split("_")[0])
        click.echo(f"wrote {output_uri} for {len(by_run)} runs")
