"""copick CLI commands (registered through the ``copick.process.commands`` entry point).

    copick process helix-polarity -c config.json -i "microtubule:baseline/v1" -t "wbp-filtered@10.005" \\
        --family microtubule --tilt-range -45 63 --work-dir out/ -o "microtubule:helix/1"
"""

from __future__ import annotations

import json
import os

import click
import mrcfile
import numpy as np
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option

from . import __version__, io
from .families import get_family
from .geometry import Straightened, TiltGeometry, straighten as straighten_one
from .iterative import reference_volume  # noqa: F401  (re-exported for scripting)
from .pipeline import (Result, average_reference, combine, lattice_gate, measure_parameters, registration_particles,
                       run_invariants, run_iterative, save)


def _filament_uri(uri: str):
    obj, rest = uri.split(":", 1)
    user, session = rest.split("/", 1)
    return obj, user, session


def _tomo_uri(uri: str):
    tomo_type, vs = uri.rsplit("@", 1)
    return tomo_type, float(vs)


def _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis, min_length, work_dir,
                    keep_volumes):
    """Straighten every filament of the input set; cached in WORK_DIR/straightened only with ``keep_volumes``."""
    root = io.open_project(config)
    obj, user, session = _filament_uri(input_uri)
    tomo_type, vs = _tomo_uri(tomogram)
    geom = TiltGeometry(tilt_axis=(0.0, 1.0, 0.0) if tilt_axis == "y" else (1.0, 0.0, 0.0), tilt_range=tuple(tilt_range))
    runs = list(run_names) if run_names else [r.name for r in root.runs]
    out = {}
    sdir = os.path.join(work_dir, "straightened")
    if keep_volumes:
        os.makedirs(sdir, exist_ok=True)
    for run in runs:
        fils = io.read_filaments(root, run, obj, user, session)
        fils = [f for f in fils if np.linalg.norm(np.diff(f[1], axis=0), axis=1).sum() >= min_length]
        if not fils:
            continue
        tomo = None
        for iid, pts, _ in fils:
            stem = os.path.join(sdir, f"{run}_f{iid}")
            if keep_volumes and os.path.exists(stem + ".json"):
                out[f"{run}_f{iid}"] = Straightened.load(stem)
                continue
            if tomo is None:
                tomo = io.read_tomogram(root, run, vs, tomo_type)
                tomo = (tomo - tomo.mean()) / tomo.std()
            st = straighten_one(pts, tomo, vs, family, geom, normalise=False, window=family.recentre_window,
                                lim_A=family.recentre_max_shift)
            st.meta.update({"run": run, "instance_id": int(iid), "source": input_uri, "tomogram": tomogram})
            if keep_volumes:
                st.save(stem)
            out[f"{run}_f{iid}"] = st
        click.echo(f"{run}: {len(fils)} filaments")
    return root, out


COMMON = [
    add_config_option,
    add_run_names_option,
    click.option("-i", "--input", "input_uri", required=True, help="Filaments: object:user/session."),
    click.option("-t", "--tomogram", required=True, help="Tomogram: type@voxel_spacing (CTF-corrected)."),
    click.option("--family", "family_name", default="microtubule",
                 help="microtubule[_N_S] | intermediate_filament (actin in progress)."),
    click.option("--tilt-range", nargs=2, type=float, default=(-60.0, 60.0), help="Tilt range (deg)."),
    click.option("--tilt-axis", type=click.Choice(["y", "x"]), default="y", help="Tomogram axis of the tilt axis."),
    click.option("--min-length", type=float, default=None, help="Minimum filament length (A); default two segments."),
    click.option("--work-dir", required=True, type=click.Path(), help="Result tables, reference map, optional cache."),
    click.option("--keep-volumes", is_flag=True, help="Keep the straightened volumes in WORK_DIR/straightened."),
]


def _common(f):
    for opt in reversed(COMMON):
        f = opt(f)
    return f


@click.command("helix-straighten", context_settings={"show_default": True})
@_common
@add_debug_option
def straighten(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, work_dir,
               keep_volumes, debug):
    """Straighten and recentre traced filaments into WORK_DIR/straightened."""
    family = get_family(family_name)
    _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                    min_length or 2 * family.segment_length, work_dir, keep_volumes=True)


@click.command("helix-polarity", context_settings={"show_default": True})
@_common
@click.option("--methods", default="invariants,iterative", help="Comma-separated: invariants, iterative.")
@click.option("--label/--no-label", default=True, help="Name the groups plus/minus with the family's atomic model.")
@click.option("-o", "--output", "output_uri", default=None,
              help="object:user/session for the results: filaments (recentred centre lines, Catmull-Rom, ordered "
                   "minus -> plus when known, analysis in metadata) and picks (one registration per segment).")
@click.option("--picks", "picks_uri", default=None, help="Also write dense lattice-registered picks: object:user/session.")
@click.option("--every", type=int, default=1, help="With --picks: every n-th lattice point (MT dimer, helical subunit).")
@click.option("--seeds-only/--all-filaments", default=True, help="Write registration and dense picks for seeds only.")
@add_debug_option
def polarity(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, work_dir,
             keep_volumes, methods, label, output_uri, picks_uri, every, seeds_only, debug):
    """Polarity, lattice registration and an initial model for traced filaments, from their own helical Fourier
    signal (phase invariants and the iterative, data-built reference). Writes WORK_DIR/calls.tsv, summary.json and
    reference_<family>.mrc (the fast average of the registration particles, +Z towards the plus end)."""
    family = get_family(family_name)
    root, fils = _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                                 min_length or 2 * family.segment_length, work_dir, keep_volumes)
    params = measure_parameters(family, fils)
    gate = lattice_gate(family, fils, params)
    click.echo("lattice gate (pooled enrichment, data / decoy): " +
               ", ".join(f"{k}: {a:.1f}/{b:.1f}" for k, (a, b) in gate["terms"].items()) +
               ("" if gate["detected"] else "  -> lattice NOT detected: no filament gets polarity_known"))
    model = family.label_model(next(iter(fils.values()))) if (label and family.label_model) else None
    which = {m.strip() for m in methods.split(",")}
    mc, mc_summary = run_invariants(family, fils, params, model) if "invariants" in which else (None, None)
    if "iterative" in which and not gate["detected"]:
        # no polarity can be called, so skip the polarity search: a few seeded starts give the registration
        click.echo("lattice not detected: iterative reference run for registration only (3 seeded starts)")
        it = run_iterative(family, fils, params, model, random_starts=0, seed_starts=3, strength_draws=0)
    else:
        it = run_iterative(family, fils, params, model) if "iterative" in which else None
    table = combine(mc, it).merge(params.reset_index(), on="filament", how="left")
    if "seed" in table:
        table["seed"] = table["seed"] & gate["detected"]
    if not gate["detected"] and "call" in table:
        table["call"] = "uncertain"  # per-method calls stay in inv_call / it_call for inspection
    res = Result(table, {k: v for k, v in (mc_summary or {}).items() if k != "segments"}, it)
    save(res, work_dir)
    ref_path = None
    if it is not None:
        seeds = set(table.filament[table.seed]) if "seed" in table and table.seed.any() else None
        ref = average_reference(family, fils, params, it, seeds)
        ref_path = os.path.abspath(os.path.join(work_dir, f"reference_{family.name}.mrc"))
        with mrcfile.new(ref_path, overwrite=True) as m:
            m.set_data(ref.astype(np.float32))
            m.voxel_size = next(iter(fils.values())).step
    json.dump({"family": family.name, "version": __version__, "invariants": res.invariants_summary,
               "iterative_strength": it.strength if it else None, "reference_map": ref_path,
               "lattice_gate": {"detected": gate["detected"], "terms": {str(k): v for k, v in gate["terms"].items()}}},
              open(os.path.join(work_dir, "summary.json"), "w"), default=str)
    click.echo(table.to_string(index=False))
    if not (output_uri or picks_uri) or it is None:
        return

    call_col = "call" if "call" in table else ("inv_call" if mc is not None else "it_call")
    rows = table.set_index("filament")
    reg = registration_particles(family, fils, params, it)
    dense = registration_particles(family, fils, params, it, every=every) if picks_uri else {}
    by_run: dict = {}
    for name, st in fils.items():
        by_run.setdefault(st.meta["run"], []).append(name)
    for run, names in by_run.items():
        if output_uri:
            obj, user, session = _filament_uri(output_uri)
            items = []
            for name in names:
                row = rows.loc[name]
                call = row[call_col]
                seed = bool(row["seed"]) if "seed" in rows else False
                info = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if not str(k).startswith("_")}
                items.append({
                    "instance_id": fils[name].meta["instance_id"], "centres": fils[name].centres,
                    "reverse": call == "plus",  # 'plus': plus end at the first trace point -> reverse to minus -> plus
                    "known": seed and call in ("plus", "minus"),
                    "metadata": {"copick_helix": {"version": __version__, "family": family.name, "call": call,
                                                  "seed": seed, "lattice_detected": gate["detected"],
                                                  "polarity_convention": "minus_to_plus" if call in ("plus", "minus") else None,
                                                  "registration_picks": output_uri, "reference_map": ref_path,
                                                  "analysis": info}}})
            io.write_centrelines(root, run, obj, user, session, items)
            sel = [n for n in names if n in reg and (not seeds_only or bool(rows.loc[n].get("seed", False)))]
            if sel:
                io.write_particles(root, run, obj, user, session, np.concatenate([reg[n][0] for n in sel]),
                                   np.concatenate([reg[n][1] for n in sel]),
                                   np.concatenate([np.full(len(reg[n][0]), fils[n].meta["instance_id"]) for n in sel]),
                                   np.concatenate([reg[n][2] for n in sel]))
        if picks_uri:
            obj, user, session = _filament_uri(picks_uri)
            sel = [n for n in names if n in dense and (not seeds_only or bool(rows.loc[n].get("seed", False)))]
            if sel:
                io.write_particles(root, run, obj, user, session, np.concatenate([dense[n][0] for n in sel]),
                                   np.concatenate([dense[n][1] for n in sel]),
                                   np.concatenate([np.full(len(dense[n][0]), fils[n].meta["instance_id"]) for n in sel]),
                                   np.concatenate([dense[n][2] for n in sel]))
    click.echo(f"wrote {output_uri or ''} {picks_uri or ''} for {len(by_run)} runs")


@click.command("helix-picks", context_settings={"show_default": True})
@add_config_option
@add_run_names_option
@click.option("-i", "--input", "input_uri", required=True,
              help="helix-polarity results: object:user/session (its filaments and registration picks).")
@click.option("-o", "--output", "output_uri", required=True, help="Dense picks: object:user/session.")
@click.option("--every", type=int, default=1, help="Every n-th lattice point (MT dimer, helical subunit).")
@click.option("--seeds-only/--all-filaments", default=True, help="Only filaments with polarity_known.")
@add_debug_option
def picks(config, run_names, input_uri, output_uri, every, seeds_only, debug):
    """Lattice-registered, oriented picks for averaging, sampled from stored helix-polarity results (no
    recomputation): every n-th lattice point along each filament, oriented from its segment registrations (+Z
    towards the plus end)."""
    from .sampling import dense_from_registration

    root = io.open_project(config)
    obj, user, session = _filament_uri(input_uri)
    oobj, ouser, osession = _filament_uri(output_uri)
    runs = list(run_names) if run_names else [r.name for r in root.runs]
    total = 0
    for run_name in runs:
        run = root.get_run(run_name)
        fsets = run.get_filaments(object_name=obj, user_id=user, session_id=session)
        psets = run.get_picks(object_name=obj, user_id=user, session_id=session)
        if not fsets or not psets:
            continue
        pos, T = psets[0].numpy()
        ids = np.array([p.instance_id for p in psets[0].points])
        P, R, I = [], [], []
        for f in fsets[0].filaments:
            if seeds_only and not f.polarity_known:
                continue
            sel = ids == f.instance_id
            if not sel.any():
                continue
            p, r = dense_from_registration(f.points, f.metadata, pos[sel], T[sel][:, :3, :3], every=every)
            if len(p):
                P.append(p)
                R.append(r)
                I.append(np.full(len(p), f.instance_id))
        if P:
            io.write_particles(root, run_name, oobj, ouser, osession, np.concatenate(P), np.concatenate(R),
                               np.concatenate(I), np.ones(sum(len(x) for x in P)))
            total += sum(len(x) for x in P)
    click.echo(f"wrote {total} picks to {output_uri}")
