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


def _straighten_run(args):
    """Straighten the filaments of one run (a worker: opens the project and the run's tomogram itself)."""
    config, run, input_uri, tomogram, family_name, geom, min_length, sdir, keep_volumes = args
    root = io.open_project(config)
    family = get_family(family_name)
    obj, user, session = _filament_uri(input_uri)
    tomo_type, vs = _tomo_uri(tomogram)
    fils = io.read_filaments(root, run, obj, user, session)
    fils = [f for f in fils if np.linalg.norm(np.diff(f[1], axis=0), axis=1).sum() >= min_length]
    out = {}
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
    return run, len(fils), out


def _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis, min_length, work_dir,
                    keep_volumes, workers: int = 1):
    """Straighten every filament of the input set, one run per worker; cached in WORK_DIR/straightened only with
    ``keep_volumes``."""
    root = io.open_project(config)
    geom = TiltGeometry(tilt_axis=(0.0, 1.0, 0.0) if tilt_axis == "y" else (1.0, 0.0, 0.0), tilt_range=tuple(tilt_range))
    runs = list(run_names) if run_names else [r.name for r in root.runs]
    sdir = os.path.join(work_dir, "straightened")
    if keep_volumes:
        os.makedirs(sdir, exist_ok=True)
    jobs = [(config, run, input_uri, tomogram, family.name, geom, min_length, sdir, keep_volumes) for run in runs]
    out = {}
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            results = ex.map(_straighten_run, jobs)
            for run, n, fils in results:
                out.update(fils)
                if n:
                    click.echo(f"{run}: {n} filaments")
    else:
        for job in jobs:
            run, n, fils = _straighten_run(job)
            out.update(fils)
            if n:
                click.echo(f"{run}: {n} filaments")
    return root, out


COMMON = [
    add_config_option,
    add_run_names_option,
    click.option("-i", "--input", "input_uri", required=True, help="Filaments: object:user/session."),
    click.option("-t", "--tomogram", required=True, help="Tomogram: type@voxel_spacing (CTF-corrected)."),
    click.option("--family", "family_name", default="microtubule",
                 help="microtubule[_N_S] | actin | intermediate_filament."),
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


def _axial_step(family) -> float:
    """The family's subunit step along the axis (A): monomer repeat (MT) or helical rise."""
    p = family.reference_params
    return float(p.get("rise", p.get("monomer_repeat", 40.0)))


def _write_recentred(root, fils, family, output_uri, picks_uri, spacing):
    """Recentred centre lines as filaments (trace order, polarity unknown) and, with ``picks_uri``, picks every
    ``spacing`` A along them oriented with the rotation-minimising frame (+Z along the trace direction; rotation about
    the axis arbitrary but continuous)."""
    by_run: dict = {}
    for name, st in fils.items():
        by_run.setdefault(st.meta["run"], []).append(name)
    n_picks = 0
    for run, names in by_run.items():
        if output_uri:
            obj, user, session = _filament_uri(output_uri)
            items = [{"instance_id": fils[n].meta["instance_id"], "centres": fils[n].centres, "reverse": False,
                      "known": False,
                      "metadata": {"copick_helix": {"version": __version__, "family": family.name, "recentred": True,
                                                    "lattice_analysis": "not attempted", "polarity": "unknown",
                                                    "source": fils[n].meta.get("source"),
                                                    "tomogram": fils[n].meta.get("tomogram"),
                                                    "recentring": fils[n].recentring}}} for n in names]
            io.write_centrelines(root, run, obj, user, session, items)
        if picks_uri:
            P, R, ids = [], [], []
            for n in names:
                st = fils[n]
                k = np.unique(np.round(np.arange(0.0, st.length + 1e-6, spacing) / st.step).astype(int))
                k = k[k < len(st.centres)]
                P.append(st.centres[k])
                R.append(np.stack([st.e1[k], st.e2[k], st.t[k]], axis=2))  # columns e1, e2, t
                ids.append(np.full(len(k), st.meta["instance_id"]))
            if P:
                obj, user, session = _filament_uri(picks_uri)
                io.write_particles(root, run, obj, user, session, np.concatenate(P), np.concatenate(R),
                                   np.concatenate(ids), np.ones(sum(len(p) for p in P)))
                n_picks += sum(len(p) for p in P)
    click.echo(f"recentred {len(fils)} filaments in {len(by_run)} runs"
               + (f" -> {output_uri}" if output_uri else "") + (f"; {n_picks} picks -> {picks_uri}" if picks_uri else ""))


@click.command("helix-recentre", context_settings={"show_default": True})
@_common
@click.option("-o", "--output", "output_uri", required=True,
              help="object:user/session for the recentred centre lines (Catmull-Rom filaments, polarity unknown).")
@click.option("--picks", "picks_uri", default=None,
              help="Also write picks along the recentred lines: object:user/session.")
@click.option("--spacing", type=float, default=None, help="With --picks: spacing along the line (A); default two "
                                                           "subunit steps of the family.")
@click.option("--workers", type=int, default=4, help="Worker processes (one run each).")
@add_debug_option
def recentre(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, work_dir,
             keep_volumes, output_uri, picks_uri, spacing, workers, debug):
    """Recentre traced filaments on their density (the family's cross-section profile) and write the recentred centre
    lines, without any lattice or polarity analysis. Picks, if asked for, are evenly spaced and oriented along the
    line only (+Z along the trace direction, which is not a polarity)."""
    family = get_family(family_name)
    root, fils = _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                                 min_length or family.segment_length, work_dir, keep_volumes, workers=workers)
    _write_recentred(root, fils, family, output_uri, picks_uri, spacing or 2 * _axial_step(family))


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
@click.option("--source", type=click.Choice(["tomogram", "tiltseries"]), default="tomogram",
              help="Term-route families (actin): segments from the straightened tomograms or reconstructed from the "
                   "tilt series (zarr-particle-tools).")
@click.option("--tomograms-star", default=None, type=click.Path(exists=True),
              help="--source tiltseries: RELION tomograms.star of the tilt series.")
@click.option("--tiltseries-dir", default=None, type=click.Path(exists=True),
              help="--source tiltseries: directory the tilt-series star paths are relative to (default: the project "
                   "directory three levels above --tomograms-star, as for <project>/Import/jobNNN/tomograms.star).")
@click.option("--bin", "binning", type=int, default=3, help="--source tiltseries: binning of the reconstructions.")
@click.option("--size-unit", "size_unit_A", type=float, default=None,
              help="--source tiltseries: A per rlnTomoSize unit (default: from the copick tomogram extent).")
@click.option("--snr", type=float, default=0.1, help="--source tiltseries: Wiener CTF correction offset 1/SNR.")
@click.option("--workers", type=int, default=4, help="Worker processes (term route).")
@click.option("--route", type=click.Choice(["auto", "terms", "cylindrical"]), default="auto",
              help="auto: the family's own (actin: terms; microtubule, IF: cylindrical). terms: band-limited helical "
                   "terms (copick_helix.bands) for any family with an atomic model.")
@add_debug_option
def polarity(config, run_names, input_uri, tomogram, family_name, tilt_range, tilt_axis, min_length, work_dir,
             keep_volumes, methods, label, output_uri, picks_uri, every, seeds_only, source, tomograms_star,
             tiltseries_dir, binning, size_unit_A, snr, workers, route, debug):
    """Polarity, lattice registration and an initial model for traced filaments, from their own helical Fourier
    signal (phase invariants and the iterative, data-built reference). Writes WORK_DIR/calls.tsv, summary.json and
    reference_<family>.mrc (the fast average of the registration particles, +Z towards the plus end).

    Actin uses the term route (band-limited helical terms, a data-built reference and decoys; see copick_helix.bands)
    and needs --source tiltseries on 10 A data; it also writes segments.tsv (per-segment registration) and
    decoy_calls.tsv."""
    family = get_family(family_name)
    if not family.lattice_analysis and route == "auto":
        click.echo(f"{family.name}: no lattice or polarity analysis for this family "
                   f"({family.notes.get('lattice_analysis', 'off')}); recentring only, as helix-recentre")
        root, fils = _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                                     min_length or family.segment_length, work_dir, keep_volumes, workers=workers)
        _write_recentred(root, fils, family, output_uri, picks_uri, every * _axial_step(family))
        return
    terms = route == "terms" or (route == "auto" and family.route == "terms")
    root, fils = _straighten_all(config, run_names, input_uri, tomogram, family, tilt_range, tilt_axis,
                                 min_length or 2 * (family.term_segment_length if terms else family.segment_length),
                                 work_dir, keep_volumes, workers=workers)
    if terms:
        return _polarity_terms(root, fils, family, work_dir, label, output_uri, picks_uri, every, seeds_only, tomogram,
                               source, tomograms_star, tiltseries_dir, binning, size_unit_A, snr, workers)
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
    reg = registration_particles(family, fils, params, it)
    dense = registration_particles(family, fils, params, it, every=every) if picks_uri else {}
    _write_copick(root, fils, family, table, call_col, reg, dense, output_uri, picks_uri, seeds_only, gate["detected"],
                  ref_path)


def _write_copick(root, fils, family, table, call_col, reg, dense, output_uri, picks_uri, seeds_only, lattice_detected,
                  ref_path):
    """Centre-line filaments (ordered minus -> plus when known; analysis in metadata), registration picks under the
    same URI, and optional dense picks."""
    rows = table.set_index("filament")
    by_run: dict = {}
    for name, st in fils.items():
        if name in rows.index:
            by_run.setdefault(st.meta["run"], []).append(name)

    def selected(names, store):
        return [n for n in names if n in store and (not seeds_only or bool(rows.loc[n].get("seed", False)))]

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
                                                  "seed": seed, "lattice_detected": lattice_detected,
                                                  "polarity_convention": "minus_to_plus" if call in ("plus", "minus") else None,
                                                  "registration_picks": output_uri, "reference_map": ref_path,
                                                  "analysis": info}}})
            io.write_centrelines(root, run, obj, user, session, items)
            sel = selected(names, reg)
            if sel:
                io.write_particles(root, run, obj, user, session, np.concatenate([reg[n][0] for n in sel]),
                                   np.concatenate([reg[n][1] for n in sel]),
                                   np.concatenate([np.full(len(reg[n][0]), fils[n].meta["instance_id"]) for n in sel]),
                                   np.concatenate([reg[n][2] for n in sel]))
        if picks_uri:
            obj, user, session = _filament_uri(picks_uri)
            sel = selected(names, dense)
            if sel:
                io.write_particles(root, run, obj, user, session, np.concatenate([dense[n][0] for n in sel]),
                                   np.concatenate([dense[n][1] for n in sel]),
                                   np.concatenate([np.full(len(dense[n][0]), fils[n].meta["instance_id"]) for n in sel]),
                                   np.concatenate([dense[n][2] for n in sel]))
    click.echo(f"wrote {output_uri or ''} {picks_uri or ''} for {len(by_run)} runs")


def _polarity_terms(root, fils, family, work_dir, label, output_uri, picks_uri, every, seeds_only, tomogram, source,
                    tomograms_star, tiltseries_dir, binning, size_unit_A, snr, workers):
    """The term route (actin; any family with a term model): segments from the tilt series or the straightened
    tomograms, band-limited terms, data-built reference, decoys."""
    from . import bands, tiltseries
    from .pipeline import term_average, term_registration_particles, term_segments

    L = family.term_segment_length
    if source == "tiltseries":
        if not tomograms_star:
            raise click.UsageError("--source tiltseries needs --tomograms-star")
        src = tiltseries.TiltSeriesSource(tomograms_star, tiltseries_dir or os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(tomograms_star)))), binning=binning, snr=snr, size_unit_A=size_unit_A,
            scratch=os.path.join(work_dir, "tmp"))
        os.makedirs(src.scratch, exist_ok=True)
        tomo_type, vs = _tomo_uri(tomogram)
        extents = {}
        for run in {st.meta["run"] for st in fils.values()}:
            extents[run] = (io.tomogram_extent(root, run, vs, tomo_type)[0], vs)
        segs = tiltseries.reconstruct(fils, L, family.term_half_width, src, extents,
                                      os.path.join(work_dir, "segments"), workers=workers, log=click.echo)
    else:
        segs = term_segments(fils, L)
    res = bands.analyse(segs, family, bands.TermConfig(r_out=family.term_r_out), workers=workers, label=label,
                        log=click.echo)
    s = res.summary
    geo = {n: (st.meta["run"], st.centres[::10], st.t[::10]) for n, st in fils.items()}
    s["bundle_pairs_same_polarity"] = {}
    for key, tab in (("data", res.table), ("decoy", res.decoy_table)):
        if tab is None:
            continue
        pr = bands.bundle_pairs(geo, dict(zip(tab.filament, tab.pol)), list(tab.filament))
        s["bundle_pairs_same_polarity"][key] = [int(sum(p[2] for p in pr)), len(pr)]
    table = res.table
    os.makedirs(work_dir, exist_ok=True)
    table.to_csv(os.path.join(work_dir, "calls.tsv"), sep="\t", index=False)
    res.segments.to_csv(os.path.join(work_dir, "segments.tsv"), sep="\t", index=False)
    if res.decoy_table is not None:
        res.decoy_table.to_csv(os.path.join(work_dir, "decoy_calls.tsv"), sep="\t", index=False)
    seeds = set(table.filament[table.seed])
    ref, rstep = term_average(family, segs, res, seeds or None)
    ref_path = None
    if ref is not None:
        ref_path = os.path.abspath(os.path.join(work_dir, f"reference_{family.name}.mrc"))
        with mrcfile.new(ref_path, overwrite=True) as m:
            m.set_data(ref.astype(np.float32))
            m.voxel_size = rstep
    s.update({"family": family.name, "version": __version__, "route": "terms", "source": source,
              "reference_map": ref_path})
    json.dump(s, open(os.path.join(work_dir, "summary.json"), "w"), indent=1, default=str)
    click.echo(f"enrichment (data / decoy): " + ", ".join(f"{b}: {v['data']:.2f}/{v.get('decoy', float('nan')):.2f}"
                                                          for b, v in s["enrichment"].items()))
    if "data" in s:
        click.echo(f"halves agree {s['data']['halves_agree']}/{s['data']['halves_n']} (decoy "
                   f"{s['decoy']['halves_agree']}/{s['decoy']['halves_n']}); z >= 3: {s['data']['z_ge_seed']} (decoy "
                   f"{s['decoy']['z_ge_seed']}); bundle pairs same polarity {s['bundle_pairs_same_polarity']}; "
                   f"polarity detected: {s['polarity_detected']}; seeds {s['seeds']}")
    if not (output_uri or picks_uri):
        return
    reg = term_registration_particles(family, fils, res, L)
    dense = term_registration_particles(family, fils, res, L, every=every) if picks_uri else {}
    _write_copick(root, fils, family, table, "call", reg, dense, output_uri, picks_uri, seeds_only,
                  s.get("lattice_detected"), ref_path)


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
