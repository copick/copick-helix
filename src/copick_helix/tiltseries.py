"""Local reconstructions of filament segments from the tilt series (zarr-particle-tools API).

For thin filaments (actin, intermediate filaments) a CTF-deconvolved tomogram cannot follow the depth-dependent
defocus, and the polar terms beyond the first CTF zero are lost. Here every segment is reconstructed on its own from
the tilt series: one particle per segment, centered on the recentered center line (from the straightened tomogram) at
the segment center and oriented as the local filament frame A = [e1 e2 t] (RELION / zpt map a reference vector a to
A a in the tomogram). It is extracted with zarr-particle-tools' 2D extraction (its own per-particle defocus),
back-projected on its own, gridding-corrected, and Wiener CTF-corrected with offset 1 / ``snr``. RELION's
radial-average heuristic is not used, because a single particle has too few Fourier planes for it.

Each output is ``vol[z = t, y = e2, x = e1]``, cropped in-plane to +-``half_width`` around the axis, centered at the
box center (index n // 2 along every axis), in the frame the term route (``bands``) and the registration use.

Coordinates: zarr-particle-tools takes particle centers relative to the tomogram center, rlnTomoSize * unit / 2.
The unit is a tilt-series pixel in RELION / ApexAgent stars, but a tomogram voxel in zarr-particle-tools' stars made
from portal data. ``size_unit_A=None`` picks whichever candidate (tilt-series pixel, original pixel, the copick
tomogram's voxel) makes rlnTomoSize match the copick tomogram's physical extent.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OPTICS_COLS = ["rlnOpticsGroup", "rlnOpticsGroupName", "rlnSphericalAberration", "rlnVoltage", "rlnAmplitudeContrast",
               "rlnTomoTiltSeriesPixelSize", "rlnTomoName"]  # zarr_particle_tools.core.constants.OPTICS_DF_COLUMNS


@dataclass
class TiltSeriesSource:
    tomograms_star: str  # RELION tomograms.star (rlnTomoTiltSeriesStarFile per tilt series)
    project_dir: str  # the tilt-series star paths are relative to this directory
    binning: int = 3
    snr: float = 0.1  # Wiener offset 1 / snr
    size_unit_A: float | None = None  # A per rlnTomoSize unit; None: from the copick tomogram extent
    scratch: str | None = None  # temporary extraction directory (default $TMPDIR)


def matrix_euler(A):
    """RELION Euler_matrix2angles (degrees)."""
    abs_sb = np.hypot(A[0, 2], A[1, 2])
    if abs_sb > 16 * np.finfo(float).eps:
        psi = np.arctan2(A[1, 2], -A[0, 2])
        rot = np.arctan2(A[2, 1], A[2, 0])
        sign_sb = np.sign(A[1, 2] / np.sin(psi)) if abs(np.sin(psi)) > 1e-6 else np.sign(-A[0, 2] / np.cos(psi))
        tilt = np.arctan2(sign_sb * abs_sb, A[2, 2])
    else:
        rot = 0.0
        tilt, psi = (0.0, np.arctan2(-A[1, 0], A[0, 0])) if A[2, 2] > 0 else (np.pi, np.arctan2(A[1, 0], -A[0, 0]))
    return np.degrees([rot, tilt, psi])


def read_tomograms(path: str) -> pd.DataFrame:
    import starfile

    t = starfile.read(path)
    t = t["global"] if isinstance(t, dict) else t
    t = t.copy()
    t["rlnTomoName"] = t["rlnTomoName"].astype(str)
    return t


def run_key(name: str) -> str:
    """copick run name of a tomograms.star rlnTomoName ('35921', or 'run_16848_tiltseries_..._spacing_...')."""
    return name.split("_")[1] if name.startswith("run_") else name


def size_unit(row, extent_A, voxel_A=None, tol=0.03) -> float:
    """A per rlnTomoSize unit: the candidate that makes rlnTomoSizeX * unit match the copick extent along x."""
    size = float(row["rlnTomoSizeX"])
    cands = [float(row["rlnTomoTiltSeriesPixelSize"])]
    if "rlnMicrographOriginalPixelSize" in row:
        cands.append(float(row["rlnMicrographOriginalPixelSize"]))
    if voxel_A:
        cands.append(float(voxel_A))
    err = [abs(size * u - extent_A) / extent_A for u in cands]
    i = int(np.argmin(err))
    if err[i] > tol:
        raise ValueError(f"rlnTomoSizeX {size} matches no pixel size {cands} for a {extent_A:.0f} A tomogram "
                         f"(best {100 * err[i]:.1f}% off); pass size_unit_A")
    return cands[i]


def plan(filaments: dict, segment_length: float, source: TiltSeriesSource, extents: dict, out_dir: str) -> dict:
    """Segments per tilt series. ``filaments``: name -> Straightened (meta: run); ``extents``: run -> (x extent A,
    copick voxel A). Segment k is centered at index k n + n // 2 of the straightened center line (n = L / step), the
    center the term route and the registration assume."""
    tomos = read_tomograms(source.tomograms_star)
    tomos.index = [run_key(n) for n in tomos.rlnTomoName]
    by_tomo = {}
    for name, st in filaments.items():
        run = str(st.meta["run"])
        if run not in tomos.index:
            continue
        row = tomos.loc[run]
        unit = source.size_unit_A or size_unit(row, *extents[run])
        size = np.array([row[k] for k in ("rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ")], float)
        center = size * unit / 2
        n = int(round(segment_length / st.step))
        for k in range(len(st.centers) // n):
            i = k * n + n // 2
            B = np.stack([st.e1[i], st.e2[i], st.t[i]], axis=1)
            by_tomo.setdefault(str(row.rlnTomoName), []).append({
                "filament": name, "segment": k, "s_index": int(i), "s_center_A": float(i * st.step),
                "center_A": st.centers[i].tolist(), "B": B, "centered": (st.centers[i] - center).tolist(),
                "euler": matrix_euler(B).tolist(), "beam": st.beam_local[i].tolist(), "tilt_axis": st.tilt_local[i].tolist(),
                "stem": os.path.join(out_dir, name, f"seg{k:03d}")})
    return by_tomo


def _reconstruct_tomogram(job):
    tname, segs, source, box, half_px = job
    from zarr_particle_tools.core.backprojection import ctf_correct_3d_wiener, gridding_correct_3d_sinc2
    from zarr_particle_tools.core.helpers import get_tiltseries_data
    from zarr_particle_tools.subtomo_extract import process_tiltseries
    from zarr_particle_tools.subtomo_reconstruct import reconstruct_single_tiltseries

    todo = [s for s in segs if not os.path.exists(s["stem"] + ".npy")]
    if not todo:
        return tname, 0, 0.0
    t0 = time.time()
    project = Path(source.project_dir)
    tomos = read_tomograms(source.tomograms_star)
    row = tomos[tomos.rlnTomoName == tname].iloc[0]
    parts = pd.DataFrame([{
        "rlnTomoName": tname, "rlnCenteredCoordinateXAngst": s["centered"][0],
        "rlnCenteredCoordinateYAngst": s["centered"][1], "rlnCenteredCoordinateZAngst": s["centered"][2],
        "rlnAngleRot": s["euler"][0], "rlnAngleTilt": s["euler"][1], "rlnAnglePsi": s["euler"][2],
        "rlnOriginXAngst": 0.0, "rlnOriginYAngst": 0.0, "rlnOriginZAngst": 0.0, "rlnOpticsGroup": 1,
        "rlnTomoParticleName": f"{tname}/{j + 1}"} for j, s in enumerate(todo)])
    parts.index = np.arange(1, len(parts) + 1)
    optics = tomos[OPTICS_COLS].drop_duplicates()
    optics = optics[optics.rlnTomoName == tname].reset_index(drop=True)
    args = get_tiltseries_data(particles_df=parts, optics_df=optics, trajectories_dict=None, tiltseries_row_entry=row,
                               tiltseries_relative_dir=project, tomograms_starfile=Path(source.tomograms_star),
                               tomograms_data=tomos)
    scratch = Path(tempfile.mkdtemp(prefix="helixts_", dir=source.scratch or os.environ.get("TMPDIR", "/tmp")))
    try:
        # the constants zarr-particle-tools' reconstruct_local passes to its extraction
        extracted, _ = process_tiltseries(
            **args, box_size=box, crop_size=box, bin=source.binning, float16=False, no_ctf=True, circle_precrop=True,
            no_circle_crop=True, dont_apply_offsets=False, no_ic=False, normalize_bin=False, write_fourier=True,
            tiltseries_relative_dir=project, output_dir=scratch, debug=False)
        opt = args["optics_row"].copy()  # as zarr-particle-tools updates the optics for 2D stacks
        opt["rlnCtfDataAreCtfPremultiplied"] = 0
        opt["rlnImageDimensionality"] = 2
        opt["rlnTomoSubtomogramBinning"] = float(source.binning)
        opt["rlnImagePixelSize"] = opt["rlnTomoTiltSeriesPixelSize"] * source.binning
        opt["rlnImageSize"] = box
        apix = float(opt["rlnImagePixelSize"].iloc[0])
        names = set(extracted["rlnTomoParticleName"])
        for j, s in enumerate(todo):
            pname = f"{tname}/{j + 1}"
            if pname not in names:
                continue
            one = extracted[extracted.rlnTomoParticleName == pname]
            d1, w1, d2, w2, _ = reconstruct_single_tiltseries(
                no_ctf=False, cutoff_fraction=0.01, filtered_particles_df=one, filtered_trajectories_dict=None,
                tiltseries_row_entry=row, individual_tiltseries_df=args["individual_tiltseries_df"], optics_row=opt)
            vol = gridding_correct_3d_sinc2(particle_fourier_volume=d1 + d2)
            vol = ctf_correct_3d_wiener(real_space_volume=vol, weights_fourier_volume=w1 + w2,
                                        wiener_offset=1.0 / source.snr)
            c = box // 2
            core = np.asarray(vol[:, c - half_px:c + half_px + 1, c - half_px:c + half_px + 1], dtype=np.float32)
            os.makedirs(os.path.dirname(s["stem"]), exist_ok=True)
            np.save(s["stem"] + ".npy", core)
            meta = {k: v for k, v in s.items() if k not in ("B", "stem")}
            meta.update({"frame_columns_e1_e2_t": np.asarray(s["B"]).tolist(), "step_A": apix, "box": box,
                         "bin": source.binning, "array_axes": ["z=t", "y=e2", "x=e1"], "axis_index_inplane": half_px,
                         "protein_sign": 1.0, "wiener_snr": source.snr, "tomo_name": tname,
                         "source": "zarr-particle-tools API: process_tiltseries + reconstruct_single_tiltseries"})
            json.dump(meta, open(s["stem"] + ".json", "w"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return tname, len(todo), time.time() - t0


def reconstruct(filaments: dict, segment_length: float, half_width: float, source: TiltSeriesSource, extents: dict,
                out_dir: str, workers: int = 4, log=print) -> dict:
    """Reconstruct (or reuse from ``out_dir``) every segment; returns {filament: [bands.SegmentRef]}.

    The box is cubic, the smallest even number of binned pixels holding ``segment_length``; the in-plane crop keeps
    +-``half_width`` A (CTF correction happens on the full box first, so delocalised signal is already back)."""
    import multiprocessing as mp

    from .bands import SegmentRef

    by_tomo = plan(filaments, segment_length, source, extents, out_dir)
    tomos = read_tomograms(source.tomograms_star)
    px = float(tomos.rlnTomoTiltSeriesPixelSize.iloc[0]) * source.binning
    box = int(np.ceil(segment_length / px / 2)) * 2
    half_px = int(round(half_width / px))
    n = sum(len(v) for v in by_tomo.values())
    log(f"{n} segments in {len(by_tomo)} tilt series: box {box} px at {px:.3f} A ({box * px:.0f} A), crop +-{half_px} px")
    jobs = [(t, s, source, box, half_px) for t, s in by_tomo.items()]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            for tname, k, dt in pool.imap_unordered(_reconstruct_tomogram, jobs):
                log(f"{tname}: {k} segments in {dt:.0f} s")
    else:
        for j in jobs:
            tname, k, dt = _reconstruct_tomogram(j)
            log(f"{tname}: {k} segments in {dt:.0f} s")
    out = {}
    for segs in by_tomo.values():
        for s in segs:
            if os.path.exists(s["stem"] + ".npy"):
                out.setdefault(s["filament"], []).append(SegmentRef(s["filament"], s["segment"], path=s["stem"]))
    return {k: sorted(v, key=lambda r: r.index) for k, v in out.items()}
