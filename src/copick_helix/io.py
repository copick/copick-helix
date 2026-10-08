"""copick in and out.

Reading: filament traces (``object:user/session``), tomograms (voxel spacing, type).
Writing:
- filaments: the recentered center lines as Catmull-Rom curves, ordered minus -> plus (pointed -> barbed) when the
  polarity is known, with the analysis under ``metadata["copick_helix"]``;
- picks: lattice-registered particles with full transforms (+Z towards the plus end), one per segment (the
  registration) or dense (for averaging).
"""

from __future__ import annotations

import numpy as np
import zarr


def open_project(config: str):
    import copick

    return copick.from_file(config)


def read_filaments(root, run_name: str, object_name: str, user_id: str, session_id: str):
    """[(instance_id, points (N, 3) in A, filament object)] of one filament set."""
    run = root.get_run(run_name)
    sets = run.get_filaments(object_name=object_name, user_id=user_id, session_id=session_id)
    if not sets:
        return []
    return [(f.instance_id, np.asarray(f.points, float), f) for f in sets[0].filaments]


def read_tomogram(root, run_name: str, voxel_spacing: float, tomo_type: str, level: str = "0") -> np.ndarray:
    run = root.get_run(run_name)
    tomo = run.get_voxel_spacing(voxel_spacing).get_tomograms(tomo_type)[0]
    return np.asarray(zarr.open(tomo.zarr(), mode="r")[level][:], dtype=np.float32)


def tomogram_extent(root, run_name: str, voxel_spacing: float, tomo_type: str):
    """(x, y, z) physical extent of a tomogram in A, from its zarr shape (no data read)."""
    run = root.get_run(run_name)
    tomo = run.get_voxel_spacing(voxel_spacing).get_tomograms(tomo_type)[0]
    shape = zarr.open(tomo.zarr(), mode="r")["0"].shape  # (z, y, x)
    return shape[2] * voxel_spacing, shape[1] * voxel_spacing, shape[0] * voxel_spacing


def write_centerlines(
    root,
    run_name: str,
    object_name: str,
    user_id: str,
    session_id: str,
    items: list[dict],
    control_spacing: float = 100.0,
    step: float = 10.0,
):
    """Filaments as Catmull-Rom curves through the recentered center line (control points every ``control_spacing``
    A, points regenerated every ``step`` A), so every copick viewer shows the line the analysis used.

    Each item: {"instance_id", "centers" (n, 3) A along the analysis direction, "reverse" (order minus -> plus),
    "known" (polarity_known), "radius", "metadata"}."""
    from copick.models import CopickFilament

    run = root.get_run(run_name)
    out = run.new_filaments(object_name, session_id, user_id, exist_ok=True)
    fils = []
    for it in items:
        c = np.asarray(it["centers"], float)
        if it["reverse"]:
            c = c[::-1]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        marks = np.unique(np.r_[np.arange(0, seg[-1], control_spacing), seg[-1]])
        ctrl = np.stack([np.interp(marks, seg, c[:, d]) for d in range(3)], 1)
        fils.append(
            CopickFilament.from_control_points(
                int(it["instance_id"]),
                ctrl.tolist(),
                step=step,
                kind="catmull-rom",
                alpha=0.5,
                polarity_known=bool(it["known"]),
                radius=it.get("radius"),
                metadata=it.get("metadata", {}),
            ),
        )
    out.filaments = fils
    out.store()
    return out


def write_particles(
    root,
    run_name: str,
    object_name: str,
    user_id: str,
    session_id: str,
    positions,
    rotations,
    instance_ids,
    scores,
):
    """copick picks with full transforms (rotation maps the particle's reference frame onto the tomogram; +Z towards
    the plus end), instance_id = filament ID."""
    run = root.get_run(run_name)
    picks = run.new_picks(object_name, session_id, user_id, exist_ok=True)
    T = np.tile(np.eye(4), (len(positions), 1, 1))
    T[:, :3, :3] = rotations
    picks.from_numpy(
        np.asarray(positions, float),
        T,
        instance_ids=np.asarray(instance_ids, int),
        scores=np.asarray(scores, float),
    )
    picks.store()
    return picks
