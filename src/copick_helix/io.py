"""copick in and out.

Reading: filament traces (``object:user/session``), tomograms (voxel spacing, type).
Writing: oriented filaments in a new session. With ``polarity_known`` the point order follows the polarity, using the
convention recorded in each filament's metadata (``copick_helix.polarity_convention``): minus -> plus for
microtubules, pointed -> barbed for actin. Every filament also carries the method, call and confidence measures under
``metadata["copick_helix"]``.
"""

from __future__ import annotations

import numpy as np
import zarr

CONVENTION = {"microtubule": "minus_to_plus", "actin": "pointed_to_barbed"}


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


def write_oriented(root, run_name: str, object_name: str, user_id: str, session_id: str, filaments: list[dict],
                   family_kind: str):
    """Write filaments with their points reoriented to the polarity convention.

    Each item: {"source": CopickFilament, "first_is_plus": bool | None, "known": bool, "info": dict}.
    ``first_is_plus`` True means the plus (barbed) end is at the trace's first point; the points are reversed so the
    order runs minus -> plus (pointed -> barbed). ``known`` sets ``polarity_known`` (confident calls only)."""
    from copick.models import CopickFilament

    run = root.get_run(run_name)
    out = run.new_filaments(object_name, session_id, user_id, exist_ok=True)
    new = []
    for item in filaments:
        f = item["source"]
        pts = np.asarray(f.points, float)
        if item["first_is_plus"]:
            pts = pts[::-1]
        meta = dict(f.metadata or {})
        meta["copick_helix"] = {**item.get("info", {}), "polarity_convention": CONVENTION.get(family_kind),
                                "reoriented": bool(item["first_is_plus"])}
        new.append(CopickFilament(instance_id=f.instance_id, points=[tuple(p) for p in pts],
                                  polarity_known=bool(item["known"]), score=f.score, radius=f.radius, metadata=meta))
    out.filaments = new
    out.store()
    return out
