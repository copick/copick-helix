"""Dense, lattice-registered sampling from stored results (no recomputation).

Input: the filaments written by ``helix-polarity`` (recentred centre lines; family and helical parameters in
``metadata["copick_helix"]``) and their registration picks (one per segment, full transform, +Z towards the plus end).

From each registration pick k at arc length s_k along the filament, lattice point j lies at s_k + j * d * P_s (P_s =
the filament's own axial screw step, d = +-1 so that +j follows the pick's +Z) with rotation

    R_j = B(s_j) B(s_k)^T R_k Rz(j omega),

B(s) the rotation-minimising frame along the stored centre line (only its transport between s_k and s_j matters) and
(P, omega) the family's screw (the exported frame is invariant under it as well). Each registration pick covers the
part of the filament nearer to it than to its neighbours, at most half a segment either side of it.
"""

from __future__ import annotations

import numpy as np

from .families import get_family
from .geometry import frames
from .registration import rz


def _arc(points):
    p = np.asarray(points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.gradient(p, s, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    e1, e2 = frames(p, t, np.array([0.0, 0.0, 1.0]))
    return p, s, t, e1, e2


def _at(s_query, s, arr):
    return np.stack([np.interp(s_query, s, arr[:, d]) for d in range(arr.shape[1])], -1)


def dense_from_registration(points, metadata: dict, reg_positions, reg_rotations, every: int = 1):
    """Positions (n, 3) and rotations (n, 3, 3) of every ``every``-th lattice point along one filament."""
    info = metadata["copick_helix"]
    family = get_family(info["family"])
    params = {k: float(info["analysis"][k]) for k in family.reference_params}
    P_ref, omega = family.screw(family.reference_params)
    key = "monomer_repeat" if "monomer_repeat" in family.reference_params else "rise"
    P_s = P_ref * params[key] / family.reference_params[key]
    p, s, t, e1, e2 = _arc(points)
    B = np.stack([e1, e2, t], axis=2)  # (n, 3, 3), columns e1, e2, t
    # registration picks along the filament
    sk = np.array([s[np.argmin(np.linalg.norm(p - q, axis=1))] for q in reg_positions])
    order = np.argsort(sk)
    sk, Rk = sk[order], np.asarray(reg_rotations)[order]
    # each registration covers the stretch nearer to it than to its neighbours, at most half a segment either side
    # (what its segment registered); filament tails outside every analysed segment are not sampled
    half = 0.5 * family.segment_length * params[key] / family.reference_params[key]
    bounds = np.r_[max(s[0], sk[0] - half), 0.5 * (sk[1:] + sk[:-1]), min(s[-1], sk[-1] + half)]
    pos, rots = [], []
    for k in range(len(sk)):
        tk = _at(sk[k], s, t)
        tk /= np.linalg.norm(tk)
        d = 1.0 if Rk[k][:, 2] @ tk >= 0 else -1.0
        lo, hi = (bounds[k] - sk[k]) / P_s, (bounds[k + 1] - sk[k]) / P_s  # range of j * d
        j_lo, j_hi = (int(np.ceil(lo)), int(np.floor(hi))) if d > 0 else (int(np.ceil(-hi)), int(np.floor(-lo)))
        Bk = _nearest_frame(B, s, sk[k])
        for j in range(j_lo, j_hi + 1):
            if j % every:
                continue
            sj = sk[k] + j * d * P_s
            if sj < s[0] or sj > s[-1]:
                continue
            Bj = _nearest_frame(B, s, sj)
            pos.append(_at(sj, s, p))
            rots.append(Bj @ Bk.T @ Rk[k] @ rz(j * omega))
    return np.array(pos), np.array(rots)


def _nearest_frame(B, s, sq):
    i = int(np.clip(np.searchsorted(s, sq), 0, len(s) - 1))
    return B[i]
