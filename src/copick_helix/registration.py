"""From per-segment registrations to lattice-registered, oriented particles.

The iterative reference registers each segment: the oriented reference G equals the segment (or, with ``flip``, its
copy turned 180 deg about x) rotated by +roll about the axis and shifted by +shift along it. A reference point x
therefore sits in the segment's straightened frame at

    y = F_p Rz(-roll) (x - shift z^) (+ L z^ if flip),   F_p = diag(1, -1, -1) if flip else 1,

because the flip (complex conjugation in Bessel space) reflects z about the segment centre once unwrapped.

and a particle whose reference frame maps onto the tomogram by the rotation A (RELION / copick convention:
tomogram vector = A reference vector) has A = [e1 e2 t] F_p Rz(-roll).

Equivalent lattice positions differ by the family's screw (P, omega): the reference is invariant under a rotation by
omega about z plus a shift by P along z (a microtubule dimer with omega = 0; a helical subunit with its twist). The
particle at reference point (0, 0, z0 + j P) sees the same density with the reference frame turned by Rz(j omega).

Exported frames point +Z towards the plus (barbed) end, copick's filament pick convention once points are ordered
minus -> plus. If the family's reference has its plus end at -z, the exported rotation is A diag(1, -1, -1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Registration:
    segment: int
    flip: bool
    roll_deg: float
    shift_A: float  # on the reference grid
    score: float


def lattice_particles(st, reg: Registration, segment_length: float, factor: float, screw: tuple[float, float],
                      plus_at_minus_z: bool, z0: float | None = None, every: int = 1, centre_only: bool = False):
    """Lattice-registered particles of one segment.

    st: Straightened filament (centres, t, e1, e2 at ``st.step``); factor: original / reference axial scale (the
    segment was stretched by 1/factor onto the reference grid); screw: (P A on the reference grid, omega deg).
    Returns positions (n, 3) in tomogram A, rotations (n, 3, 3) mapping the exported reference frame to the tomogram,
    and arc lengths (n,) in A along ``st``."""
    L = segment_length
    P, omega = screw
    z0 = L / 2 if z0 is None else z0
    F_p = FLIP if reg.flip else np.eye(3)
    R_loc = F_p @ rz(-reg.roll_deg)
    sign = -1.0 if reg.flip else 1.0
    # the alignment shift is circular over the segment, but the lattice is not periodic in L: read it as the
    # nearest equivalent in (-L/2, L/2] and place lattice points without wrapping
    shift = (reg.shift_A + L / 2) % L - L / 2
    js = np.arange(-int(np.ceil(2 * L / P)) - 2, int(np.ceil(2 * L / P)) + 3)
    # flip = complex conjugation in Bessel space = z -> -z modulo L, i.e. a reflection about the segment centre once
    # unwrapped (z -> L - z), not about its start
    u = sign * (z0 + js * P - shift) + (L if reg.flip else 0.0)  # position within the segment (reference grid)
    inside = (u >= 0) & (u < L)
    js, u = js[inside], u[inside]
    order = np.argsort(u)
    js, u = js[order], u[order]
    if centre_only:
        i = int(np.argmin(np.abs(u - L / 2)))
        js, u = js[i:i + 1], u[i:i + 1]
    elif every > 1:
        js, u = js[::every], u[::every]
    s_ref = reg.segment * L + u
    s = s_ref * factor  # arc length along the original straightened filament
    n_s = len(st.centres)
    idx = np.clip(s / st.step, 0, n_s - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n_s - 1)
    w = (idx - i0)[:, None]
    pos = (1 - w) * st.centres[i0] + w * st.centres[i1]
    rots = []
    for k, j in enumerate(js):
        i = int(round(idx[k]))
        B = np.stack([st.e1[i], st.e2[i], st.t[i]], axis=1)
        A = B @ R_loc @ rz(j * omega)
        rots.append(A @ FLIP if plus_at_minus_z else A)
    return pos, np.array(rots), s


def extract(vol: np.ndarray, step: float, origin: np.ndarray, pos: np.ndarray, R: np.ndarray, half: tuple,
            out_step: float | None = None) -> np.ndarray:
    """Subvolume in a particle's reference frame: sample ``vol[z, y, x]`` (voxel ``step``, voxel 0 at ``origin`` A)
    at pos + R q for q on a grid of half-widths ``half`` = (hz, hy, hx) A. Returns [z, y, x]."""
    from scipy.ndimage import map_coordinates

    out_step = out_step or step
    gz, gy, gx = (np.arange(-h, h + 1e-6, out_step) for h in half)
    Q = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1)  # (x, y, z) components, shape (nx, ny, nz, 3)
    X = pos[None, None, None, :] + Q @ R.T - np.asarray(origin)[None, None, None, :]
    idx = np.stack([X[..., 2], X[..., 1], X[..., 0]], 0) / step
    sub = map_coordinates(vol, idx.reshape(3, -1), order=1, mode="constant").reshape(Q.shape[:3])
    return np.transpose(sub, (2, 1, 0)).astype(np.float32)


def local_frames(n_s: int, step: float, n_inplane: int):
    """A Straightened-like object with identity frames for a volume on the straightened / reference grid: tomogram
    coordinates equal grid coordinates (x = e1, y = e2, z = s; axis at the in-plane centre index)."""
    from types import SimpleNamespace

    c = np.stack([np.full(n_s, (n_inplane // 2) * step), np.full(n_s, (n_inplane // 2) * step),
                  np.arange(n_s) * step], 1)
    eye = np.eye(3)
    return SimpleNamespace(centres=c, step=step, t=np.tile(eye[2], (n_s, 1)), e1=np.tile(eye[0], (n_s, 1)),
                           e2=np.tile(eye[1], (n_s, 1)))


def term_particles(st, s_centre: float, reg: Registration, offset: tuple, screw: tuple[float, float],
                   plus_at_minus_z: bool, half_length: float, every: int = 1, centre_only: bool = False):
    """Lattice-registered particles of a segment registered in the term route (``bands``), whose coordinates have the
    segment centre as origin: a reference point x sits in the segment at

        y = F_p Rz(-roll) (x - shift z^) + (dx, dy, 0),

    (dx, dy) = ``offset``, the refined axis position along (e1, e2). ``st``: centre line and frames along the filament
    (centres, e1, e2, t sampled every st.step A); ``s_centre``: arc length of the segment centre. Lattice points
    (0, 0, j P) within +-half_length of the centre; rotations A = [e1 e2 t] F_p Rz(-roll) Rz(j omega), turned by FLIP
    when the reference has its plus end at -z. Returns positions (n, 3), rotations (n, 3, 3), arc lengths (n,)."""
    P, omega = screw
    F_p = FLIP if reg.flip else np.eye(3)
    R_loc = F_p @ rz(-reg.roll_deg)
    sign = -1.0 if reg.flip else 1.0
    k = int(np.ceil(half_length / P)) + 2
    js = np.arange(-k, k + 1)
    y = sign * (js * P - reg.shift_A)
    inside = np.abs(y) <= half_length
    js, y = js[inside], y[inside]
    order = np.argsort(y)
    js, y = js[order], y[order]
    if centre_only:
        i = int(np.argmin(np.abs(y)))
        js, y = js[i:i + 1], y[i:i + 1]
    elif every > 1:
        js, y = js[::every], y[::every]
    s = s_centre + y
    n_s = len(st.centres)
    idx = np.clip(s / st.step, 0, n_s - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, n_s - 1)
    w = (idx - i0)[:, None]
    pos = (1 - w) * st.centres[i0] + w * st.centres[i1]
    rots = []
    for q, j in enumerate(js):
        i = int(round(idx[q]))
        B = np.stack([st.e1[i], st.e2[i], st.t[i]], axis=1)
        pos[q] = pos[q] + offset[0] * st.e1[i] + offset[1] * st.e2[i]
        A = B @ R_loc @ rz(j * omega)
        rots.append(A @ FLIP if plus_at_minus_z else A)
    return pos, np.array(rots).reshape(-1, 3, 3), s
