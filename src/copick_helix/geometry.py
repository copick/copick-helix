"""Straightening and recentering of a traced filament.

The trace is resampled at 10 A, smoothed (Gaussian, sigma 40 A) and evaluated every ``step`` A with a cubic spline.
Frames are rotation-minimizing (double reflection): t = unit tangent, e1 starts as the beam projected onto the
normal plane, e2 = t x e1. The tomogram is sampled on c(s) + u e2 + v e1 (u, v within +-half width), giving
``vol[s, u, v]`` = (z, y, x) of a right-handed volume with x = e1, y = e2, z = t. +s runs from the first to the last
trace point.

Recentering: in windows along s, the mean cross-section is correlated with the family's radial template (a ring for
microtubules, a rod for actin), using only the cross-section Fourier directions the tilt series sampled; the
offsets are smoothed along s, applied to the center line, and the volume is resampled (twice by default).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline

from .families import Family


@dataclass
class TiltGeometry:
    """Tilt axis and beam in tomogram (x, y, z) coordinates, and the tilt range (deg).

    Defaults: AreTomo3 / CryoET Data Portal tomograms, tilt axis along y, beam along z."""

    tilt_axis: tuple = (0.0, 1.0, 0.0)
    beam: tuple = (0.0, 0.0, 1.0)
    tilt_range: tuple = (-60.0, 60.0)

    def normals(self, step_deg: float = 1.0) -> np.ndarray:
        """Normals of the Fourier planes the tilts sample (beam rotated about the tilt axis)."""
        a = np.asarray(self.tilt_axis, float) / np.linalg.norm(self.tilt_axis)
        b = np.asarray(self.beam, float) / np.linalg.norm(self.beam)
        th = np.radians(np.arange(self.tilt_range[0], self.tilt_range[1] + 1e-6, step_deg))
        return (np.cos(th)[:, None] * b[None] + np.sin(th)[:, None] * np.cross(a, b)[None]
                + (1 - np.cos(th))[:, None] * (a @ b) * a[None])


def resample(p: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, seg > 1e-6]]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    n = max(int(np.floor(s[-1] / step)) + 1, 2)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, p[:, d]) for d in range(3)], 1)


def curve(points, step: float, sigma: float = 40.0):
    """Smoothed center line every ``step`` A and unit tangents."""
    p = resample(np.asarray(points, float), 10.0)
    p = ndi.gaussian_filter1d(p, sigma / 10.0, axis=0, mode="nearest")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    cs = CubicSpline(s, p, axis=0)
    si = np.arange(int(np.floor(s[-1] / step)) + 1) * step
    c = cs(si)
    t = cs(si, 1)
    return c, t / np.linalg.norm(t, axis=1, keepdims=True)


def frames(c, t, beam):
    """Rotation-minimizing frames (double reflection); e1 starts as the beam projected on the normal plane."""
    beam = np.asarray(beam, float)
    e1 = np.empty_like(c)
    r = beam - np.dot(beam, t[0]) * t[0]
    if np.linalg.norm(r) < 1e-3:
        r = np.array([1.0, 0, 0]) - t[0, 0] * t[0]
    e1[0] = r / np.linalg.norm(r)
    for i in range(len(c) - 1):
        v1 = c[i + 1] - c[i]
        c1 = v1 @ v1
        rl = e1[i] - (2 / c1) * (v1 @ e1[i]) * v1
        tl = t[i] - (2 / c1) * (v1 @ t[i]) * v1
        v2 = t[i + 1] - tl
        c2 = v2 @ v2
        r = rl - (2 / c2) * (v2 @ rl) * v2 if c2 > 1e-12 else rl
        r -= (r @ t[i + 1]) * t[i + 1]
        e1[i + 1] = r / np.linalg.norm(r)
    return e1, np.cross(t, e1)


def sample(tomo: np.ndarray, voxel: float, c, e1, e2, coord, chunk: int = 64) -> np.ndarray:
    """tomo[z, y, x] (voxel size ``voxel``, origin at voxel 0) on c + u e2 + v e1 -> (s, u, v), nan outside."""
    out = np.empty((len(c), len(coord), len(coord)), np.float32)
    U, V = np.meshgrid(coord, coord, indexing="ij")
    for a in range(0, len(c), chunk):
        b = min(a + chunk, len(c))
        P = c[a:b, None, None, :] + U[None, :, :, None] * e2[a:b, None, None, :] + V[None, :, :, None] * e1[a:b, None, None, :]
        idx = np.moveaxis(P[..., ::-1] / voxel, -1, 0).reshape(3, -1)
        out[a:b] = ndi.map_coordinates(tomo, idx, order=1, mode="constant", cval=np.nan).reshape(b - a, len(coord), len(coord))
    return out


@dataclass
class Straightened:
    vol: np.ndarray  # (s, e2, e1), protein positive
    step: float
    centers: np.ndarray
    t: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    beam_local: np.ndarray  # beam in (e1, e2, t) per s
    tilt_local: np.ndarray  # tilt axis in (e1, e2, t) per s
    protein_sign: float
    eq_coverage_deg: float
    geometry: TiltGeometry
    recentering: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return (len(self.centers) - 1) * self.step

    def segments(self, length: float):
        """(k, slice) of non-overlapping segments of ``length`` A."""
        n = int(round(length / self.step))
        return [(k, slice(k * n, (k + 1) * n)) for k in range(self.vol.shape[0] // n)]

    def save(self, stem: str):
        np.save(stem + ".npy", self.vol.astype(np.float32))
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items() if k != "vol"}
        d["geometry"] = asdict(self.geometry)
        json.dump(d, open(stem + ".json", "w"))

    @classmethod
    def load(cls, stem: str) -> "Straightened":
        d = json.load(open(stem + ".json"))
        d["geometry"] = TiltGeometry(**d["geometry"])
        for k in ("centers", "t", "e1", "e2", "beam_local", "tilt_local"):
            d[k] = np.asarray(d[k])
        return cls(vol=np.load(stem + ".npy"), **d)


def cross_section_mask(t, e1, e2, geom: TiltGeometry, kgrid, slack: float = 2.0) -> np.ndarray:
    """Cross-section (equatorial) Fourier directions the tilt series sampled, FFT layout (axes e2, e1)."""
    n = geom.normals()
    d = np.cross(t[None, :], n)
    psi = np.arctan2(d @ e1, d @ e2)
    K0, K1 = kgrid
    phi = np.arctan2(K1, K0)
    diff = (phi[..., None] - psi[None, None, :] + np.pi / 2) % np.pi - np.pi / 2
    m = np.abs(diff).min(-1) < np.radians(slack)
    m[0, 0] = True
    return m


def equatorial_coverage(t, geom: TiltGeometry) -> float:
    """Sampled arc (deg of 180) of the equatorial Fourier plane, median over s."""
    n = geom.normals()
    ta = np.asarray(geom.tilt_axis, float)
    cov = []
    for ti in t[:: max(len(t) // 50, 1)]:
        d = np.cross(ti[None, :], n)
        ok = np.linalg.norm(d, axis=1) > 1e-6
        d = d[ok] / np.linalg.norm(d[ok], axis=1, keepdims=True)
        ref1 = np.cross(ti, ta)
        if np.linalg.norm(ref1) < 1e-6:
            ref1 = np.cross(ti, np.asarray(geom.beam, float))
        ref1 /= np.linalg.norm(ref1)
        ref2 = np.cross(ti, ref1)
        psi = np.sort(np.degrees(np.arctan2(d @ ref2, d @ ref1)) % 180)
        cov.append(180 - np.diff(np.r_[psi, psi[0] + 180]).max() + 1.0)
    return float(np.median(cov))


class Recenterer:
    def __init__(self, family: Family, step: float, coord: np.ndarray, lowpass_res: float = 30.0):
        R = np.hypot(*np.meshgrid(coord, coord, indexing="ij"))
        tpl = family.recenter_profile(R)
        tpl = tpl - tpl.mean()
        self.tpl_f = np.conj(np.fft.fft2(np.fft.ifftshift(tpl)))
        kf = np.fft.fftfreq(len(coord), step)
        self.kgrid = np.meshgrid(kf, kf, indexing="ij")
        self.lowpass = np.exp(-0.5 * (np.hypot(*self.kgrid) * lowpass_res) ** 2 / 0.5)
        self.step, self.n = step, len(coord)

    def offset(self, x, mask, lim_A: float):
        """(du along e2, dv along e1) A of the template center in the mean cross-section x (protein positive)."""
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        x = x - x.mean()
        cc = np.real(np.fft.ifft2(np.fft.fft2(x) * self.tpl_f * mask * self.lowpass))
        c, lim = self.n // 2, int(lim_A / self.step)
        sub = cc[c - lim:c + lim + 1, c - lim:c + lim + 1]
        j, k = np.unravel_index(np.argmax(sub), sub.shape)
        off = []
        for ax, m in ((0, j), (1, k)):
            line = sub[:, k] if ax == 0 else sub[j, :]
            d = 0.0
            if 0 < m < len(line) - 1:
                den = line[m - 1] - 2 * line[m] + line[m + 1]
                d = 0.5 * (line[m - 1] - line[m + 1]) / den if den != 0 else 0.0
            off.append((m - lim + d) * self.step)
        return off[0], off[1]


def protein_sign(vol, coord, band) -> float:
    """+1 if protein is bright: the family's density band against the region outside it."""
    x = np.nanmean(vol, axis=0)
    R = np.hypot(*np.meshgrid(coord, coord, indexing="ij"))
    inside = np.nanmean(x[(R > band[0]) & (R < band[1])])
    outside = np.nanmean(x[(R > band[1] + 20) & (R < coord.max())])
    return 1.0 if inside > outside else -1.0


def straighten(points, tomo: np.ndarray, voxel: float, family: Family, geom: TiltGeometry, step: float = 5.0,
               iterations: int = 2, window: float = 300.0, smooth: float = 300.0, lim_A: float = 100.0,
               normalize: bool = True) -> Straightened:
    """Straighten and recenter one filament (points in A, tomogram coordinates; tomo[z, y, x] with voxel size)."""
    if normalize:
        tomo = (tomo - float(tomo.mean())) / float(tomo.std())
    half = family.in_plane_half_width
    coord = (np.arange(int(round(2 * half / step)) + 1) - int(round(half / step))) * step
    rec = Recenterer(family, step, coord)
    beam = np.asarray(geom.beam, float)
    c, t = curve(points, step)
    e1, e2 = frames(c, t, beam)
    sv = sample(tomo, voxel, c, e1, e2, coord)
    sign = protein_sign(sv, coord, family.profile_band)
    win, stride = int(window / step), int(window / step / 2)
    history = []
    for it in range(iterations + 1):
        sc, du, dv = [], [], []
        for a in range(0, max(len(sv) - win, 0) + 1, stride):
            mid = min(a + win // 2, len(sv) - 1)
            mask = cross_section_mask(t[mid], e1[mid], e2[mid], geom, rec.kgrid)
            u, v = rec.offset(sign * np.nanmean(sv[a:a + win], axis=0), mask, lim_A)
            sc.append((a + win / 2) * step)
            du.append(u)
            dv.append(v)
        du, dv = np.array(du), np.array(dv)
        history.append({"iteration": it, "rms_du_A": float(np.sqrt(np.mean(du**2))), "rms_dv_A": float(np.sqrt(np.mean(dv**2)))})
        if it == iterations:
            break
        s_now = np.arange(len(c)) * step
        U = np.interp(s_now, sc, ndi.gaussian_filter1d(du, smooth / window * 2, mode="nearest"))
        V = np.interp(s_now, sc, ndi.gaussian_filter1d(dv, smooth / window * 2, mode="nearest"))
        c, t = curve(c + U[:, None] * e2 + V[:, None] * e1, step)
        e1, e2 = frames(c, t, beam)
        sv = sample(tomo, voxel, c, e1, e2, coord)

    def local(v):
        v = np.asarray(v, float)
        return np.stack([e1 @ v, e2 @ v, t @ v], 1)

    vol = np.nan_to_num(sign * sv)
    return Straightened(vol=vol.astype(np.float32), step=step, centers=c, t=t, e1=e1, e2=e2,
                        beam_local=local(geom.beam), tilt_local=local(geom.tilt_axis), protein_sign=sign,
                        eq_coverage_deg=equatorial_coverage(t, geom), geometry=geom, recentering=history,
                        meta={"nan_fraction": float(np.mean(~np.isfinite(sv)))})
