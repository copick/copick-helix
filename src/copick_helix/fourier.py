"""Fourier tools on straightened filament volumes.

Volumes are ``vol[z, y, x]`` with z along the filament (t), y = e2, x = e1 (about the beam), the axis at the in-plane
center index, and isotropic sampling ``step`` (A).

Transformations of a filament's Fourier–Bessel coefficients c_n (both g(r, n, Z) and F(R, n) on a layer line):
rotation by phi0 about the axis: c_n -> c_n exp(-i n phi0); shift by z0: c_n -> c_n exp(-2 pi i Z z0);
polarity flip (180 deg about x): c_n -> (-1)^n conj(c_n) on the layer-line planes, conj(g) for g(r, n, Z);
mirror (hand): c_n -> c_{-n}.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def tukey(n: int, alpha: float = 0.3) -> np.ndarray:
    w = np.ones(n)
    m = int(alpha * n / 2)
    if m > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
        w[:m], w[-m:] = ramp, ramp[::-1]
    return w


def lowpass(vol: np.ndarray, step: float, res: float | None, edge: float = 0.15) -> np.ndarray:
    """Raised-cosine low-pass to ``res`` A."""
    if res is None:
        return vol
    F = np.fft.rfftn(vol)
    kz = np.fft.fftfreq(vol.shape[0], step)[:, None, None]
    ky = np.fft.fftfreq(vol.shape[1], step)[None, :, None]
    kx = np.fft.rfftfreq(vol.shape[2], step)[None, None, :]
    w = np.clip((1 - np.sqrt(kz**2 + ky**2 + kx**2) * res) / edge, 0, 1)
    return np.fft.irfftn(F * (0.5 - 0.5 * np.cos(np.pi * w)), s=vol.shape).astype(np.float32)


def measured_mask_3d(shape, step, beam, tilt_axis, half_wedge, lp=None, hp=None) -> np.ndarray:
    """rfft-layout mask for a [z=t, y=e2, x=e1] volume: k within +-half_wedge of the specimen plane about the tilt
    axis (both given in local (e1, e2, t) components), optionally with raised-cosine low-pass ``lp`` and
    high-pass ``hp`` (A). A symmetric half_wedge keeps the tilt-range sign convention out of every result."""
    b = np.asarray(beam, float)
    b = b / np.linalg.norm(b)
    y = np.asarray(tilt_axis, float)
    y = y - b * (y @ b)
    y /= np.linalg.norm(y)
    x = np.cross(y, b)
    kt = np.fft.fftfreq(shape[0], step)[:, None, None]
    k2 = np.fft.fftfreq(shape[1], step)[None, :, None]
    k1 = np.fft.rfftfreq(shape[2], step)[None, None, :]
    kb = k1 * b[0] + k2 * b[1] + kt * b[2]
    kx = k1 * x[0] + k2 * x[1] + kt * x[2]
    ang = (np.degrees(np.arctan2(kb, kx)) + 90) % 180 - 90
    m = (np.abs(ang) <= half_wedge).astype(np.float32)
    k = np.sqrt(kt**2 + k2**2 + k1**2)
    if lp is not None:
        lo = np.clip((1 - k * lp) / 0.15, 0, 1)
        m = m * (0.5 - 0.5 * np.cos(np.pi * lo))
    if hp is not None:
        hi = np.clip((k * hp - 1) / 0.5, 0, 1)
        m = m * (0.5 - 0.5 * np.cos(np.pi * hi))
    return m


def cylindrical(vol, step, rmin, rmax, dr, nphi=256, center=None):
    """rho(r, phi, z) from vol[z, y, x]; phi from x (e1) towards y (e2). Returns (r, phi, cyl[r, phi, z])."""
    nz, ny, nx = vol.shape
    cy, cx = (ny // 2, nx // 2) if center is None else center
    r = np.arange(rmin, rmax + 1e-6, dr)
    phi = np.arange(nphi) * 2 * np.pi / nphi
    out = np.empty((len(r), nphi, nz), np.float32)
    zz = np.arange(nz, dtype=np.float32)
    for i, ri in enumerate(r):
        x = cx + ri / step * np.cos(phi)
        y = cy + ri / step * np.sin(phi)
        coords = np.stack(
            [
                np.broadcast_to(zz[None, :], (nphi, nz)),
                np.broadcast_to(y[:, None], (nphi, nz)),
                np.broadcast_to(x[:, None], (nphi, nz)),
            ],
        )
        out[i] = map_coordinates(vol, coords, order=1, mode="constant")
    return r, phi, out


def bessel_coefficients(cyl, step, window=0.3):
    """g[r, n, Z] = FFT over phi and z of rho(r, phi, z), z Tukey-windowed. Returns (n, Z, g)."""
    w = tukey(cyl.shape[2], window).astype(np.float32)
    g = np.fft.fft(np.fft.fft(cyl * w[None, None, :], axis=1), axis=2) / (cyl.shape[1] * w.sum())
    Z = np.fft.fftfreq(cyl.shape[2], step)
    n = np.fft.fftfreq(cyl.shape[1], 1.0 / cyl.shape[1]).round().astype(int)
    return n, Z, g


class PolarPlanes:
    """Exact layer-line planes F(R, Phi) of segments, on a fixed polar grid, and their measured regions."""

    def __init__(
        self,
        step: float,
        n_inplane: int,
        r_band: tuple[float, float],
        r_out_mask: float,
        npad: int | None = None,
        nphi: int = 180,
        mask_edge: float = 15.0,
    ):
        self.step = step
        self.n_inplane = n_inplane
        self.npad = npad or 2 * n_inplane
        self.kf = np.fft.fftshift(np.fft.fftfreq(self.npad, step))
        dk = 1.0 / (self.npad * step)
        self.R = np.arange(np.ceil(r_band[0] / dk), np.floor(r_band[1] / dk) + 1) * dk
        self.phi = np.arange(nphi) * 2 * np.pi / nphi
        c = (np.arange(n_inplane) - n_inplane // 2) * step
        rr = np.hypot(c[:, None], c[None, :])
        w = np.clip((r_out_mask - rr) / mask_edge, 0, 1)
        self.rmask = (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.float32)

    def plane(self, seg: np.ndarray, Z: float) -> np.ndarray:
        """F(R, Phi) on kz = Z of seg[z, y, x] (exact DFT along z, Tukey-windowed; axis at the in-plane center)."""
        n = seg.shape[0]
        w = tukey(n)
        s = np.arange(n) * self.step
        ph = (w * np.exp(-2j * np.pi * Z * s)).astype(np.complex64)
        plane = np.tensordot(ph, seg * self.rmask[None], axes=(0, 0))
        pad = np.zeros((self.npad, self.npad), np.complex64)
        o = (self.npad - plane.shape[0]) // 2
        pad[o : o + plane.shape[0], o : o + plane.shape[1]] = plane
        c = self.n_inplane // 2
        pad = np.roll(pad, (-(o + c), -(o + c)), axis=(0, 1))
        Fp = np.fft.fftshift(np.fft.fft2(pad))
        kx = self.R[:, None] * np.cos(self.phi)[None, :]
        ky = self.R[:, None] * np.sin(self.phi)[None, :]
        iy = (ky - self.kf[0]) / (self.kf[1] - self.kf[0])
        ix = (kx - self.kf[0]) / (self.kf[1] - self.kf[0])
        return map_coordinates(Fp.real, [iy, ix], order=1) + 1j * map_coordinates(Fp.imag, [iy, ix], order=1)

    def measured(self, beam, tilt_axis, Z: float, half_wedge: float) -> np.ndarray:
        """Boolean [R, Phi]: k = R(cos Phi e1 + sin Phi e2) + Z t inside +-half_wedge of the specimen plane."""
        b = np.asarray(beam, float) / np.linalg.norm(beam)
        y = np.asarray(tilt_axis, float)
        y = y - b * (y @ b)
        y /= np.linalg.norm(y)
        x = np.cross(y, b)
        k1 = self.R[:, None] * np.cos(self.phi)[None, :]
        k2 = self.R[:, None] * np.sin(self.phi)[None, :]
        kb = k1 * b[0] + k2 * b[1] + Z * b[2]
        kx = k1 * x[0] + k2 * x[1] + Z * x[2]
        ang = (np.degrees(np.arctan2(kb, kx)) + 90) % 180 - 90
        return np.abs(ang) <= half_wedge
