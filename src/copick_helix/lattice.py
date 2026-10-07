"""Lattice read-outs measured from a filament's own data.

- ``axial_repeat``: the strongest layer line near an expected axial period, from a side projection (along the beam,
  close to the 0-degree view every tilt series samples), parabolically refined, with a signal-to-noise ratio. For
  microtubules the monomer repeat (about 41 A), and with it the rise of the monomer helix.
- ``equator_count``: protofilament number of a microtubule from the angular power at the wall near the equator. Only
  meaningful with enough equatorial coverage (about 85 deg of 180 or more); with less, the wedge leaks order N into
  N +- 2.
"""

from __future__ import annotations

import numpy as np

from .fourier import bessel_coefficients, cylindrical, measured_mask_3d


def axial_repeat(vol: np.ndarray, step: float, k_range: tuple[float, float], half_width_beam: float = 150.0,
                 k_perp_max: float = 0.02, npad: int = 2**15) -> tuple[float, float]:
    """(period A, peak SNR) of the strongest layer line with axial frequency in k_range (1/A).

    vol[s, e2, e1], protein positive. The side view sums |e1| <= half_width_beam; power is summed over
    |k_e2| <= k_perp_max."""
    n_in = vol.shape[2]
    coord = (np.arange(n_in) - n_in // 2) * step
    side = np.nan_to_num(vol)[:, :, np.abs(coord) <= half_width_beam].sum(axis=2)
    side = side - side.mean(0, keepdims=True)
    ke = np.fft.fftfreq(vol.shape[1], step)
    F = np.fft.fft(side, axis=1)[:, np.abs(ke) <= k_perp_max]
    F = np.fft.fft(F * np.hanning(len(side))[:, None], n=npad, axis=0)
    P = (np.abs(F) ** 2).sum(1)
    ks = np.fft.fftfreq(npad, step)
    sel = (ks >= k_range[0]) & (ks <= k_range[1])
    i = np.where(sel)[0][np.argmax(P[sel])]
    den = P[i - 1] - 2 * P[i] + P[i + 1]
    k = ks[i] + (0.5 * (P[i - 1] - P[i + 1]) / den if den != 0 else 0) * (ks[1] - ks[0])
    lo, hi = k_range[0] - 0.002, k_range[1] + 0.002
    bg = (ks >= lo) & (ks <= hi) & (np.abs(ks - k) > 0.0005)
    return float(1 / k), float(P[i] / np.median(P[bg]))


def microtubule_repeat(vol: np.ndarray, step: float) -> tuple[float, float]:
    """Monomer repeat (A) and SNR from the 4 nm layer line (search 38.5-43.5 A)."""
    return axial_repeat(vol, step, (0.023, 0.026))


def equator_count(vol: np.ndarray, step: float, beam, tilt_axis, orders=range(9, 18), wall=(80.0, 145.0),
                  seg_len: float = 1250.0) -> dict:
    """Angular power share per order n at the wall near the equator (|Z| <= 1/1500 A), whole filament.

    Returns {"n_best": int, "share": {n: fraction}}. vol[s, e2, e1], protein positive; beam / tilt axis in local
    components (the median along the filament)."""
    n_seg = int(round(seg_len / step))
    n_s = (vol.shape[0] // n_seg) * n_seg
    v = vol[:n_s] - vol[:n_s].mean()
    m = measured_mask_3d(v.shape, step, beam, tilt_axis, 90.0, 20.0, 400.0)  # band limits only: the data's own coverage
    v = np.fft.irfftn(np.fft.rfftn(v) * m, s=v.shape).astype(np.float32)
    c = v.shape[1] // 2
    r, _, cyl = cylindrical(v, step, 60.0, 160.0, 2.5, 256, centre=(c, c))
    n, Z, g = bessel_coefficients(cyl, step)
    zsel = np.abs(Z) <= 1 / 1500.0
    rsel = (r >= wall[0]) & (r <= wall[1])
    pw = {k: float((r[rsel, None] * np.abs(g[rsel][:, np.where(n == k)[0][0]][:, zsel]) ** 2).sum()) for k in orders}
    tot = sum(pw.values())
    return {"n_best": int(max(pw, key=pw.get)), "share": {k: v / tot for k, v in pw.items()}}
