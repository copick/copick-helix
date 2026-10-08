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
    r, _, cyl = cylindrical(v, step, 60.0, 160.0, 2.5, 256, center=(c, c))
    n, Z, g = bessel_coefficients(cyl, step)
    zsel = np.abs(Z) <= 1 / 1500.0
    rsel = (r >= wall[0]) & (r <= wall[1])
    pw = {k: float((r[rsel, None] * np.abs(g[rsel][:, np.where(n == k)[0][0]][:, zsel]) ** 2).sum()) for k in orders}
    tot = sum(pw.values())
    return {"n_best": int(max(pw, key=pw.get)), "share": {k: v / tot for k, v in pw.items()}}


def layer_scan(vol: np.ndarray, step: float, order: int, z_pred: float, frac: float = 0.15, r_max: float = 80.0,
               n_z: int = 241) -> tuple[np.ndarray, np.ndarray]:
    """Power of Bessel order ``order`` against Z in [z_pred (1 - frac), z_pred (1 + frac)] for a whole straightened
    filament (exact DFT along z, Hann window, radially weighted). Returns (Z, P)."""
    c = vol.shape[1] // 2
    r, _, cyl = cylindrical(vol - vol.mean(), step, 4.0, r_max, step / 2, 256, center=(c, c))
    a = np.fft.fft(cyl, axis=1)
    nidx = np.fft.fftfreq(256, 1 / 256).round().astype(int)
    zc = np.arange(cyl.shape[2]) * step
    an = a[:, np.where(nidx == order)[0][0], :] * np.hanning(len(zc))[None, :]
    Zs = np.linspace(z_pred * (1 - frac), z_pred * (1 + frac), n_z)
    P = np.sum(r[:, None] * np.abs(an @ np.exp(-2j * np.pi * np.outer(zc, Zs))) ** 2, axis=0)
    return Zs, P


def rise_twist_from_lines(vol: np.ndarray, step: float, symmetry, line_a, line_b, r_max: float = 80.0,
                          frac: float = 0.15) -> dict:
    """Rise and twist of a helical filament from the measured positions of two layer lines (Terms with different n).

    With Z(n, m) = (m - n * twist / 360) / rise, two lines give a 2 x 2 linear system in (1/rise, twist/360/rise).
    The predicted positions come from ``symmetry`` (the family's prior). Also returns each line's peak / median power
    over the scan (a per-filament lattice signal-to-noise)."""
    out, zs = {}, []
    for key, t in (("a", line_a), ("b", line_b)):
        Z, P = layer_scan(vol, step, t.n, symmetry.Z(t), frac=frac, r_max=r_max)
        zs.append(Z[np.argmax(P)])
        out[f"snr_{t.n}_{t.m}"] = float(P.max() / np.median(P))
    A = np.array([[line_a.m, -line_a.n], [line_b.m, -line_b.n]], float)
    inv_h, tau_h = np.linalg.solve(A, np.array(zs))
    out["rise"] = float(1.0 / inv_h)
    out["twist"] = float(360.0 * tau_h / inv_h)
    return out


def lattice_enrichment(power_nz: np.ndarray, n: np.ndarray, Z: np.ndarray, symmetry, terms, near=(3, 12)) -> dict:
    """Pooled lattice gate: power at each term's (n, nearest Z bin) over the median power of the same n at Z bins
    ``near`` bins away. power_nz: summed radially weighted |g|^2 over segments, on the (n, Z) FFT grid. Values near 1
    mean no helical order is detectable; compare with phase-scrambled decoys."""
    dz = abs(Z[1] - Z[0])
    out = {}
    for t in terms:
        i = np.where(n == t.n)[0]
        if not len(i):
            continue
        j = int(np.argmin(np.abs(Z - symmetry.Z(t))))
        nb = [k for k in range(len(Z)) if near[0] <= abs(Z[k] - Z[j]) / dz <= near[1]]
        out[(t.n, t.m)] = float(power_nz[i[0], j] / np.median(power_nz[i[0], nb]))
    return out
