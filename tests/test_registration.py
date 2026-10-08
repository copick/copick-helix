"""Round trip: synthetic microtubules with known polarity, rotation and shift -> iterative reference -> registration
picks. Particles extracted at the picks must agree across filaments (consistent starting orientations), and the
exported frame must have the plus end at +z."""

import os

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import rotate

from copick_helix import pipeline
from copick_helix.families import microtubule
from copick_helix.fourier import bessel_coefficients, cylindrical, measured_mask_3d
from copick_helix.geometry import Straightened, TiltGeometry
from copick_helix.models import MicrotubuleLattice, on_grid
from copick_helix.registration import FLIP, Registration, extract, lattice_particles

LOCAL_6DPV = "/hpc/projects/group.czii/utz.ermel/mt-polarity/models/6DPV.cif"
STEP = 5.0


def _straightened(vol, beam, tilt):
    n = vol.shape[0]
    c = np.stack([np.full(n, 40 * STEP), np.full(n, 40 * STEP), np.arange(n) * STEP], 1)
    eye = np.eye(3)
    return Straightened(vol=vol, step=STEP, centers=c, t=np.tile(eye[2], (n, 1)), e1=np.tile(eye[0], (n, 1)),
                        e2=np.tile(eye[1], (n, 1)), beam_local=np.tile(beam, (n, 1)), tilt_local=np.tile(tilt, (n, 1)),
                        protein_sign=1.0, eq_coverage_deg=100.0, geometry=TiltGeometry())


def _cc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _cc_max_bessel(a, b):
    """Best correlation over rotation about and shift along z (cylindrical Bessel space)."""
    ga = bessel_coefficients(cylindrical(a, STEP, 60, 165, 2.5, 128)[2], STEP)[2]
    r, _, cb = cylindrical(b, STEP, 60, 165, 2.5, 128)
    gb = bessel_coefficients(cb, STEP)[2]
    ga[:, 0, 0] = gb[:, 0, 0] = 0
    cross = np.tensordot(r, ga * np.conj(gb), axes=(0, 0))
    na = np.sqrt(np.tensordot(r, np.abs(ga) ** 2, axes=(0, 0)).sum())
    nb = np.sqrt(np.tensordot(r, np.abs(gb) ** 2, axes=(0, 0)).sum())
    return float((np.real(np.fft.ifft2(cross)) * cross.size / (na * nb)).max())


@pytest.fixture(scope="module")
def run():
    fam = microtubule()
    lat = MicrotubuleLattice(LOCAL_6DPV if os.path.exists(LOCAL_6DPV) else None)
    plus = on_grid(*lat.atoms(13, 3, 2500.0), STEP, 81, 2500.0)
    rng = np.random.default_rng(21)
    fils, clean, truth = {}, {}, {}
    for i in range(6):
        pol = "plus" if i % 2 == 0 else "minus"
        v = rotate(plus, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
        v = np.roll(v, int(rng.integers(0, 16)), axis=0)
        if pol == "minus":
            v = v[::-1, ::-1, :].copy()  # 180 deg about x (axis at in-plane index 40)
        beam, tilt = np.array([1.0, 0, 0]), np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])
        noisy = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, STEP, beam, tilt, 60.0, 10.0), s=v.shape)
        noisy = (noisy + rng.normal(0, v[:, 25:56, 25:56].std() / np.sqrt(0.3), v.shape)).astype(np.float32)
        fils[f"f{i}"] = _straightened(noisy, beam, tilt)
        clean[f"f{i}"] = v.astype(np.float32)
        truth[f"f{i}"] = pol
    params = pd.DataFrame({"monomer_repeat": lat.a}, index=list(fils))
    res = pipeline.run_iterative(fam, fils, params, model=(on_grid(*lat.atoms(13, 3, 1250.0), STEP, 81, 1250.0),
                                                           {"monomer_repeat": lat.a}), random_starts=4, seed_starts=2)
    return fam, lat, plus, fils, clean, truth, res


def _picks(fam, lat, fils, res, center_only=True):
    out = []
    screw = fam.screw({"monomer_repeat": lat.a})
    for row in res.segments.itertuples():
        reg = Registration(row.segment, row.flip, row.roll_deg, row.shift_A, row.score)
        pos, rots, _ = lattice_particles(fils[row.filament], reg, fam.segment_length, 1.0, screw,
                                         fam.plus_at_minus_z, center_only=center_only)
        out += [(row.filament, p, R) for p, R in zip(pos, rots)]
    return out


def test_calls(run):
    fam, lat, plus, fils, clean, truth, res = run
    calls = res.calls.set_index("filament").call
    assert all(calls[k] == truth[k] for k in truth)


def test_registered_particles_agree_across_filaments(run):
    fam, lat, plus, fils, clean, truth, res = run
    picks = _picks(fam, lat, fils, res)
    subs = [extract(clean[f], STEP, np.zeros(3), p, R, (300.0, 150.0, 150.0)) for f, p, R in picks]
    ccs = [_cc(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    assert np.median(ccs) > 0.8, np.median(ccs)
    assert min(ccs) > 0.6, min(ccs)


def test_exported_frame_has_plus_at_plus_z(run):
    fam, lat, plus, fils, clean, truth, res = run
    picks = _picks(fam, lat, fils, res)
    avg = np.mean([extract(clean[f], STEP, np.zeros(3), p, R, (300.0, 200.0, 200.0)) for f, p, R in picks], axis=0)
    model_plus_up = on_grid(*lat.atoms(13, 3, 605.0, flip=True), STEP, 81, 605.0)  # plus end at +z
    n = min(len(avg), len(model_plus_up))
    up, down = model_plus_up[:n], model_plus_up[:n][::-1, ::-1, :]
    assert _cc_max_bessel(avg[:n], up) > _cc_max_bessel(avg[:n], down) + 0.05


def test_helical_registration_round_trip():
    """Vimentin (helical screw 42.461 A / 73.73 deg): particles registered from both polarities agree, and every
    deliberate convention error (roll sign, flip, shift sign, screw sign) breaks that agreement."""
    from copick_helix.families import intermediate_filament
    from copick_helix.models import HelicalModel

    local = "/hpc/projects/group.czii/utz.ermel/mt-polarity/if/data/8RVE.cif"
    fam = intermediate_filament()
    m = HelicalModel("8RVE", 42.461, 73.7308, cif=local if os.path.exists(local) else None)
    plus = on_grid(*m.atoms(2500.0), STEP, 81, 2500.0)
    rng = np.random.default_rng(31)
    fils, clean = {}, {}
    for i in range(6):
        v = np.roll(rotate(plus, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1), int(rng.integers(0, 40)), axis=0)
        if i % 2:
            v = v[::-1, ::-1, :].copy()
        beam, tilt = np.array([1.0, 0, 0]), np.array([0.0, 0.5, np.sqrt(0.75)])
        noisy = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, STEP, beam, tilt, 60.0, 10.0), s=v.shape)
        noisy = (noisy + rng.normal(0, v[:, 30:51, 30:51].std() / np.sqrt(0.3), v.shape)).astype(np.float32)
        fils[f"f{i}"], clean[f"f{i}"] = _straightened(noisy, beam, tilt), v.astype(np.float32)
    params = pd.DataFrame({"rise": 42.461, "twist": 73.7308}, index=list(fils))
    res = pipeline.run_iterative(fam, fils, params, model=(on_grid(*m.atoms(1250.0), STEP, 81, 1250.0), {}),
                                 random_starts=4, seed_starts=2)
    assert res.calls.call.tolist() == ["plus", "minus"] * 3
    P, om = fam.screw(fam.reference_params)

    def median_cc(mod, om_sign=1.0):
        subs = []
        for row in res.segments.itertuples():
            reg = Registration(row.segment, *mod(row.flip, row.roll_deg, row.shift_A), row.score)
            pos, rots, _ = lattice_particles(fils[row.filament], reg, fam.segment_length, 1.0, (P, om_sign * om),
                                             fam.plus_at_minus_z, center_only=True)
            subs.append(extract(clean[row.filament], STEP, np.zeros(3), pos[0], rots[0], (200.0, 80.0, 80.0)))
        return np.median([_cc(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))])

    good = median_cc(lambda f, r, s: (f, r, s))
    assert good > 0.95, good
    for mod, om_sign in ((lambda f, r, s: (f, -r, s), 1.0), (lambda f, r, s: (False, r, s), 1.0),
                         (lambda f, r, s: (f, r, -s), 1.0), (lambda f, r, s: (f, r, s), -1.0)):
        assert median_cc(mod, om_sign) < good - 0.2
