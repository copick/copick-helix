"""Synthetic F-actin (6DJO) through the term route: per filament a random rotation about and shift along the axis, an
in-plane axis offset, random polarity, the missing wedge and noise. The relative polarity must be recovered and named
by the model, an apolar control (each filament the sum of both polarities) must show no polarity, and particles placed
from the registrations must agree across filaments of both polarities, while deliberate convention errors break
that agreement."""

import os

import numpy as np
import pytest
from scipy.ndimage import rotate, shift as nd_shift

from copick_helix import bands
from copick_helix.families import get_family
from copick_helix.fourier import measured_mask_3d
from copick_helix.geometry import Straightened, TiltGeometry
from copick_helix.pipeline import term_registration_particles, term_segments
from copick_helix.registration import Registration, extract, term_particles

LOCAL = "/hpc/projects/group.czii/utz.ermel/mt-polarity/actin/models/6DJO.cif"
if os.path.exists(LOCAL):
    os.environ.setdefault("COPICK_HELIX_6DJO", LOCAL)
STEP, N_IN, LENGTH = 5.0, 49, 3800.0


def _straightened(vol, beam, tilt):
    n = vol.shape[0]
    c = np.stack([np.full(n, (N_IN // 2) * STEP), np.full(n, (N_IN // 2) * STEP), np.arange(n) * STEP], 1)
    eye = np.eye(3)
    return Straightened(vol=vol, step=STEP, centres=c, t=np.tile(eye[2], (n, 1)), e1=np.tile(eye[0], (n, 1)),
                        e2=np.tile(eye[1], (n, 1)), beam_local=np.tile(beam, (n, 1)), tilt_local=np.tile(tilt, (n, 1)),
                        protein_sign=1.0, eq_coverage_deg=100.0, geometry=TiltGeometry())


def _cc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@pytest.fixture(scope="module")
def data():
    fam = get_family("actin")
    plus = fam.term_model(STEP, N_IN, LENGTH)[0]  # barbed (plus) end at -z
    rng = np.random.default_rng(3)
    fils, clean, truth, apolar = {}, {}, {}, {}
    beam, tilt = np.array([1.0, 0, 0]), np.array([0.0, np.sin(np.radians(25)), np.cos(np.radians(25))])
    mask = measured_mask_3d(plus.shape, STEP, beam, tilt, 60.0, 12.0)
    for i in range(10):
        pol = "plus" if i % 2 == 0 else "minus"
        v = rotate(plus, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
        v = np.roll(v, int(rng.integers(0, 6)), axis=0)
        if pol == "minus":
            v = v[::-1, ::-1, :].copy()  # 180 deg about x through the volume centre
        off = rng.uniform(-4, 4, 2)  # axis offset (A) along (e1, e2)
        v = nd_shift(v, (0, off[1] / STEP, off[0] / STEP), order=1)
        sym = 0.5 * (v + v[::-1, ::-1, :])  # apolar: both polarities superposed
        for store, vol in ((fils, v), (apolar, sym)):
            noisy = np.fft.irfftn(np.fft.rfftn(vol) * mask, s=vol.shape)
            noisy = noisy + rng.normal(0, plus[:, 18:31, 18:31].std() / np.sqrt(0.5), vol.shape)
            store[f"f{i}"] = _straightened(noisy.astype(np.float32), beam, tilt)
        clean[f"f{i}"] = v.astype(np.float32)
        truth[f"f{i}"] = pol
    cfg = bands.TermConfig(r_out=fam.term_r_out, starts=3)
    res = bands.analyse(term_segments(fils, fam.segment_length), fam, cfg, workers=1, log=lambda *a: None)
    res_apolar = bands.analyse(term_segments(apolar, fam.segment_length), fam, cfg, workers=1, log=lambda *a: None)
    return fam, fils, clean, truth, res, res_apolar


def test_polarity_recovered_and_named(data):
    fam, fils, clean, truth, res, _ = data
    calls = dict(zip(res.table.filament, res.table.call))
    assert calls == truth, (calls, truth)
    assert res.summary["polarity_detected"], res.summary
    assert (res.table.z >= 3).sum() >= 8


def test_apolar_control_shows_no_polarity(data):
    *_, res_apolar = data
    assert not res_apolar.summary["polarity_detected"], res_apolar.summary
    assert not res_apolar.table.seed.any()


def test_registered_particles_agree_and_conventions_matter(data):
    fam, fils, clean, truth, res, _ = data
    L = fam.segment_length
    screw = fam.screw({"rise": res.summary["rise"], "twist": res.summary["twist"]})

    def median_cc(mod=lambda f, r, s, o: (f, r, s, o), om_sign=1.0):
        subs = []
        for row in res.segments.itertuples():
            st = fils[row.filament]
            n = int(round(L / st.step))
            f, r, s, o = mod(bool(row.flip), row.roll_deg, row.shift_A, (row.offset_e1_A, row.offset_e2_A))
            reg = Registration(int(row.segment), f, r, s, row.score)
            s_c = (reg.segment * n + n // 2) * st.step
            pos, rots, s_arc = term_particles(st, s_c, reg, o, (screw[0], om_sign * screw[1]), fam.plus_at_minus_z,
                                              L / 2)
            k = int(np.argmin(np.abs(s_arc - (s_c + 85.0))))  # about three subunits off-centre: the screw acts
            subs.append(extract(clean[row.filament], STEP, np.zeros(3), pos[k], rots[k], (150.0, 50.0, 50.0)))
        return float(np.median([_cc(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]))

    good = median_cc()
    assert good > 0.9, good
    for mod, om in ((lambda f, r, s, o: (f, -r, s, o), 1.0), (lambda f, r, s, o: (False, r, s, o), 1.0),
                    (lambda f, r, s, o: (f, r, -s, o), 1.0), (lambda f, r, s, o: (f, r, s, o), -1.0)):
        assert median_cc(mod, om) < good - 0.15
    reg = term_registration_particles(fam, fils, res, L)
    assert all(len(v[0]) == len(res.segments[res.segments.filament == k]) for k, v in reg.items())
