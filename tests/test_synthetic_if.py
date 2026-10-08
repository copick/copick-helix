"""Synthetic vimentin filaments (8RVE geometry): the invariants split polarities, and an exactly apolar control
(filament + its flipped copy) shows no polarity structure."""

import os

import numpy as np
import pytest
from scipy.ndimage import rotate

from copick_helix import invariants
from copick_helix.families import intermediate_filament
from copick_helix.fourier import PolarPlanes, measured_mask_3d
from copick_helix.invariants import SegmentGeometry
from copick_helix.models import HelicalModel, on_grid

LOCAL_8RVE = "/hpc/projects/group.czii/utz.ermel/mt-polarity/if/data/8RVE.cif"


@pytest.fixture(scope="module")
def vols():
    fam = intermediate_filament()
    m = HelicalModel("8RVE", 42.461, 73.7308, cif=LOCAL_8RVE if os.path.exists(LOCAL_8RVE) else None)
    plus = on_grid(*m.atoms(2500.0), 5.0, 81, 2500.0)
    minus = on_grid(*m.atoms(2500.0, flip=True), 5.0, 81, 2500.0)
    return fam, plus, minus


def _fits(fam, vol_by_name, snr, rng):
    cfg = invariants.InvariantConfig(
        terms=fam.terms,
        triples=fam.triples,
        r_out=fam.r_out,
        r_mask=fam.r_mask,
        plane_tol_frac=fam.plane_tol_bins,
    )
    planes = PolarPlanes(5.0, 81, cfg.r_band, cfg.r_mask)
    sym = fam.symmetry(**fam.reference_params)
    out = {}
    for name, v in vol_by_name.items():
        v = rotate(v, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
        v = np.roll(v, int(rng.integers(0, 40)), axis=0)
        tilt = np.array([0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])
        beam = np.array([1.0, 0.0, 0.0])
        v = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, 5.0, beam, tilt, 60.0, 10.0), s=v.shape)
        v = (v + rng.normal(0, v[:, 30:51, 30:51].std() / np.sqrt(snr), v.shape)).astype(np.float32)
        segs = [v[k * 250 : (k + 1) * 250] - v[k * 250 : (k + 1) * 250].mean() for k in range(2)]
        out[name] = [invariants.fit_segment(s, SegmentGeometry(beam, tilt), 5.0, sym, cfg, planes) for s in segs]
    return out, cfg


def test_if_polarity_split(vols):
    fam, plus, minus = vols
    rng = np.random.default_rng(11)
    names = {f"f{i}": (plus if i % 2 == 0 else minus) for i in range(8)}
    fits, cfg = _fits(fam, names, 0.1, rng)
    table, summ = invariants.assign(fits, cfg)
    groups = table.set_index("filament").group
    assert len({groups[f"f{i}"] for i in range(0, 8, 2)}) == 1
    assert len({groups[f"f{i}"] for i in range(1, 8, 2)}) == 1
    assert groups["f0"] != groups["f1"]
    assert summ["eigen_gap"] > 3


def test_if_apolar_control(vols):
    fam, plus, minus = vols
    rng = np.random.default_rng(12)
    apolar = 0.5 * (plus + minus)
    fits, cfg = _fits(fam, {f"a{i}": apolar for i in range(8)}, 0.1, rng)
    _, summ = invariants.assign(fits, cfg)
    assert summ["eigen_gap"] < 3
