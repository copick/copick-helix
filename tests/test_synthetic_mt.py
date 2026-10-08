"""Synthetic microtubules: support, phase invariants, iterative reference, and an apolar control (fast; needs gemmi + 6DPV)."""

import os

import numpy as np
import pytest
from scipy.ndimage import rotate

from copick_helix import invariants, iterative
from copick_helix.families import microtubule
from copick_helix.fourier import PolarPlanes, measured_mask_3d
from copick_helix.helix import Term
from copick_helix.invariants import SegmentGeometry

gemmi = pytest.importorskip("gemmi")
LOCAL_6DPV = "/hpc/projects/group.czii/utz.ermel/mt-polarity/models/6DPV.cif"


def test_microtubule_support_orders():
    fam = microtubule()
    sym = fam.symmetry(monomer_repeat=40.836)
    assert abs(sym.Z(Term(13, -1))) < 1e-12  # equator
    assert np.isclose(sym.Z(Term(3, 0)), 1 / 40.836)
    tol = 1e-6
    assert set(sym.plane_orders(Term(13, -1), 30, tol)) == {-26, -13, 0, 13, 26}
    assert set(sym.plane_orders(Term(3, 0), 30, tol)) == {-23, -10, 3, 16, 29}


@pytest.fixture(scope="module")
def synthetic():
    from copick_helix.models import MicrotubuleLattice, on_grid

    lat = MicrotubuleLattice(LOCAL_6DPV if os.path.exists(LOCAL_6DPV) else None)
    plus = on_grid(*lat.atoms(13, 3, 2500.0), 5.0, 81, 2500.0)
    minus = on_grid(*lat.atoms(13, 3, 2500.0, flip=True), 5.0, 81, 2500.0)
    rng = np.random.default_rng(3)
    fils, truth = {}, {}
    for i in range(8):
        pol = "plus" if i % 2 == 0 else "minus"
        v = rotate(plus if pol == "plus" else minus, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
        v = np.roll(v, int(rng.integers(0, 16)), axis=0)
        psi = [10, 40, 70, 25][i % 4]  # angle between filament and tilt axis
        tilt = np.array([0.0, np.sin(np.radians(psi)), np.cos(np.radians(psi))])
        beam = np.array([1.0, 0.0, 0.0])
        v = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, 5.0, beam, tilt, 60.0, 10.0), s=v.shape)
        v = (v + rng.normal(0, v[:, 25:56, 25:56].std() / np.sqrt(0.05), v.shape)).astype(np.float32)
        fils[f"f{i}"] = (v, SegmentGeometry(beam, tilt))
        truth[f"f{i}"] = pol
    return lat, plus, fils, truth


def _segments(v, n=250):
    return [v[k * n:(k + 1) * n] - v[k * n:(k + 1) * n].mean() for k in range(v.shape[0] // n)]


def test_invariants_split_and_label(synthetic):
    lat, plus, fils, truth = synthetic
    fam = microtubule()
    cfg = invariants.InvariantConfig(terms=fam.terms, triples=fam.triples, r_out=fam.r_out, r_mask=fam.r_mask)
    planes = PolarPlanes(5.0, 81, cfg.r_band, cfg.r_mask)
    sym = fam.symmetry(monomer_repeat=lat.a)
    fits = {k: [invariants.fit_segment(s, g, 5.0, sym, cfg, planes) for s in _segments(v)] for k, (v, g) in fits_items(fils)}
    mcfg = invariants.InvariantConfig(terms=fam.terms, triples=fam.triples, r_out=fam.r_out, r_mask=fam.r_mask, half_wedge=90.0)
    model = [invariants.fit_segment(_segments(plus)[0], SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                                 5.0, sym, mcfg, planes)]
    table, summ = invariants.assign(fits, cfg, model)
    assert all(table.set_index("filament").call[k] == truth[k] for k in truth)
    assert summ["eigen_gap"] > 5


def fits_items(fils):
    return fils.items()


def test_iterative_converges_and_labels(synthetic):
    lat, plus, fils, truth = synthetic
    fam = microtubule()
    cfg = iterative.IterativeConfig(support=fam.support(z_max=1 / 20.0, n_max=56), random_starts=4, seed_starts=2)
    gby = {}
    r = n = Z = None
    for k, (v, g) in fils.items():
        gl = []
        for s in _segments(v):
            r, n, Z, gg = iterative.segment_g(s, g, 5.0, cfg)
            gl.append(gg)
        gby[k] = np.stack(gl)
    _, _, _, gm = iterative.segment_g(_segments(plus)[0], SegmentGeometry(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
                                      5.0, iterative.IterativeConfig(half_wedge=90.0))
    res = iterative.assign(gby, r, n, Z, cfg, model_plus_g=gm)
    calls = res.calls.set_index("filament").call
    assert all(calls[k] == truth[k] for k in truth)
    assert (res.runs[res.runs.start.str.startswith(("random", "seed"))].agree_with_consensus == 1.0).all()


def test_apolar_control_has_no_polarity_structure(synthetic):
    """A non-polar lattice (plus + minus) must not produce polarity groups: no eigenvalue gap, and the consensus
    reference no more polar than random assignments."""
    lat, plus, fils, truth = synthetic
    minus = np.flip(np.flip(plus, 0), 1)  # 180 deg about x on the grid (axis at the center)
    apolar = 0.5 * (plus + minus)
    rng = np.random.default_rng(5)
    fam = microtubule()
    cfg = invariants.InvariantConfig(terms=fam.terms, triples=fam.triples, r_out=fam.r_out, r_mask=fam.r_mask)
    planes = PolarPlanes(5.0, 81, cfg.r_band, cfg.r_mask)
    sym = fam.symmetry(monomer_repeat=lat.a)
    fits = {}
    for i in range(8):
        v = rotate(apolar, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
        v = (v + rng.normal(0, v[:, 25:56, 25:56].std() / np.sqrt(0.05), v.shape)).astype(np.float32)
        geom = SegmentGeometry(np.array([1.0, 0, 0]), np.array([0.0, 0.3, 0.95]))
        fits[f"a{i}"] = [invariants.fit_segment(s, geom, 5.0, sym, cfg, planes) for s in _segments(v)]
    _, summ = invariants.assign(fits, cfg)
    assert summ["eigen_gap"] < 3
