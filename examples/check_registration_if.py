"""Sensitivity check of the registration conventions on synthetic vimentin filaments (helical screw)."""
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import rotate

sys.path.insert(0, "tests")
import test_registration as T  # noqa: E402

from copick_helix import pipeline  # noqa: E402
from copick_helix.families import intermediate_filament  # noqa: E402
from copick_helix.fourier import measured_mask_3d  # noqa: E402
from copick_helix.models import HelicalModel, on_grid  # noqa: E402
from copick_helix.registration import Registration, extract, lattice_particles  # noqa: E402

fam = intermediate_filament()
m = HelicalModel("8RVE", 42.461, 73.7308, cif="/hpc/projects/group.czii/utz.ermel/mt-polarity/if/data/8RVE.cif")
plus = on_grid(*m.atoms(2500.0), 5.0, 81, 2500.0)
rng = np.random.default_rng(31)
fils, clean = {}, {}
for i in range(6):
    v = rotate(plus, rng.uniform(0, 360), axes=(1, 2), reshape=False, order=1)
    v = np.roll(v, int(rng.integers(0, 40)), axis=0)
    if i % 2:
        v = v[::-1, ::-1, :].copy()
    beam, tilt = np.array([1.0, 0, 0]), np.array([0.0, 0.5, np.sqrt(0.75)])
    noisy = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, 5.0, beam, tilt, 60.0, 10.0), s=v.shape)
    noisy = (noisy + rng.normal(0, v[:, 30:51, 30:51].std() / np.sqrt(0.3), v.shape)).astype(np.float32)
    fils[f"f{i}"], clean[f"f{i}"] = T._straightened(noisy, beam, tilt), v.astype(np.float32)
params = pd.DataFrame({"rise": 42.461, "twist": 73.7308}, index=list(fils))
res = pipeline.run_iterative(fam, fils, params, model=(on_grid(*m.atoms(1250.0), 5.0, 81, 1250.0), {}),
                             random_starts=4, seed_starts=2)
print("calls", res.calls.call.tolist(), "(truth alternates plus/minus)")
screw = fam.screw(fam.reference_params)


def med_cc(mod, omega_sign=1.0):
    subs = []
    for row in res.segments.itertuples():
        reg = Registration(row.segment, *mod(row.flip, row.roll_deg, row.shift_A), row.score)
        pos, rots, _ = lattice_particles(fils[row.filament], reg, fam.segment_length, 1.0,
                                         (screw[0], omega_sign * screw[1]), fam.plus_at_minus_z, center_only=True)
        subs.append(extract(clean[row.filament], 5.0, np.zeros(3), pos[0], rots[0], (200.0, 80.0, 80.0)))
    cc = [T._cc(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    return np.median(cc)


for lab, mod, om in (("correct", lambda f, r, s: (f, r, s), 1.0), ("roll sign inverted", lambda f, r, s: (f, -r, s), 1.0),
                     ("flip ignored", lambda f, r, s: (False, r, s), 1.0),
                     ("shift sign inverted", lambda f, r, s: (f, r, -s), 1.0), ("screw sign inverted", lambda f, r, s: (f, r, s), -1.0)):
    print(f"{lab:20s} median pairwise CC {med_cc(mod, om):.3f}")
