"""Estimated vs true registrations on synthetic vimentin filaments, modulo the helical screw."""
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

fam = intermediate_filament()
m = HelicalModel("8RVE", 42.461, 73.7308, cif="/hpc/projects/group.czii/utz.ermel/mt-polarity/if/data/8RVE.cif")
plus = on_grid(*m.atoms(2500.0), 5.0, 81, 2500.0)
rng = np.random.default_rng(31)
fils, truth = {}, {}
snr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
for i in range(6):
    th, sh = rng.uniform(0, 360), int(rng.integers(0, 40))
    v = rotate(plus, th, axes=(1, 2), reshape=False, order=1)
    v = np.roll(v, sh, axis=0)
    if i % 2:
        v = v[::-1, ::-1, :].copy()
    beam, tilt = np.array([1.0, 0, 0]), np.array([0.0, 0.5, np.sqrt(0.75)])
    noisy = np.fft.irfftn(np.fft.rfftn(v) * measured_mask_3d(v.shape, 5.0, beam, tilt, 60.0, 10.0), s=v.shape)
    noisy = (noisy + rng.normal(0, v[:, 30:51, 30:51].std() / np.sqrt(snr), v.shape)).astype(np.float32)
    fils[f"f{i}"] = T._straightened(noisy, beam, tilt)
    truth[f"f{i}"] = (th, sh * 5.0, bool(i % 2))
params = pd.DataFrame({"rise": 42.461, "twist": 73.7308}, index=list(fils))
res = pipeline.run_iterative(fam, fils, params, model=(on_grid(*m.atoms(1250.0), 5.0, 81, 1250.0), {}),
                             random_starts=4, seed_starts=2)
P, om = 42.461, 73.7308
seg = res.segments
print(seg.round(2).to_string(index=False))
# helical phase: rotation and shift are equivalent along the screw; the invariant is roll - shift * om / P (mod 360)
for name, (th, sh, fl) in truth.items():
    rows = seg[seg.filament == name]
    ph = ((rows.roll_deg - rows.shift_A * om / P) % 360).round(1).tolist()
    print(f"{name}: true rotation {th:6.1f}, shift {sh:5.1f}, flip {fl} | estimated helical phase per segment {ph} "
          f"| segments' shift differences {np.diff(rows.shift_A).round(1).tolist()}")

# pairwise agreement of particles extracted at the registration picks (clean volumes rebuilt from truth)
from copick_helix.registration import Registration, extract, lattice_particles  # noqa: E402

clean = {}
for name, (th, sh, fl) in truth.items():
    v = np.roll(rotate(plus, th, axes=(1, 2), reshape=False, order=1), int(sh / 5), axis=0)
    clean[name] = (v[::-1, ::-1, :].copy() if fl else v).astype(np.float32)
subs, labels = [], []
for row in seg.itertuples():
    reg = Registration(row.segment, row.flip, row.roll_deg, row.shift_A, row.score)
    pos, rots, s = lattice_particles(fils[row.filament], reg, fam.segment_length, 1.0, (P, om), fam.plus_at_minus_z,
                                     centre_only=True)
    subs.append(extract(clean[row.filament], 5.0, np.zeros(3), pos[0], rots[0], (200.0, 80.0, 80.0)))
    labels.append(f"{row.filament}/{row.segment}{'F' if row.flip else ''}")
M = np.array([[T._cc(a, b) for b in subs] for a in subs])
print(pd.DataFrame(M, index=labels, columns=labels).round(2).to_string())
