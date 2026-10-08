# copick-helix: plan

Working name. Fast, data-driven polarity, lattice and in-plane orientation of traced filaments in cryo-ET, without
subtomogram refinement. Microtubules (MT) work today; this plan extends the same approach to actin and intermediate
filaments (IF) and turns the prototype scripts in `/hpc/projects/group.czii/utz.ermel/mt-polarity/scripts` into a
package.

## Principles (shared by every filament family)

1. **Measure only where structure says signal is.** A helical assembly puts its Fourier signal on layer lines, and on a
   layer line only the Bessel orders its selection rule allows. Each family declares that support (layer lines from
   rise and twist, allowed orders, radius band). It is an inductive bias from known structures, not a template.
2. **Use only measured data.** Fourier samples inside the missing wedge are excluded, never zero-filled. Where the
   tilt range is asymmetric, only its symmetric part is used, so the asymmetry cannot create chirality.
3. **Fit the family's parameters from each filament's own data**: the axial repeat (MT); rise and twist (actin, IF);
   protofilament number from the equator (MT).
4. **Polarity from the data, the model only names it.** Two independent read-outs, both kept:
   - **Phase invariants:** wedge-aware least-squares fit of the allowed orders, then quantities that ignore
     rotation and shift but complex-conjugate under a polarity flip (radial phase drift per term; triple products
     whose orders and layer lines add up). They are averaged over a filament's segments with no alignment. A
     filament x filament matrix gives the relative assignment (its leading eigenvector).
   - **Iterative reference:** 2-parameter alignment (rotation about, shift along the axis) plus a per-filament
     polarity, averaged into a reference built only from the data. It must converge from random and mixed starts,
     and it is restricted to the family's Fourier support, which removes missing-wedge leakage.
   - **Absolute label (optional):** an atomic model passed through the same pipeline once.
5. **A screen must be able to find nothing.** An apolar filament (IF) has to come out apolar: no eigenvalue gap,
   halves agreeing at chance, reference polarity no stronger than with random assignments. This is the negative
   control for the whole framework.
6. **Confidence is part of the answer**: leave-one-out projection, per-segment and odd/even agreement, agreement
   between the two methods, bootstrap. Only confident filaments become refinement seeds.
7. **Fast**: seconds per filament on tomograms. No refinement.

## Families

| | Microtubule | Actin | Intermediate filament (vimentin) |
|---|---|---|---|
| Structure | N_S lattice (13_3 in these cells), seam | 1-start left-handed helix, 2 long-pitch strands | 5 protofibrils, luminal fiber (PDB 8RVE) |
| Polar | yes | yes | **no** (antiparallel tetramers) |
| Helical parameters | monomer repeat about 41 A (measured per filament) | rise about 27.5 A, twist about -166.6 deg (fit per filament; varies, e.g. with cofilin) | rise 42.5 A, twist 73.7 deg (8RVE); about 21 nm protofibril repeat |
| Radius band | 80-145 A (wall) | 0-50 A | about 0-60 A |
| Recentering prior | ring, r 112 A | solid rod, r about 35 A | rod or tube, r about 50 A |
| Fourier support | equator n = 0, +-13 k; monomer line n = 3 + 13 k | (n, m) terms with Z = (m + n * twist/360) / rise: 59 A (n = -1), 51 A (n = +1), 36 nm crossover (n = +2), 27.5 A meridian (n = 0) | from 42.5 A / 73.7 deg; to be mapped from 8RVE |
| Lattice read-out | protofilament number from the real-space equator count | twist / crossover length per filament | rise / twist, protofibril repeat, diameter |
| Polarity read-out | phase invariants + iterative (done) | term route from tilt-series segments (done): (1,0), (-1,1), (3,-1) low band, (-4,2), (3,-1), (-1,1), (-3,2) mid band | no lattice in 10 A tomograms or 10521 tilt series (term route; the synthetic control recovers it) |
| Biological checks | neighbors, lamella-level clustering | bundle consistency (filopodia-like bundles are uniformly polar) | n/a |

## Package layout

```
src/copick_helix/
  families/{base,microtubule,actin,intermediate}.py   # FilamentFamily: support, priors, terms, recentering profile
  straighten.py      # trace -> rotation-minimizing frames -> sampled volume; recentering with the family's profile
  repeat.py          # axial repeat (MT), rise / twist fit (actin, IF) from layer lines
  fourier.py         # cylindrical resampling, exact layer-line planes, polar resampling, measured-region masks
  fit.py             # wedge-aware least squares of the allowed orders
  invariants.py      # phase invariants: invariants, relative assignment, confidence
  iterative.py       # data-built reference: alignment, per-filament polarity, convergence, cross-validation
  lattice.py         # MT protofilament number (equator count)
  models.py          # helical models from PDB (gemmi): labels and simulations
  io.py              # copick in (filaments, tomograms, tilt geometry) / out (oriented filaments, polarity_known, metadata)
  tiltseries.py      # local CTF-corrected segment reconstructions through the zarr-particle-tools API
  cli.py             # click commands, registered into the copick CLI by entry point
tests/               # synthetic only (fast): MT plus/minus split, actin plus/minus split, an exactly apolar control
```

## Dependencies and release

- Runtime: numpy, scipy, pandas, zarr, mrcfile, click, gemmi (atomic models for labels), zarr-particle-tools
  (tilt-series reconstructions), and copick >= 1.28 on `main` or copick 2 on the `v2.0` branch. Both copick lines
  have the filament model with `polarity_known`.
- No requirement is hidden in an extra.
- **No circular dependency.** zarr-particle-tools depends on copick (core) but not on copick-utils. This package
  depends on copick and zarr-particle-tools. Neither copick nor copick-utils depends on it.
- It appears in the copick CLI through copick's plugin entry points (as copick-utils does with
  `copick.process.commands`), so copick needs no change.
- Keeping it out of copick-utils keeps gemmi, zarr-particle-tools and the scientific stack out of copick-utils'
  dependencies.
- Proposed home: a repository under the copick organization. The name is to be decided; candidates are
  `copick-helix` and `copick-polarity`. The package also measures lattices and in-plane rotation, which argues for
  the former.
- **Copick convention to settle:** with `polarity_known: true` the point order follows the polarity, but copick does
  not say which end comes first. Proposal: minus -> plus (MT), pointed -> barbed (actin). Record it in the filament
  `metadata` now, and propose a `FilamentSpec` field upstream (FilamentSpec already allows extra keys).

## Milestones

1. Package core and the MT family ported from the prototype scripts. Regression on 10521: identical calls to the
   prototype (54/54 phase invariants, 51 confident in the iterative run).
2. Actin. Science first, then the family: model (e.g. 8D13), Fourier signature, polarity signal against resolution,
   feasibility at 10 A and bin 2 with wedge and noise, per-filament rise / twist. Then real traces (10426 unroofed
   cells, 10521): both methods, consistency checks, bundle check. Tilt-series reconstructions if 10 A is not enough.
3. IF. Science (8RVE: support, confirm apolarity), then the null test on real IF traces and the lattice read-out.
4. copick I/O and CLI: write oriented filaments (session per method) with `polarity_known` for confident filaments,
   and method, confidence and convention in `metadata`; RELION export through copick's filament convention.
5. Release: name, repository, versioning (needs a decision; nothing is pushed before that).

## Data

- Copick projects: `/hpc/projects/group.czii/utz.ermel/10521/copick_config.json`,
  `/hpc/projects/group.czii/utz.ermel/10426/copick_config.json` (10 A tomograms).
- Traces:
  - `microtubule:baseline/v1`
  - `actin:trace-cyl-consensus/v2-min100`
  - `intermediate-filament:baseline/v1`
- Tilt geometry and CTF (10521): `/hpc/projects/group.czii/utz.ermel/portal_avg/10521/microtubule/Import/job001/`
  (read only).
- Prototype results: `/hpc/projects/group.czii/utz.ermel/mt-polarity/results/`.

## Decisions since the first draft (2026-10-07)

- **Copick lines:** `main` targets copick 1 (>= 1.28) and `v2.0` adds one commit pinning copick 2; the code is shared.
  gemmi and zarr-particle-tools are regular dependencies, not extras.
- **Twist:** the physical (deposited) twist, Z = (m - n twist / 360) / rise: MT -360/N, vimentin +73.73, actin about
  -166.6.
- **Outputs of `helix-polarity -o object:user/session`:**
  - **filaments:** the recentered center line as a Catmull-Rom curve, ordered minus -> plus when known, with
    `polarity_known` for seeds and the analysis in `metadata["copick_helix"]`;
  - **picks** (same name): one registration per segment, a lattice-registered particle with its full transform
    (+Z towards the plus end);
  - **work directory:** calls, summary, and `reference_<family>.mrc` (the fast average of the registration
    particles, the initial model). Straightened volumes are kept only on request.
- **`helix-picks`:** dense lattice-registered sampling from the stored results (every n-th dimer or subunit), with no
  recomputation.
- **Lattice gate:** pooled layer-line power against phase-scrambled decoys. A family whose lattice is not detected
  gets no `polarity_known`.
- **Validation:** synthetic round trips (microtubule and vimentin) check the roll, flip, shift and screw conventions,
  each with a control that has to fail. On real 10521 microtubules the package reproduces the prototype calls
  (54/54).
- **Open:** registration continuity along a filament. Neighboring microtubule segments can register to seam positions
  that differ by a 13-fold near-symmetry. Measure it, then choose per segment the equivalent closest to its
  neighbors.
- **Actin (done): the term route** (`bands.py`, `tiltseries.py`, `--source tiltseries`).
  - **Why tilt series:** per-filament polarity is not measurable in 10 A tomograms. Local reconstructions from the
    tilt series carry it at 300-28 A.
  - **What lost it at first:** the method, not the data. It weighted noise-dominated radii, and its term selection
    counted the unresolvable crossover term (-2, 1), which pushed out (3, -1).
  - **Method:** terms selected per band from 6DJO among those the segment length resolves; axis offset refined per
    segment on the low band; dataset axial scale fitted, twist from the model; data-built leave-one-out reference;
    phase-scrambled decoys as the false-positive baseline.
  - **Reproduces the second actin fork on its segments (data / decoy):**

    | | 10426 | 10521 |
    |---|---|---|
    | Halves agree | 154/220 / 112/220 | 105/135 / 60/135 |
    | z >= 3 | 55 / 13 | 43 / 11 |
    | Bundle pairs same polarity | 46/69 / 33/69 | 147/177 / 93/177 |

    Calls agree 0.93 with the fork's data-built calls.
- **Open (actin):**
  - seeds include decoy-level false positives (decoys reach 7-9 'seeds' at the same rule against 36-44): calibrate
    the seed threshold on the decoys;
  - per-filament twist is not fitted (+-0.5 deg moves terms by << 1 Z bin);
  - straightening is sequential, and is the slowest step of a tilt-series run.
- **IF through the term route from the tilt series (10521): no lattice.**
  - 359 filaments, 1,240 segments of 764 A at 4.02 A.
  - Low-band enrichment 1.46-1.48 against 1.55 for decoys, at the fitted rise and at 8RVE's.
  - Halves 63-69/140 (decoys 63-82/140); z >= 3 for 14-16 (decoys 11-13); bundle pairs 10/20 (decoys 10/20).
  - Data-built calls agree with the model's at chance (0.49-0.53).
  - Positive control: synthetic 8RVE filaments through the same route are all called correctly at SNR 0.1 and 0.5
    (low band 6.7-15.9 against 1.1-1.2).
  - So the null is informative: these tilt series do not carry an IF lattice measurable per segment (heterogeneous
    or non-8RVE-like filaments, or below the noise). Not written to copick; 10426 not tried.
- **Status:** test refinements in ApexAgent (MT in portal_avg/10521/microtubule; IF in
  portal_avg/10521/intermediate-filament) are running.
