# copick-helix

Fast, data-driven polarity, lattice and in-plane orientation of traced filaments (microtubules; actin and
intermediate filaments in progress) in cryo-electron tomograms, for copick projects. No subtomogram refinement: each
filament's own helical Fourier signal, restricted to where its structure puts signal, measured only where the tilt
series sampled it.

## Install

```bash
pip install -e .        # branch main: copick 1.x; branch v2.0: copick 2
```

gemmi (atomic models, used to name which polarity group is 'plus') and zarr-particle-tools (local CTF-corrected
reconstructions from tilt series) are regular dependencies.

## Use

```bash
copick process helix-polarity -c config.json \
    -i "microtubule:baseline/v1" -t "wbp-filtered@10.0" \
    --family microtubule --tilt-range -45 63 \
    --work-dir helix_out/ -o "microtubule:helix/1"

copick process helix-picks -c config.json -i "microtubule:helix/1" -o "microtubule:helix-picks/1" --every 1
```

`helix-polarity` straightens and recentres each filament and measures its helical parameters. It runs both polarity
methods and registers every segment to a reference built from the data. With `-o object:user/session` it writes:

- **filaments**: the recentred centre lines as Catmull-Rom curves, visible in every copick viewer. Each is ordered
  minus -> plus when the polarity is known, and carries `polarity_known` for seeds (confident in both methods,
  methods agreeing, lattice detected). The calls, confidences, helical parameters and lattice gate go under
  `metadata["copick_helix"]`.
- **picks** (same name): one registration per segment, placed at a lattice point. The full transform maps a reference
  frame with +Z towards the plus end onto the tomogram; `instance_id` is the filament ID.

In WORK_DIR it writes:
- `calls.tsv` and `summary.json`;
- `reference_<family>.mrc`, the fast average of the registration particles in the same frame (an initial model);
- with `--keep-volumes`, the straightened volumes.

`helix-picks` samples every n-th lattice point along each filament from those stored results, with no recomputation.
Every particle sits on the same lattice position and in the same frame, so a refinement can start from local
searches. `helix-polarity --picks URI --every n` writes the same dense picks directly from the analysis.

`--family`: `microtubule` (13_3), `microtubule_N_S`, `intermediate_filament` (vimentin, 8RVE; needs the lattice
gate). Actin is in progress. Use CTF-corrected tomograms: the polarity signal sits at 20-40 A, where the first CTF zero
usually lies.

## Methods

1. **Straightening and recentring.** Rotation-minimising frames along the trace. Windows of the cross-section are
   matched to the family's radial profile (a ring for microtubules) inside the measured Fourier directions.
2. **Per-filament parameters.** The microtubule monomer repeat comes from the 4 nm layer line; the protofilament
   number from the equator.
3. **Phase invariants** (`copick_helix.invariants`). A least-squares fit of the Bessel orders the family's helical
   selection rule allows, on its layer lines, using only measured Fourier samples. From that fit come quantities that
   ignore rotation and shift but turn into their complex conjugate under a polarity flip. A filament x filament
   comparison splits all filaments into two polarity groups.
4. **Iterative data-built reference** (`copick_helix.iterative`). A 2-parameter alignment (rotation about, shift
   along the axis) plus a per-filament polarity, averaged into a reference built from the data. It is restricted
   to the family's Fourier support and converges from random starts.
5. **Labels.** An atomic model (6DPV for microtubules) passed through the same steps names which group is plus. It
   is optional; relative polarity needs no model.

## Dependencies

copick (1.x on `main`, 2.x on `v2.0`), zarr-particle-tools and gemmi. zarr-particle-tools depends on copick core
but not on copick-utils, and neither copick nor copick-utils depends on this package, so there is no cycle. The
commands enter the copick CLI through the `copick.process.commands` entry point.

See `PLAN.md` for the design, the filament families and the status.
