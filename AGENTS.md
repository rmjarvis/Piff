# Piff Repository Notes

## Default Environment

Use the project's preferred conda environment before running commands:

```bash
source /Users/Mike/lsst/lsstsw/miniconda/etc/profile.d/conda.sh
conda activate /Users/Mike/miniforge3/envs/py3.12
```

## Test Workflow

Run the full test suite from the repository root:

```bash
mkdir -p /tmp/piff-mpl
export MPLCONFIGDIR=/tmp/piff-mpl
pytest -q
```

Notes:

- `MPLCONFIGDIR` is required in this environment because `/Users/Mike/.matplotlib` is not writable.
- The full suite passed in this checkout with `138 passed` under Python 3.12.
- `joblib/loky` emits a non-fatal warning about physical core detection on this machine. If desired, set `LOKY_MAX_CPU_COUNT` explicitly to silence it.

## Docs Workflow

Build the Sphinx docs from `docs/`:

```bash
export LC_ALL=C
export LANG=C
mkdir -p /tmp/piff-mpl
export MPLCONFIGDIR=/tmp/piff-mpl
make html
```

Notes:

- The docs currently build successfully.
- The current docs build is clean with no warnings in this environment.
- `docs/_build/` is intentionally kept in the working tree for local documentation builds and is only committed on release branches when docs are ready to publish. Do not add it to `.gitignore`; treat it as expected local output during normal development.

## Development Scratch Area

- In `devel/`, disposable generated artifacts should go in `devel/output/`.
- `devel/.gitignore` ignores `output/`, so scratch outputs there should stay out of `git status`.

## Development Expectations

- Preserve backward compatibility by default, especially for YAML config interfaces and user-facing behavior.
- Prefer semantic-versioning discipline: deprecate old usage in a minor release before removing it in a later major release.
- Avoid breaking existing user workflows without a strong reason and an explicit migration path.
- Maintain very high test coverage; aim for effectively 100% coverage of new behavior.
- All new lines of code should be exercised by at least one test in the test suite.
- Use a mix of targeted unit tests and broader integration, accuracy, and regression tests as appropriate.
- It is acceptable, and often preferred, to cover multiple related edge cases in one test when that
  keeps runtime down and still makes the coverage intent clear.
- When practical, tests should validate results using an independent calculation, analytic expectation, or simpler reference algorithm rather than only reusing the implementation under test.
- Prefer using real functions in tests rather than `mock`/`patch` where practical. Use mocking
  sparingly for hard-to-trigger branches (e.g. forced exception paths).
- Do not add or keep branches that are only reachable via mocking. If a branch cannot be
  triggered by plausible real inputs/calls, simplify/remove that branch instead of testing it
  with synthetic mocks.
- Keep user-added `print` statements in tests unless explicitly asked to remove them. The user
  uses these for interactive inspection while iterating on behavior.
- Do not overwrite unrelated user edits (including comments/annotations/style tweaks) while
  making targeted changes. If a user change appears problematic, ask before altering it.
- For scientific analysis code in this repo, tests should usually include explicit accuracy
  assertions, not merely code-path/smoke checks. Prioritize verifying numerical correctness and
  recovery of known truth whenever feasible.
- Prefer tests that exercise user-facing/public workflows. Avoid calling private/internal helper
  methods directly unless there is a compelling reason; needing direct private-method tests can be
  a sign of unnecessary/extraneous internal code paths.
- Prefer a house style of 100 characters per line or less, except where exceeding that limit is
  clearly the less awkward choice (for example some long URLs).
- For debugging star-level issues from a `.piff` file, prefer using `InputFiles.load_images` to
  repopulate original star cutouts from the source inputs rather than manually reconstructing
  stamp bounds.
- Prefer explicit `if/else` structure over early `return` style when either form is equivalent.
  Use early returns only when they materially improve clarity.
- For non-functional edits only (comments/docstrings/docs/text), do not rerun tests just to
  reconfirm no behavior change. Run tests for functional/code-path changes; full suite is run
  before commit anyway.

## Build Notes

- `setup.py` builds a `pybind11` extension and needs Eigen headers.
- If Eigen is not found locally, `setup.py` attempts to download Eigen 3.4.0 automatically. On restricted/offline systems, set `EIGEN_DIR` (or `C_INCLUDE_PATH`) to a local installation instead of relying on the fallback download.

## Architecture Summary

Piff is a modular PSF-modeling pipeline configured primarily through YAML. The core workflow is:

1. `Input` reads star candidates and image metadata.
2. `Select` filters usable stars.
3. `Model` defines the single-location PSF parameterization.
4. `Interp` models how fitted parameters vary across the focal plane.
5. `Outliers` removes bad stars during iterative fitting.
6. `Output` writes the PSF solution; `Stats` provides diagnostics.

Important scientific/engineering context from the DES paper and current docs:

- A central design goal is modeling the PSF across the full focal plane, not per-chip in isolation, while correctly accounting for WCS transformations.
- The fitting loop jointly refines per-star model parameters and field interpolation, with reserve/validation stars and residual diagnostics used to assess overfitting and model quality.
- Correlation-function diagnostics (`rho` statistics) are a first-class validation tool and should be treated as part of model evaluation, not an afterthought.
- The codebase is intentionally extensible: new development should usually slot into one of the established plugin-style abstractions (`Input`, `Select`, `Model`, `Interp`, `Outliers`, `Stats`, `PSF`) rather than bypassing them.

## Current Capability Snapshot

Compared to the 2020 DES paper, the current code/docs expose a broader set of built-in components:

- Models include `PixelGrid`, `GSObjectModel` variants (`Gaussian`, `Kolmogorov`, `Moffat`), and `Optical`.
- Composite PSFs are supported through `SumPSF` and `ConvolvePSF`.
- Interpolators include `Mean`, `Polynomial`, `BasisPolynomial`, `KNNInterp`, and `GPInterp`.
- Diagnostics include `RhoStats`, `ShapeHistStats`, `HSMCatalogStats`, `TwoDHistStats`, `WhiskerStats`, `StarStats`, and `SizeMagStats`.

When extending functionality, check whether an existing abstraction or diagnostic already covers part of the desired behavior before adding a new top-level concept.

## Roman Notes

- `RomanOptics` expects an explicit `sca` star property from the input configuration.
- For multi-SCA Roman FITS inputs, set `input.properties.sca` from `image_num` in the YAML
  rather than relying on implicit chip numbering or FITS header parsing.
- `InputFiles` supports a generic `properties` dict for attaching constant per-image star
  properties; use this instead of adding Roman-specific keys to the generic input code.
- `RomanOptics` defaults to `chromatic: True`; set `chromatic: False` explicitly when using the
  achromatic effective-wavelength approximation.
- `RomanOptics` uses `aberration_interp` (`global`, `constant`, or `linear`) to control
  aberration interpolation behavior; do not use `per_sca` in config for this class.
- Keep Roman fitting path explicit and specialized: `RomanOptics` should always use batched
  fitting and the four-corner bilinear approximation. Avoid generic `SimplePSF`-style fallback
  branches for Roman.
- `piff/roman_psf.py` exposes a module-level `pupil_bin` control (default 4) used in
  `galsim.roman.getPSF` calls. Tests can temporarily set `roman_psf.pupil_bin` to 8 or 16 to
  speed runtime while still exercising real Roman optics code paths.
