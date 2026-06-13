# Golden snapshots

Committed reference outputs generated from a pinned conda-forge **libxc**, used by
`tests/golden.rs` so the verification suite runs in CI without libxc present.

Each `*.json` file is an array of `xcx_validation::GoldenCase` records (functional,
libxc id + version, spin mode, packed inputs, and libxc `exc`/`vrho`/`vsigma`).

Regenerate with (see `../README.md`):

```bash
cargo run -p xcx-validation --features libxc-ffi --bin gen_golden
```

Each functional has a snapshot here, alongside the end-to-end SCF cross-check
artifacts under `scf/` and the biased real-grid subsets `scf_grid_*.json`.
`constructor/` holds corpora for the **public parameterized constructors**
(libxc functionals not registered as xcx ids, e.g. the original B97 vs
`Functional::b97_xc`), loaded by `tests/constructor.rs` instead of the
by-name golden test. The
golden test loads every committed `*.json` and compares against `xcx`.
