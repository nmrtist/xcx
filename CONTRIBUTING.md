# Contributing to xcx

Thanks for your interest in contributing! `xcx` is a pure-Rust XC-functional
library built on automatic differentiation; it keeps
[libxc](https://libxc.gitlab.io/) ids and conventions for interoperability and
ships functionals of its own beyond libxc. The bar for changes is **numerical
fidelity**: anything that touches functional math must reproduce pinned libxc
to ≤ 1e-10 where the two overlap, or be verified against the published
literature where they don't.

## Ground rules

- Licensing is **per file** (see [`NOTICE`](NOTICE)): original xcx code is
  **MIT OR Apache-2.0** ([`LICENSE-MIT`](LICENSE-MIT),
  [`LICENSE-APACHE`](LICENSE-APACHE)); files derived from libxc stay under the
  **MPL-2.0** ([`LICENSE-MPL`](LICENSE-MPL)). New original work (clean-room
  functionals, API/harness code, tests) is contributed under MIT OR
  Apache-2.0; modifications to MPL-tagged files remain MPL.
- Keep the scope fence (see [`docs/api-convention.md`](docs/api-convention.md)):
  xcx maps `(rho, sigma, tau[, lapl]) → energy + derivatives + metadata`. No
  grids, AO evaluation, SCF, or dispersion.
- The public API and metadata are semver-stable. Additive changes (new
  functionals / derivative orders / families) are fine; breaking the surface in
  §4 of the API convention requires a major-version bump.

## Development setup

```bash
cargo build --workspace
cargo test  --workspace
cargo fmt   --all
cargo clippy --workspace --all-targets -- -D warnings
```

CI runs `fmt --check`, `clippy -D warnings`, and `test` on stable and on the
MSRV (currently Rust **1.87**), with **default features only** — no libxc needed.

## Verification model

There are two independent test layers:

1. **Public, dependency-free tests** (ship with the crate, run in CI): derivative
   finite-difference self-consistency, polarized/unpolarized consistency, and a
   fuzz gate asserting finite energy **and every derivative component** across the
   physical input range.
2. **Golden cross-check vs. libxc** in the unpublished `crates/xcx-validation`: values
   are compared against committed snapshots generated from a **pinned conda-forge
   libxc (6.1.0)**. CI uses the committed snapshots, so no libxc is required.

Regenerating snapshots requires libxc and the `libxc-ffi` feature — see
[`crates/xcx-validation/README.md`](crates/xcx-validation/README.md).

## Adding a functional

1. Implement it as a **single scalar energy expression** generic over
   `N: DualNum<f64>`; derivatives come from forward-mode AD. Seed the duals on the
   raw inputs (`rho`/`sigma`) and map to reduced variables inside the closure.
2. Reuse shared building blocks rather than forking copies (e.g. `pw92_ec`,
   the shared GGA-exchange skeleton, the VWN rows).
3. Tag the file's provenance: `Provenance: ported-from-libxc` (derived from libxc
   Maple/C — carries the MPL-2.0 header) or `Provenance: clean-room` (from
   published literature — carries the `SPDX-License-Identifier: MIT OR
   Apache-2.0` header).
4. Add golden snapshots and confirm ≤ 1e-10 against libxc, including edge cases
   (full spin polarization, small/large `rho` and `sigma`). For functionals
   with no libxc counterpart, verify via `xc_func_set_ext_params`
   re-parameterization where possible and literature reproduction otherwise.
5. Wire the id into the registry; `build()` matches every id explicitly.

## Numerical-stability changes

If you reformulate an expression for stability, make sure it does not perturb the
golden-verified numbers, and prefer cancellation-free / sqrt-free forms that keep
forward-AD derivatives well-behaved (see the notes in the relevant source files
and `docs/api-convention.md` §8).

## Pull requests

- Keep PRs focused; one functional or one concern per PR where possible.
- Ensure `fmt`, `clippy -D warnings`, and the full test suite pass.
- Describe how the change was verified (golden, finite difference, etc.).
