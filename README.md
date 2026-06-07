# xcx

[![crates.io](https://img.shields.io/crates/v/xcx.svg)](https://crates.io/crates/xcx)
[![docs.rs](https://img.shields.io/docsrs/xcx)](https://docs.rs/xcx)
[![CI](https://github.com/nmrtist/xcx/actions/workflows/ci.yml/badge.svg)](https://github.com/nmrtist/xcx/actions/workflows/ci.yml)
[![license: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](LICENSE)

A **pure-Rust** library of exchange–correlation (XC) functionals for
density-functional theory (DFT) — a [libxc](https://libxc.gitlab.io/)-compatible
reimplementation with **no C dependency**, no build-time toolchain requirements,
and trivial cross-compilation.

## What it does

Given a density (and, depending on the functional, its gradient and/or kinetic
energy density), `xcx` returns the XC **energy per particle** and its first
derivatives, plus rich **metadata** (family, input requirements, exact-exchange
fraction, range-separation and VV10 parameters).

Each functional is written **once** as a single scalar energy expression;
derivatives are obtained by **forward-mode automatic differentiation**
([`num-dual`](https://crates.io/crates/num-dual)), so they are correct by
construction. Functional IDs and names match libxc for drop-in interoperability.

## What it does *not* do (scope fence)

`xcx` maps `(rho, sigma, tau[, lapl]) → energy density + derivatives + metadata
+ linear mixing` and nothing else. It deliberately contains **no** integration
grids, atomic-orbital evaluation, SCF driver, or dispersion correction. For
hybrids and VV10 it **exposes the parameters** (EXX fraction, CAM ω/α/β, VV10
b/C) so the host electronic-structure code can build those terms; `xcx` never
computes the exact-exchange or nonlocal-correlation integrals.

See [`docs/api-convention.md`](docs/api-convention.md) for the full,
semver-stable data/ABI contract.

## Install

```toml
[dependencies]
xcx = "0.1"
```

MSRV: Rust 1.87.

## Quick start

```rust
use xcx::{Functional, FunctionalId, Spin, XcInput};

fn main() -> Result<(), xcx::XcError> {
    // Spin-unpolarized LDA exchange over three grid points.
    let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized)?;
    let rho = [0.1_f64, 0.2, 0.3];
    let out = f.eval(rho.len(), &XcInput::lda(&rho))?;

    // out.exc[i]  = XC energy per particle ε_xc at point i
    // out.vrho[i] = ∂(n·ε_xc)/∂n at point i
    println!("{:?}", out.exc);
    Ok(())
}
```

## Implemented functionals (v0.1)

All with **energy + all first derivatives**, in **both** spin-polarized and
unpolarized modes. The parenthesised number is the libxc functional id.

| Family  | Functional | libxc id | Notes |
|---------|------------|----------|-------|
| LDA     | `lda_x`            | 1   | Slater exchange |
| LDA     | `lda_c_pw`         | 12  | PW92 correlation |
| LDA     | `lda_c_vwn`        | 7   | VWN5 |
| LDA     | `lda_c_vwn_3`      | 30  | VWN3 |
| LDA     | `lda_c_vwn_rpa`    | 8   | VWN5 (RPA) |
| GGA     | `gga_x_pbe`        | 101 | PBE exchange |
| GGA     | `gga_c_pbe`        | 130 | PBE correlation |
| GGA     | `gga_x_b88`        | 106 | Becke 88 exchange |
| GGA     | `gga_c_lyp`        | 131 | Lee–Yang–Parr correlation |
| Hybrid  | `hyb_gga_xc_b3lyp` | 402 | B3LYP — uses **VWN_RPA** (matching libxc 402) |
| Hybrid  | `hyb_gga_xc_b3lyp5`| 475 | B3LYP/VWN5 |
| Hybrid  | `hyb_gga_xc_pbeh`  | 406 | PBE0 |

Meta-GGA, range-separated hybrids, and second derivatives (`fxc`) are planned
post-v0.1 via the same AD path.

## Verification

Correctness is checked two ways:

- **Public, dependency-free tests** (run in CI, no libxc needed): finite-difference
  self-consistency of derivatives, polarized/unpolarized consistency, analytic
  spot-checks, and a fuzz gate asserting **finite** energy *and every derivative
  component* (no NaN/Inf/panic) across the physical input range for all
  functionals, both spins.
- **Golden cross-check vs. libxc** (`crates/xcx-validation`, never published): values
  are compared against snapshots generated from a **pinned conda-forge libxc
  (6.1.0)** to **≤ 1e-10 relative**, plus an end-to-end SCF cross-check on real
  integration grids. The snapshots are committed, so CI runs deterministically
  **without** libxc present.

## Repository layout

This is a Cargo workspace with two crates:

- **`crates/xcx`** — the published library. Its only dependencies are `num-dual`
  and `nalgebra`; it carries **no C/FFI surface** of any kind.
- **`crates/xcx-validation`** — an **unpublished** (`publish = false`) crate that
  cross-checks `xcx` against the reference C **libxc**. The libxc FFI
  (`libloading`, behind the `libxc-ffi` feature), the committed reference
  snapshots, and the snapshot-regeneration tools all live here on purpose, so the
  published `xcx` crate stays dependency-light and its packaged artifact can never
  include test data. See
  [`crates/xcx-validation/README.md`](crates/xcx-validation/README.md).

The everyday `xcx` tests (unit, finite-difference, and the fuzz gate) live in
`crates/xcx` itself and run in CI without libxc.

## Relationship to libxc (and known divergences)

`xcx` aims to reproduce pinned libxc to ≤ 1e-10, **even where libxc is the less
accurate of the two** — faithfulness beats "being right" so results are
interchangeable. The handful of intentional, documented numerical divergences
(all outside the physically-relevant domain or below the golden tolerance) are
described in [`docs/api-convention.md`](docs/api-convention.md#faithfulness--known-divergences-from-libxc).

## License

Licensed under the **Mozilla Public License 2.0** ([`LICENSE`](LICENSE)),
matching upstream libxc. MPL-2.0 is *file-level* copyleft: you may depend on
`xcx` from MIT/Apache/proprietary projects freely; only modifications to `xcx`'s
own source files must remain under the MPL. Portions are derived from libxc; see
[`NOTICE`](NOTICE) for attribution and per-file provenance.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All contributions are made under the MPL-2.0;
new functionals must be golden-verified against libxc.
