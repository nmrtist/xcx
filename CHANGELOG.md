# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-08

### Changed
- **Breaking:** `XcInput` and `XcResult` are now `#[non_exhaustive]`. Adding
  future optional inputs (meta-GGA `lapl`/`tau`) and higher derivative orders
  (`fxc`+) is now additive rather than breaking, as `docs/api-convention.md` §7
  promised. Downstream code must construct `XcInput` via `XcInput::lda` /
  `XcInput::gga` (plus the new `with_lapl` / `with_tau` builders) and obtain
  `XcResult` from `eval` (or `XcResult::default()`), not via struct literals.

### Added
- **Second derivatives (`fxc`)** via the same forward-mode AD path:
  `Functional::eval_fxc` returns energy, first derivatives, and the second
  derivatives `v2rho2` / `v2rhosigma` / `v2sigma2` (new `XcResult` fields, empty
  after a plain `eval`). Packing matches libxc's `xc.h` (see
  `docs/api-convention.md` §3); hybrids inherit `fxc` from their semilocal parts.
  Golden-verified against pinned libxc 6.1.0 to ≤ 1e-10 for all 12 functionals,
  both spins — **including the small-σ band, down to σ = 1e-8 and exact 0**. The
  per-spin reduced gradient is carried *squared and sqrt-free*, so the second
  derivatives stay accurate as σ → 0 (a `√σ` form would lose the cancellation
  there); finite-difference and finiteness (fuzz) gated, libxc-free. One
  measure-zero nuance — B88's `v2sigma2` at *exactly* σ = 0, where it is libxc's
  analytic value that is a floor artifact — is detailed in
  `docs/api-convention.md` §8.
- `XcInput::with_lapl` and `XcInput::with_tau` builders for meta-GGA inputs.
- **Meta-GGA functionals** (energy, first derivatives, and `fxc`; both spins),
  built on a new sqrt-free meta-GGA harness (reduced kinetic-energy density and
  the iso-orbital indicator α carried squared/sqrt-free for AD safety):
  `mgga_x_tpss` / `mgga_c_tpss` (TPSS), `mgga_x_r2scan` / `mgga_c_r2scan`
  (r²SCAN), and `mgga_x_m06_l` / `mgga_c_m06_l` (M06-L).
- **PBE-family GGAs:** `gga_x_pbe_r` (revPBE), `gga_x_rpbe` (RPBE), and
  `gga_x_pbe_sol` / `gga_c_pbe_sol` (PBEsol exchange and correlation).
- All new functionals are golden-verified against pinned libxc 6.1.0 to ≤ 1e-10,
  in both spin-polarized and unpolarized modes.

## [0.1.0] - 2026-06-07

Initial public release.

### Added
- Pure-Rust, libxc-compatible exchange–correlation functionals with **no C
  dependency**.
- 12 functionals, each with **energy + all first derivatives**, in both
  spin-polarized and unpolarized modes:
  - LDA: `lda_x`, `lda_c_pw`, `lda_c_vwn` (VWN5), `lda_c_vwn_3` (VWN3),
    `lda_c_vwn_rpa`.
  - GGA: `gga_x_pbe`, `gga_c_pbe`, `gga_x_b88`, `gga_c_lyp`.
  - Hybrids: `hyb_gga_xc_b3lyp` (402, VWN_RPA), `hyb_gga_xc_b3lyp5` (475, VWN5),
    `hyb_gga_xc_pbeh` (406, PBE0).
- Forward-mode automatic differentiation (`num-dual`) for all derivatives.
- Stable, `#[non_exhaustive]` public API (see `docs/api-convention.md`); libxc
  ids and names; linear mixing of functionals.
- Hybrid metadata (exact-exchange fraction; CAM and VV10 parameter slots) without
  computing the exact-exchange or nonlocal integrals.
- Verification: golden cross-check vs. pinned libxc 6.1.0 to ≤ 1e-10, an
  end-to-end SCF cross-check on real integration grids, and a dependency-free
  fuzz gate asserting finiteness of every output across the physical input range.

[Unreleased]: https://github.com/nmrtist/xcx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nmrtist/xcx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nmrtist/xcx/releases/tag/v0.1.0
