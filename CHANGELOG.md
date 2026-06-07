# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nmrtist/xcx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nmrtist/xcx/releases/tag/v0.1.0
