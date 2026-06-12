# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-12

### Changed
- **Per-file dual licensing.** Original xcx code — the public API/data-layout
  layer, the AD family harnesses, the test suites, and all
  `Provenance: clean-room` functionals — is now dual-licensed **MIT OR
  Apache-2.0** (the standard Rust-ecosystem terms); files derived from libxc
  (`Provenance: ported-from-libxc`, plus mixed-provenance files) remain
  **MPL-2.0**. `LICENSE` was renamed to `LICENSE-MPL`; `LICENSE-MIT` and
  `LICENSE-APACHE` were added; `NOTICE` documents the full scheme; the crate
  SPDX expression is now `(MIT OR Apache-2.0) AND MPL-2.0`. Because MPL-2.0 is
  file-level copyleft, downstream usage is unaffected (dependents could and
  still can use xcx from any project).
- **Documentation repositioning.** README, crate docs, and CONTRIBUTING now
  describe xcx as an independent, AD-first XC-functional library that keeps
  libxc ids/conventions for interoperability — rather than as a libxc
  reimplementation — and highlight the xcx-only features (double-hybrid family
  with structured PT2/CAM metadata, `fxc` everywhere via AD, pure-Rust
  portability).

### Added
- **Double hybrids:** first functionals on the `DoubleHybrid` rung.
  xcx emits the scaled semilocal XC mix; the host adds exact exchange
  (`exx_fraction` / CAM) and the PT2 correlation scaled by the new
  `double_hybrid()` coefficients (xcx never evaluates PT2 -- scope fence).
  libxc (any release) ships no double hybrids, so these carry **xcx-private
  ids >= 100000** (names keep the libxc convention; see
  docs/api-convention.md "Functional-id namespace"):
  - **B2PLYP** (Grimme, JCP 124, 034108 (2006)): `hyb_gga_xc_b2plyp`
    (100001) -- `0.47*B88-x + 0.73*LYP-c`, EXX 0.53, PT2 c_os = c_ss = 0.27.
  - **revDSD-PBEP86-D4** semilocal core (Santra, Sylvetsky & Martin, JPCA
    123, 5129 (2019)): `hyb_gga_xc_revdsd_pbep86_d4` (100002) --
    `0.31*PBE-x + 0.4210*P86-c`, EXX 0.69, PT2 c_os = 0.5922, c_ss = 0.0636.
  - **PWPB95** (Goerigk & Grimme, JCTC 7, 291 (2011)): `hyb_mgga_xc_pwpb95`
    (100003) -- `0.50*mPW-x(reopt.) + 0.731*B95-c(reopt.)` (reusing the
    parameterized mPW91/B95 sources shared with PW6B95), EXX 0.50, SOS-PT2
    c_os = 0.269, c_ss = 0.
  - **wB97M(2)** (Mardirossian & Head-Gordon, JCP 148, 241736 (2018)):
    `hyb_mgga_xc_wb97m_2` (100004) -- the wB97M-V machinery (shared SR-erf
    attenuation + {w, u} series) with the paper's Table-II coefficient set,
    CAM w = 0.30, a = c_x = 0.62194, b = 0.37806 (UEG constraint
    c_x + c_x,00 = 1), single-coefficient PT2 c_os = c_ss = c_PT2 = 0.34096,
    and VV10 retained (b = 6.0, C = 0.01) scaled by the host as
    c_VV10 = 1 - c_PT2 = 0.65904 (the paper's constraint makes the scale
    derivable from `double_hybrid()`). Clean-room.
  Validation: B2PLYP / revDSD-PBEP86 / PWPB95 are golden-verified <= 1e-10
  against pinned libxc 6.1.0 **component mixes** (PWPB95 through
  `xc_func_set_ext_params` overrides for its reoptimized parameters);
  wB97M(2), whose coefficient tables exist in no libxc, carries FD-validated
  xcx-self regression snapshots (marked "not libxc-verified" -- new
  `gen_self` convention in xcx-validation).
- **`gga_c_p86`** (libxc 132, Perdew 86 correlation: PZ81 LDA + gradient
  term), needed by revDSD-PBEP86 and registered as a public id.
  Golden-verified vs libxc 6.1.0 (vxc + fxc, both spins, full point set
  including the sigma_ab clamp corner and exact sigma = 0) to <= 1e-10. P86 is
  the first functional **odd** in the total reduced gradient (its `H` term
  carries `exp(-Phi)`, `Phi ~ sqrt(sigma_tot)`): the AD-safe branch at exactly
  `sigma_tot = 0` keeps value and first derivative exact where the naive sqrt
  chain would NaN, and its `v2sigma2 ~ sigma^(-1/2)` small-sigma divergence is
  a property of the published functional (locked by a dedicated scaling test,
  and excluded from the even-functional small-sigma stability gate).
- **Metadata v2 population:** `double_hybrid()` now returns the published PT2
  coefficients for the four double hybrids (else `None`); `rung()` reports
  `DoubleHybrid` for them (including the range-separated wB97M(2));
  `dispersion()` now recommends **D4** (with the dftd4-convention `param_set`
  key, e.g. "b3lyp", "pbe0", "r2scan", "b2plyp", "revdsdpbep86", "pwpb95")
  for every registered functional with a published D4 parameter set, keeping
  the existing VV10 pairings; LDAs, ambiguous components (B88/LYP/P86),
  M06-2X, and wB97M(2) stay `None`.
- **Range separation + VV10:** first range-separated and
  VV10-carrying functionals, sharing a new AD-safe SR-erf attenuation kernel
  (`attenuation.mpl` port: exact Tawada closed form below a = 1.35, libxc's
  order-16 asymptotic series above it; `erf` for dual numbers via exact
  nilpotent Taylor reconstruction over `libm::erf`):
  - **B97M-V** (Mardirossian & Head-Gordon, JCP 142, 074111 (2015)):
    `mgga_xc_b97m_v` (libxc 254) -- the {w, u} 2D inhomogeneity expansion over
    meta-GGA variables; pure meta-GGA rung, VV10 `b = 6.0, C = 0.01` exposed
    via metadata (xcx never evaluates the nonlocal integral).
  - **wB97X-V** (Mardirossian & Head-Gordon, PCCP 16, 9904 (2014)):
    `hyb_gga_xc_wb97x_v` (libxc 466) -- SR-erf-attenuated B97 GGA exchange +
    B97 correlation; CAM (frozen convention EXX(r12) = a + b*erf(w*r12)):
    w = 0.30, a = 0.167, b = 0.833; VV10 b = 6.0, C = 0.01.
  - **wB97M-V** (Mardirossian & Head-Gordon, JCP 144, 214110 (2016)):
    `hyb_mgga_xc_wb97m_v` (libxc 531) -- SR-erf-attenuated {w, u}-series
    meta-GGA; CAM w = 0.30, a = 0.15, b = 0.85; VV10 b = 6.0, C = 0.01.
  All three golden-verified against pinned libxc 6.1.0 (exc/vxc + all fxc
  blocks, both spins) to <= 1e-10; minimal documented edge exclusions
  (exact-full-polarization floored minority derivatives and extreme-low-density
  vsigma for B97M-V/wB97X-V -- see docs/api-convention.md section 8; wB97M-V is
  fully pinned with no exclusions). Metadata v2 now reports
  `Rung::RangeSeparatedHybrid` for the two omega functionals, a canonical
  `DispersionModel::Vv10` pairing and level-4 grid-sensitive grids for all
  three; `rung()` keeps B97M-V (exx 0, no CAM, VV10-only hybrid record) on the
  `MetaGga` rung.
- **Metadata v2** accessors on `FunctionalInfo` for downstream functional
  selection: `rung()` (Jacob's-ladder rung derived from family + hybrid info),
  `dispersion()` (canonical dispersion pairing; `None` for all current
  functionals -- the D3(BJ)/D4 choice is host policy), `grid()` (grid-level
  recommendation; level 4 + `grid_sensitive` for the Minnesota M06 family,
  level 3 otherwise), and `double_hybrid()` (`None`; no double hybrids yet).
  New public types `Rung`, `DispersionModel`, `DispersionRec`, `GridRec`,
  `DoubleHybridParams`. See `docs/api-convention.md` (Metadata v2).
- **M06-2X** (Zhao & Truhlar 2008): `hyb_mgga_x_m06_2x` (libxc 450, hybrid
  meta-GGA exchange, 54% EXX; the M05 form -- PBE-x enhancement x kinetic
  series, no VS98 part) and `mgga_c_m06_2x` (libxc 236; the M06-L correlation
  form with the 2X parameter set, now a single parameterized source shared with
  `mgga_c_m06_l`). Golden-verified against pinned libxc 6.1.0 (vxc + fxc, both
  spins) to <= 1e-10.
- **PW6B95** (Zhao & Truhlar 2005): `hyb_mgga_xc_pw6b95` (libxc 451, hybrid
  meta-GGA XC, 28% EXX) as libxc's mix `0.72*mPW91-x(PW6 params) +
  B95-c(PW6 params)`, with the mPW91 enhancement rewritten sqrt-free in the
  squared reduced gradient (reusing B88's series-protected `x*asinh(x)`
  kernel). Golden-verified (vxc + fxc, both spins) to <= 1e-10; the
  non-physical n <= 1e-12 extreme vxc points are excluded from the golden pin
  (analytic-vs-AD floor cancellation; fuzz still covers finiteness to
  n = 1e-14).

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

[0.3.0]: https://github.com/nmrtist/xcx/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nmrtist/xcx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nmrtist/xcx/releases/tag/v0.1.0
