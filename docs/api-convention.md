# xcx API & data convention (frozen contract)

This document freezes the public contract so the API and metadata can stabilize
under semantic versioning. Anything not specified here is an implementation
detail and may change. Breaking changes to anything below require a major-version
bump.

## 1. Scope fence (what xcx is)

xcx is a pure function library:

```
(rho, sigma, tau[, lapl])  ->  energy density + derivatives + metadata + linear mixing
```

It does **not** provide, now or as part of this contract:

- integration grids or quadrature,
- atomic-orbital / basis-function evaluation,
- an SCF driver or density construction,
- dispersion corrections,
- the exact-exchange integral for hybrids, or the nonlocal integral for VV10.

For hybrids and VV10, xcx **exposes parameters** (EXX fraction; CAM ω/α/β; VV10
b/C) via metadata so the host code can build those terms itself.

## 2. Quantities and units

Atomic (Hartree) units throughout. Per grid point:

| Symbol | Meaning | Family |
|---|---|---|
| `rho`   | spin density n_σ | all |
| `sigma` | contracted density gradient ∇n_σ·∇n_σ' | GGA, meta-GGA |
| `lapl`  | Laplacian ∇²n_σ | meta-GGA (optional) |
| `tau`   | kinetic energy density τ_σ = ½ Σ_i |∇ψ_iσ|² | meta-GGA |
| `zk`    | XC energy **per particle**, ε_xc (energy density e = n·ε_xc) | all |
| `vrho`   | ∂e/∂n_σ | all |
| `vsigma` | ∂e/∂σ | GGA, meta-GGA |
| `vlapl`  | ∂e/∂(∇²n_σ) | meta-GGA |
| `vtau`   | ∂e/∂τ_σ | meta-GGA |

`zk` is energy per particle (matching libxc); the first derivatives are
derivatives of the **energy density** e = n·ε_xc, where n is the total density.

## 3. Spin and array packing

`Spin` is either `Unpolarized` or `Polarized`. Arrays are **point-major**: for
`np` points the per-point block is contiguous, components vary fastest. Let
`ns = 1` (unpolarized) or `2` (polarized).

| Array | Unpolarized length / packing | Polarized length / packing |
|---|---|---|
| `rho`    | `np`        `[n]`           | `2*np`  `[n_a, n_b]` |
| `sigma`  | `np`        `[σ]` (= ∇n·∇n) | `3*np`  `[σ_aa, σ_ab, σ_bb]` |
| `lapl`   | `np`        `[∇²n]`         | `2*np`  `[∇²n_a, ∇²n_b]` |
| `tau`    | `np`        `[τ]`           | `2*np`  `[τ_a, τ_b]` |
| `zk`     | `np`                        | `np` |
| `vrho`   | `np`                        | `2*np`  `[∂/∂n_a, ∂/∂n_b]` |
| `vsigma` | `np`                        | `3*np`  `[∂/∂σ_aa, ∂/∂σ_ab, ∂/∂σ_bb]` |
| `vtau`   | `np`                        | `2*np` |
| `vlapl`  | `np`                        | `2*np` |
| `v2rho2`     | `np`                    | `3*np`  `[aa, ab, bb]` |
| `v2rhosigma` | `np`                    | `6*np`  `[a·aa, a·ab, a·bb, b·aa, b·ab, b·bb]` |
| `v2sigma2`   | `np`                    | `6*np`  `[aa·aa, aa·ab, aa·bb, ab·ab, ab·bb, bb·bb]` |
| `v2rhotau`   | `np`                    | `4*np`  `[a·τa, a·τb, b·τa, b·τb]` |
| `v2sigmatau` | `np`                    | `6*np`  `[aa·τa, aa·τb, ab·τa, ab·τb, bb·τa, bb·τb]` |
| `v2tau2`     | `np`                    | `3*np`  `[τa·τa, τa·τb, τb·τb]` |

This matches libxc's layout exactly, easing interop and verification. The second
derivatives (`fxc`) are produced only by `eval_fxc` (§4); they are empty after a
plain `eval`. Their polarized packing follows libxc's `xc.h`: `v2rho2` is the
symmetric ρ–ρ block `[aa, ab, bb]`; `v2rhosigma` is the full ρ–σ block, ρ-spin
major (`a` then `b`) × σ minor (`aa, ab, bb`); `v2sigma2` is the symmetric σ–σ
upper triangle. `v2rhosigma`/`v2sigma2` are empty for LDA. The meta-GGA τ blocks
follow the same rule: `v2rhotau` is the full ρ–τ block (ρ-spin major × τ-spin
minor), `v2sigmatau` the full σ–τ block (σ major × τ-spin minor), and `v2tau2`
the symmetric τ–τ upper triangle; all three are empty unless the functional is
meta-GGA. The Laplacian second-derivative blocks (`v2rholapl`, `v2sigmalapl`,
`v2lapl2`, `v2lapltau`) are not yet produced — no current functional needs the
Laplacian (`needs_lapl = false`) — and are reserved for a future additive
release. (FFI-verified against libxc 6.1.0 with an asymmetric polarized point, so
the ρ/σ/τ-spin ordering of each block is pinned, not assumed.)

## 4. Public types (stable surface)

Kept intentionally small. All public enums **and** the `XcInput` / `XcResult`
structs are `#[non_exhaustive]`, so new families/kinds/IDs, new optional inputs,
and new derivative orders can be added without a major bump. Construct `XcInput`
via its constructors/builders (`lda`, `gga`, `with_lapl`, `with_tau`) and obtain
`XcResult` from `eval` (or `XcResult::default()`); struct literals are not part
of the contract.

```rust
pub enum Spin { Unpolarized, Polarized }          // #[non_exhaustive]

pub enum Family { Lda, Gga, Mgga, HybGga, HybMgga } // #[non_exhaustive]

pub enum Kind { Exchange, Correlation, ExchangeCorrelation, Kinetic } // #[non_exhaustive]

pub enum FunctionalId { /* libxc-numbered variants */ }  // #[non_exhaustive]
impl FunctionalId {
    pub fn as_u32(self) -> u32;                 // == libxc numeric id
    pub fn from_u32(id: u32) -> Option<Self>;
    pub fn from_name(name: &str) -> Option<Self>; // e.g. "gga_x_pbe"
    pub fn name(self) -> &'static str;
}

pub struct FunctionalInfo {
    pub id: Option<FunctionalId>,   // None for a user-built linear mix
    pub name: &'static str,
    pub family: Family,
    pub kind: Kind,
    pub needs_sigma: bool,
    pub needs_lapl: bool,
    pub needs_tau: bool,
    pub dens_threshold: f64,
    pub hybrid: Option<HybridInfo>,
}

pub struct HybridInfo {
    pub exx_fraction: f64,          // global exact-exchange mixing (0.0 for pure)
    pub cam: Option<CamParams>,     // range separation, if any
    pub vv10: Option<Vv10Params>,   // nonlocal VV10 parameters, if any
}
pub struct CamParams  { pub omega: f64, pub alpha: f64, pub beta: f64 }
pub struct Vv10Params { pub b: f64, pub c: f64 }

pub struct XcInput<'a> {            // #[non_exhaustive]
    pub rho:   &'a [f64],
    pub sigma: Option<&'a [f64]>,
    pub lapl:  Option<&'a [f64]>,
    pub tau:   Option<&'a [f64]>,
}
impl<'a> XcInput<'a> {
    pub fn lda(rho: &'a [f64]) -> Self;                 // density only
    pub fn gga(rho: &'a [f64], sigma: &'a [f64]) -> Self; // + gradient
    pub fn with_lapl(self, lapl: &'a [f64]) -> Self;   // meta-GGA builder
    pub fn with_tau(self, tau: &'a [f64]) -> Self;     // meta-GGA builder
}

pub struct XcResult {               // #[non_exhaustive]; build via XcResult::default()
    pub exc:    Vec<f64>,            // len np
    pub vrho:   Vec<f64>,            // len np*ns
    pub vsigma: Vec<f64>,            // len np*nsigma (empty for LDA)
    pub vtau:   Vec<f64>,            // empty unless meta-GGA
    pub vlapl:  Vec<f64>,            // empty unless meta-GGA needs lapl
    pub v2rho2:     Vec<f64>,        // fxc; empty unless eval_fxc was called
    pub v2rhosigma: Vec<f64>,        // fxc; empty for LDA / unless eval_fxc
    pub v2sigma2:   Vec<f64>,        // fxc; empty for LDA / unless eval_fxc
    pub v2rhotau:   Vec<f64>,        // fxc; meta-GGA only / unless eval_fxc
    pub v2sigmatau: Vec<f64>,        // fxc; meta-GGA only / unless eval_fxc
    pub v2tau2:     Vec<f64>,        // fxc; meta-GGA only / unless eval_fxc
}

pub struct Functional { /* opaque: boxed evaluator + spin + info */ }
impl Functional {
    pub fn new(id: FunctionalId, spin: Spin) -> Result<Self, XcError>;
    pub fn by_name(name: &str, spin: Spin) -> Result<Self, XcError>;
    pub fn info(&self) -> &FunctionalInfo;
    pub fn spin(&self) -> Spin;
    pub fn exx_fraction(&self) -> f64;                 // 0.0 if not hybrid
    /// Allocating evaluation: energy + all available first derivatives.
    pub fn eval(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError>;
    /// Allocating evaluation through second order: energy + first derivatives +
    /// `fxc` (`v2rho2`/`v2rhosigma`/`v2sigma2`). Costlier than `eval`.
    pub fn eval_fxc(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError>;
    /// Build a linear combination Σ wᵢ·fᵢ of same-spin functionals.
    pub fn mix(parts: Vec<(f64, Functional)>) -> Result<Functional, XcError>;
}

pub enum XcError {                  // #[non_exhaustive]
    NotImplemented(FunctionalId),
    UnknownFunctional,
    MissingInput(&'static str),     // e.g. a GGA called without `sigma`
    LengthMismatch { expected: usize, found: usize },
    SpinMismatch,                   // mixing functionals of different spin
}
```

### Threshold / degenerate-input behavior (contract)

- Where the **total** density at a point is below `dens_threshold`, all outputs at
  that point are exactly `0.0`.
- Outputs are **finite** for every finite **in-domain** input (no NaN/Inf),
  including `n→0`, full spin polarization (`z = ±1`), `sigma = 0`, and large
  (physically-meaningful) `sigma`. This is part of the frozen contract and is
  fuzz-tested over the physical input range (all functionals, both spins).
  *Domain caveat:* for non-physical, astronomically large gradients (reduced
  gradient `s ≫ 10³` — far beyond any real density), the gradient polynomials can
  exceed f64 range and overflow. This is outside the functional's domain and
  outside the fuzz-tested range (see §8, divergence C). The second derivatives
  (`fxc`) overflow at a *lower* reduced gradient than the first derivatives, so
  the `fxc` finiteness guarantee (and the fuzz gate that enforces it) is bounded
  to the physical reduced-gradient domain; inside that domain every `fxc`
  component is finite.

## 5. Metadata semantics

- `needs_sigma/lapl/tau`: if `true`, `eval` returns `MissingInput` when the
  corresponding `XcInput` field is `None`.
- `exx_fraction`: the global fraction of exact (Hartree–Fock) exchange the host
  must add. Pure semilocal functionals report `0.0`.
- `cam`: present only for range-separated functionals (ωB97X-V, ωB97M-V); when
  present the host builds short/long-range exact exchange from ω/α/β in the
  frozen convention `EXX(r12) = alpha + beta*erf(omega*r12)` (so α is the
  global/short-range fraction — also reported by `exx_fraction` — and α + β
  the long-range limit). xcx evaluates only the SR-attenuated semilocal
  exchange of these functionals.
- `vv10`: present only for VV10-containing functionals (B97M-V, ωB97X-V,
  ωB97M-V, all `{ b: 6.0, c: 0.01 }`); xcx never computes the nonlocal
  integral. B97M-V is otherwise pure (exx 0, no CAM): its `hybrid` record
  exists solely to carry the VV10 parameters and it still reports the
  `MetaGga` rung.
- IDs and names equal libxc's (`xc_funcs.h`); verified against
  `xc_functional_get_number` in the cross-check harness. Functionals that do
  not exist in libxc (the double hybrids) use the xcx-private id namespace
  `>= 100000` — see "Functional-id namespace" below.

### Metadata v2 (host-selection accessors on `FunctionalInfo`)

Additive accessors for downstream functional selection (frozen signatures):

```rust
pub enum Rung { Lda, Gga, MetaGga, Hybrid, RangeSeparatedHybrid, DoubleHybrid }
pub enum DispersionModel { D3Bj, D4, Vv10, None }
pub struct DispersionRec { pub model: DispersionModel, pub param_set: &'static str }
pub struct GridRec { pub level: u8, pub grid_sensitive: bool }   // level 0..=4
pub struct DoubleHybridParams { pub c_os: f64, pub c_ss: f64 }
impl FunctionalInfo {
    pub fn rung(&self) -> Rung;
    pub fn dispersion(&self) -> Option<DispersionRec>;
    pub fn grid(&self) -> GridRec;
    pub fn double_hybrid(&self) -> Option<DoubleHybridParams>;
}
```

- `rung()`: a registered functional carrying PT2 coefficients
  (`double_hybrid()` is `Some`) is `DoubleHybrid` — even when range-separated
  (ωB97M(2)): the rung reports the highest treatment the host must provide.
  Otherwise derived from `Family` + `hybrid` — a hybrid with `cam` present is
  `RangeSeparatedHybrid` (CAM convention, frozen: `EXX(r12) = alpha + beta*erf(omega*r12)`),
  any other hybrid with nonzero exact exchange (or a Hyb* family) is `Hybrid`,
  pure families map to their semilocal rung — including B97M-V, whose
  `hybrid` record carries only VV10 (exx 0, no CAM ⇒ `MetaGga`).
- `dispersion()`: the VV10-containing B97M-V/ωB97X-V/ωB97M-V/ωB97M(2) report
  `DispersionModel::Vv10` with `param_set` = the canonical functional name
  (the `b`/`C` values live in `hybrid.vv10`; for ωB97M(2) the host scales the
  VV10 energy by `c_VV10 = 1 − c_PT2`). Functionals with a **published
  D4 parameter set** (the dftd4 reference table; Caldeweyher et al., JCP 150,
  154122 (2019)) report `DispersionModel::D4` with `param_set` = the
  **dftd4-convention key** (lowercase, e.g. `"b3lyp"`, `"pbe0"`, `"r2scan"`,
  `"b2plyp"`, `"revdsdpbep86"`, `"pwpb95"`) that the host resolves against its
  own dftd4 damping-parameter table — xcx ships no dispersion data or code
  (scope fence). Component ids map to the canonical functional their name
  denotes (`mgga_x_r2scan` → `"r2scan"`, `gga_x_pbe_r` → `"revpbe"`, …);
  ambiguous components (B88, LYP, P86) and functionals without a published D4
  set (LDAs, M06-2X) return `None`. The D3(BJ)-vs-D4 *choice*
  remains host policy; this is a recommendation.
- `grid()`: level 3 / not grid-sensitive for standard LDA/GGA/meta-GGA/hybrids;
  level 4 / `grid_sensitive: true` for the Minnesota M06 family (M06-L and
  M06-2X exchange + correlation), whose documented grid sensitivity (Wheeler &
  Houk, JCTC 2010, 6, 395; Mardirossian & Head-Gordon, Mol. Phys. 2017, 115,
  2315) warrants finer quadrature, and for the combinatorially-optimized
  B97M-V/ωB97X-V/ωB97M-V, whose B97/Minnesota-class inhomogeneity expansions
  are likewise documented as grid-sensitive (Mardirossian & Head-Gordon,
  J. Chem. Theory Comput. 2016, 12, 4303; Mol. Phys. 2017, 115, 2315).
- `double_hybrid()`: `Some(DoubleHybridParams { c_os, c_ss })` for the four
  registered double hybrids — the published PT2 coefficients the host applies
  to its own PT2/MP2-like correlation energy (xcx never evaluates PT2):
  B2PLYP `0.27/0.27`, revDSD-PBEP86-D4 `0.5922/0.0636`, PWPB95 `0.269/0.0`
  (SOS-PT2), ωB97M(2) `0.34096/0.34096` (a single canonical-MP2 coefficient
  `c_PT2`; its retained VV10 partner is scaled by `c_VV10 = 1 − c_PT2 =
  0.65904`, the paper's constraint). xcx emits the *scaled semilocal mix* of a
  double hybrid (e.g. B2PLYP = `0.47·B88-x + 0.73·LYP-c`); the host adds EXX
  (and CAM for ωB97M(2)) plus the scaled PT2. `None` for everything else.

### Functional-id namespace for non-libxc functionals

`FunctionalId::as_u32` equals the libxc numeric id **except** for functionals
absent from libxc (no libxc release ships double hybrids): those use the
xcx-private namespace `>= 100000` (far above libxc's range, currently < 1000):
B2PLYP 100001, revDSD-PBEP86-D4 100002, PWPB95 100003, ωB97M(2) 100004. Names
keep the libxc convention (`hyb_gga_xc_b2plyp`, …), so `from_name` stays
forward-compatible should libxc ever add them.

## 6. Linear mixing

`Functional::mix` evaluates Σ wᵢ·fᵢ over functionals of identical `Spin`,
summing `exc`/`vrho`/`vsigma`/… componentwise. `exx_fraction` of a mix is the
weighted sum of parts' fractions. This is the only composition xcx performs;
hybrids' semilocal parts are expressed this way internally where convenient.

## 7. Stability guarantees

- The items in §4 are semver-stable; the enums **and** the `XcInput` / `XcResult`
  structs are `#[non_exhaustive]`.
- Numerical outputs are validated to ≤ 1e-10 relative vs. pinned libxc over
  rho ∈ [1e-14, 1e3] (screened/zero regions compared by absolute value).
- Because both structs are `#[non_exhaustive]`, adding functionals, optional
  inputs, derivative orders (`fxc`+), or families is additive and non-breaking.
  (Before 0.2 the structs were exhaustive, so adding a field *was* breaking; the
  0.2 `#[non_exhaustive]` change made this guarantee actually hold.)

## 8. Faithfulness & known divergences from libxc

xcx's policy is to match pinned libxc (currently **6.1.0**) to ≤ 1e-10, *even
where libxc itself is the less accurate of the two*, so the two are
interchangeable. The intentional divergences below are documented for
completeness; **none affects a physically-meaningful calculation**:

- **(A) Reproduce-libxc.** Near full spin polarization (`|ζ| → 1`), libxc computes
  the correlation spin-interpolation `f(ζ)` in a form that loses a few digits to
  catastrophic cancellation. xcx deliberately uses the **same** arithmetic so the
  two agree to ≤ 1e-10, rather than using a more accurate (but libxc-divergent)
  reformulation. Exchange is unaffected (libxc's exchange form is already
  cancellation-free, and xcx matches it exactly).

- **(B) xcx-stays-accurate.** For the GGA-exchange *minority* spin-channel
  potential at near-full polarization, libxc's analytic derivative cancels and is
  wrong by ~1e-8 relative at extreme imbalance. xcx's forward-AD derivative is
  consistent with its (correct) energy and matches an independent finite
  difference, so xcx keeps the accurate value; the golden test set simply avoids
  pinning xcx to libxc in that unreliable regime (the exact full-polarization edge
  is screened and clean).

- **(C) Out-of-domain robustness asymmetry.** For pathologically large gradients
  (reduced gradient `s ≫ 10³`, far outside any real density), xcx's forward-AD
  *derivatives* overflow f64 at a lower `sigma` than libxc's pre-simplified
  analytic derivatives. Inside the physical domain (and ~60+ orders of magnitude
  beyond it) both libraries are finite *and* agree to ≤ 1e-10; out there neither is
  meaningful. The first non-finite output is always a derivative, never the energy.

- **(B, second-order) `gga_x_b88` `v2sigma2` at *exactly* `sigma = 0`.** B88's
  enhancement is analytic in `sigma` at 0 (`x·asinh x = t − t²/6 + …`, a
  convergent series in `t = (√sigma/n^{4/3})²`), so its second derivative there
  equals the `sigma → 0` limit. xcx carries every reduced gradient *squared and
  sqrt-free*, so it computes that limit; libxc's *analytic* `v2sigma2` instead
  truncates to (5/8)× the limit at/below its `sigma`-floor and at exact 0 — a
  libxc artifact (libxc emits the correct limit for all `sigma ∈ [1e-8, 1e-20]`).
  xcx is the accurate side, confirmed by an independent finite difference of
  libxc's *own* first derivative and by the closed form `F''(0)`. The `fxc` golden
  set therefore pins the small-`sigma` band (down to `1e-8`, both spins, all GGA/
  hybrid cases ≤ 1e-10) but omits *exact* `sigma = 0` for B88 and the
  B88-containing hybrids (`b3lyp`/`b3lyp5`); PBE-x, the correlation functionals,
  and PBE0 are accurate at `sigma = 0` and *are* pinned there. (This supersedes an
  earlier, inverted claim — that xcx's forward-AD was the less-accurate side
  across a small-`sigma` *band*; the sqrt-free reformulation eliminated that band,
  leaving only this single, opposite-signed zero-gradient point.)

- **(D) Meta-GGA low-density first-derivative divergence (`mgga_c_r2scan`,
  `mgga_c_m06_l`).** At extreme low density (n ≲ 1e-8, far below any physical
  density), r2SCAN correlation's gradient derivatives (`vsigma`/`vtau`) diverge from
  libxc by more than 1e-10 through an analytic-vs-AD cancellation at very large
  `r_s`, while the energy still matches. `mgga_c_m06_l` shows the same class at
  exact **full spin polarization**: it uses the raw per-spin reduced kinetic-energy
  density `τ_σ/n_σ^(5/3)` (not the `(n_σ/n)^(5/3)`-weighted total of TPSS/r2SCAN), so
  as a minority spin density `n_b → 0` its minority-channel `vrho`/`vsigma`/`vtau`
  blow up `∝ n_b^(−8/3)` and the floored-edge value diverges from libxc (~1e-6…1e-5
  relative), while the energy and majority channel still match. Both are the same
  class as the LDA/GGA correlation full-polarization cancellation (A): xcx does not
  pin to libxc in that regime (r2SCAN-c validation stops at n = 1e-8; M06-L-c drops
  the *exact* full-polarization edge but pins the physical near-edge `(1.0, 1e-4)`,
  both ≤ 1e-10), and the finiteness contract still holds down to n = 1e-14 / exact
  full polarization. TPSS, r2SCAN exchange, and M06-L exchange are unaffected.

- **(E) Meta-GGA out-of-domain robustness (the σ-clamp corner).** Mirroring (C):
  for non-physical inputs (very large `sigma` over a tiny density, reduced gradient
  `s ≫ 10³`), the σ_ab-clamp corner drives the *total* contracted gradient to f64
  cancellation noise. There libxc's meta-GGA energy can be non-finite while xcx's
  stays finite (xcx floors the mathematically-nonnegative total gradient at 0),
  the opposite-direction robustness asymmetry to (C). No physical calculation is
  affected; inside the domain the two agree to ≤ 1e-10.

- **(B/D) B97-family floored-edge / low-density first derivatives.**
  The B97 power-series functionals use the *raw* per-spin reduced variables
  `x_σ² = σ_σσ/n_σ^(8/3)` and `t_σ = τ_σ/n_σ^(5/3)` (the M06-family form), so the
  same two artifact classes apply: at **exact full spin polarization** the
  floored minority-channel first derivatives are analytic-vs-AD f64 noise
  amplified by the floor (`hyb_gga_xc_wb97x_v` rel ~1e-4 on vrho/vsigma;
  `mgga_xc_b97m_v` additionally evaluates the opposite-spin kinetic variable
  `w_os` in a symmetric, cancellation-free regrouping — algebraically identical
  to libxc's form, which piles up `2K²` before cancelling — giving vtau_b
  rel ~0.15 at the τ-floor corner), and at **extreme low density**
  (`mgga_xc_b97m_v` n < 1e-8, `hyb_gga_xc_wb97x_v` n < 1e-12) vsigma crosses
  1e-10 (divergence-D class). The energy matches everywhere; the golden vxc set
  drops only those non-physical edge points (the physical near-full-polarization
  point `(1.0, 1e-4)` stays pinned ≤ 1e-10), and `hyb_mgga_xc_wb97m_v`
  (threshold 1e-13) matches at *every* point, including the exact edges, and is
  fully pinned. Fuzz still covers finiteness for all three at the edges.

- **Fermi-hole-curvature (FHC) clamp is build-conditional in libxc.** libxc's
  meta-GGA harness applies the constraint `sigma_σσ ← min(sigma_σσ, 8·n_σ·τ_σ)`
  **only when libxc is compiled with `XC_ENFORCE_FERMI_HOLE_CURVATURE`** (exposed
  as the functional flag `XC_FLAGS_ENFORCE_FHC`). The pinned reference build
  (conda-forge libxc 6.1.0) **has it enabled**, and xcx reproduces that clamp so
  the two match to ≤ 1e-10. Faithfulness to libxc here is therefore relative to a
  *clamp-on* build: a host whose own libxc was built **without** that flag would
  apply no clamp, and could see a tiny meta-GGA mismatch against xcx in the narrow
  region where the constraint is active (`x_σ²/(8 t_σ) > 1`, i.e. `sigma_σσ >
  8·n_σ·τ_σ`). This affects only that clamp region; everywhere else the clamp is a
  no-op and the build flag is immaterial.

In all cases the energy `zk` matches throughout the physical domain; the
differences live only in derivatives at non-physical inputs, the second
derivative's exact zero-gradient point (where it is libxc, not xcx, that is the
less accurate), the extreme-low-density first derivatives (D), the out-of-domain
σ-clamp corner (E), the build-conditional FHC clamp region, or below the 1e-10
tolerance.
