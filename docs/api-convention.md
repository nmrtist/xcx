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

This matches libxc's layout exactly, easing interop and verification.

## 4. Public types (stable surface)

Kept intentionally small. All public enums are `#[non_exhaustive]` so new
families/kinds/IDs can be added without a major bump.

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

pub struct XcInput<'a> {
    pub rho:   &'a [f64],
    pub sigma: Option<&'a [f64]>,
    pub lapl:  Option<&'a [f64]>,
    pub tau:   Option<&'a [f64]>,
}

pub struct XcResult {
    pub exc:    Vec<f64>,            // len np
    pub vrho:   Vec<f64>,            // len np*ns
    pub vsigma: Vec<f64>,            // len np*nsigma (empty for LDA)
    pub vtau:   Vec<f64>,            // empty unless meta-GGA
    pub vlapl:  Vec<f64>,            // empty unless meta-GGA needs lapl
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
  outside the fuzz-tested range (see §8, divergence C).

## 5. Metadata semantics

- `needs_sigma/lapl/tau`: if `true`, `eval` returns `MissingInput` when the
  corresponding `XcInput` field is `None`.
- `exx_fraction`: the global fraction of exact (Hartree–Fock) exchange the host
  must add. Pure semilocal functionals report `0.0`.
- `cam`: present only for range-separated functionals (none in v0.1); when
  present the host builds short/long-range exact exchange from ω/α/β.
- `vv10`: present only for VV10-containing functionals (none in v0.1); xcx never
  computes the nonlocal integral.
- IDs and names equal libxc's (`xc_funcs.h`); verified against
  `xc_functional_get_number` in the cross-check harness.

## 6. Linear mixing

`Functional::mix` evaluates Σ wᵢ·fᵢ over functionals of identical `Spin`,
summing `exc`/`vrho`/`vsigma`/… componentwise. `exx_fraction` of a mix is the
weighted sum of parts' fractions. This is the only composition xcx performs;
hybrids' semilocal parts are expressed this way internally where convenient.

## 7. Stability guarantees

- The items in §4 are semver-stable; enums are `#[non_exhaustive]`.
- Numerical outputs are validated to ≤ 1e-10 relative vs. pinned libxc over
  rho ∈ [1e-14, 1e3] (screened/zero regions compared by absolute value).
- Adding functionals, derivative orders (`fxc`+), or families is additive and
  non-breaking.

## 8. Faithfulness & known divergences from libxc

xcx's policy is to match pinned libxc (currently **6.1.0**) to ≤ 1e-10, *even
where libxc itself is the less accurate of the two*, so the two are
interchangeable. Three intentional divergences are documented for completeness;
**none affects a physically-meaningful calculation**:

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

In all three cases the energy `zk` matches throughout the physical domain; the
differences live only in derivatives at non-physical inputs or below the 1e-10
tolerance.
