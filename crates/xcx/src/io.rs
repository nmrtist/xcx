// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Input/output data types. See `docs/api-convention.md` for packing rules.

/// Inputs to a functional evaluation, as borrowed point-major slices.
///
/// Packing (with `np` points, `ns = 1` unpolarized / `2` polarized): `rho` has
/// `ns*np` entries (`[n]` or `[n_a, n_b]` per point), `sigma` has `(2*ns-1)*np`
/// (`[σ]` or `[σ_aa, σ_ab, σ_bb]`), `lapl`/`tau` have `ns*np`.
///
/// `#[non_exhaustive]`: construct via [`XcInput::lda`] / [`XcInput::gga`] (plus
/// the [`with_lapl`](XcInput::with_lapl) / [`with_tau`](XcInput::with_tau)
/// builders for meta-GGA inputs), never a struct literal — new optional fields
/// can then be added without a breaking change.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct XcInput<'a> {
    /// Spin density n_σ.
    pub rho: &'a [f64],
    /// Contracted density gradient ∇n_σ·∇n_σ'. Required for GGA / meta-GGA.
    pub sigma: Option<&'a [f64]>,
    /// Laplacian ∇²n_σ. Some meta-GGAs require it.
    pub lapl: Option<&'a [f64]>,
    /// Kinetic energy density τ_σ. Required for meta-GGA.
    pub tau: Option<&'a [f64]>,
}

impl<'a> XcInput<'a> {
    /// Convenience constructor for an LDA input (density only).
    pub fn lda(rho: &'a [f64]) -> Self {
        Self {
            rho,
            sigma: None,
            lapl: None,
            tau: None,
        }
    }

    /// Convenience constructor for a GGA input (density + gradient).
    pub fn gga(rho: &'a [f64], sigma: &'a [f64]) -> Self {
        Self {
            rho,
            sigma: Some(sigma),
            lapl: None,
            tau: None,
        }
    }

    /// Attach a Laplacian field, for meta-GGAs that need it. Builder over
    /// [`XcInput::gga`] (e.g. `XcInput::gga(rho, sigma).with_lapl(lapl)`); since
    /// the struct is `#[non_exhaustive]`, this is the construction path for
    /// `lapl` rather than a struct literal.
    pub fn with_lapl(mut self, lapl: &'a [f64]) -> Self {
        self.lapl = Some(lapl);
        self
    }

    /// Attach a kinetic-energy-density field, required for meta-GGAs. Builder
    /// over [`XcInput::gga`] (e.g. `XcInput::gga(rho, sigma).with_tau(tau)`);
    /// since the struct is `#[non_exhaustive]`, this is the construction path for
    /// `tau` rather than a struct literal.
    pub fn with_tau(mut self, tau: &'a [f64]) -> Self {
        self.tau = Some(tau);
        self
    }
}

/// Energy per particle and first derivatives. Vectors are point-major; the ones
/// not produced by a given family are left empty.
///
/// `#[non_exhaustive]`: build via [`XcResult::default`] (the library fills it),
/// never a struct literal — higher derivative orders (`fxc`, …) and meta-GGA
/// fields can then be added without a breaking change.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct XcResult {
    /// XC energy per particle ε_xc, length `np`.
    pub exc: Vec<f64>,
    /// ∂(n·ε_xc)/∂n_σ, length `ns*np`.
    pub vrho: Vec<f64>,
    /// ∂(n·ε_xc)/∂σ, length `(2*ns-1)*np` for GGA/meta-GGA, else empty.
    pub vsigma: Vec<f64>,
    /// ∂(n·ε_xc)/∂τ_σ, meta-GGA only.
    pub vtau: Vec<f64>,
    /// ∂(n·ε_xc)/∂(∇²n_σ), meta-GGA-with-Laplacian only.
    pub vlapl: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂n_σ∂n_σ′ (`fxc`). Empty unless second order
    /// was requested via [`Functional::eval_fxc`](crate::Functional::eval_fxc).
    /// Length `np` unpolarized; `3*np` polarized, point-major `[aa, ab, bb]`.
    pub v2rho2: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂n_σ∂σ (`fxc`). Empty for LDA and unless
    /// second order was requested. Length `np` unpolarized; `6*np` polarized,
    /// point-major `[a_aa, a_ab, a_bb, b_aa, b_ab, b_bb]` (ρ-spin major).
    pub v2rhosigma: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂σ∂σ (`fxc`). Empty for LDA and unless second
    /// order was requested. Length `np` unpolarized; `6*np` polarized, point-major
    /// `[aa_aa, aa_ab, aa_bb, ab_ab, ab_bb, bb_bb]` (symmetric upper triangle).
    pub v2sigma2: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂n_σ∂τ_σ′ (`fxc`, meta-GGA only). Empty unless
    /// the functional is meta-GGA and second order was requested. Length `np`
    /// unpolarized; `4*np` polarized, point-major `[a_τa, a_τb, b_τa, b_τb]`
    /// (ρ-spin major × τ-spin minor — libxc `xc.h` ordering).
    pub v2rhotau: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂σ∂τ_σ (`fxc`, meta-GGA only). Empty unless
    /// meta-GGA and second order was requested. Length `np` unpolarized; `6*np`
    /// polarized, point-major `[aa_τa, aa_τb, ab_τa, ab_τb, bb_τa, bb_τb]`
    /// (σ major × τ-spin minor).
    pub v2sigmatau: Vec<f64>,
    /// Second derivative ∂²(n·ε_xc)/∂τ_σ∂τ_σ′ (`fxc`, meta-GGA only). Empty unless
    /// meta-GGA and second order was requested. Length `np` unpolarized; `3*np`
    /// polarized, point-major `[τa_τa, τa_τb, τb_τb]` (symmetric upper triangle).
    pub v2tau2: Vec<f64>,
}
