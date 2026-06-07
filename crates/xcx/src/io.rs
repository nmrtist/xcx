// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Input/output data types. See `docs/api-convention.md` for packing rules.

/// Inputs to a functional evaluation, as borrowed point-major slices.
///
/// Packing (with `np` points, `ns = 1` unpolarized / `2` polarized): `rho` has
/// `ns*np` entries (`[n]` or `[n_a, n_b]` per point), `sigma` has `(2*ns-1)*np`
/// (`[σ]` or `[σ_aa, σ_ab, σ_bb]`), `lapl`/`tau` have `ns*np`.
#[derive(Debug, Clone, Copy)]
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
}

/// Energy per particle and first derivatives. Vectors are point-major; the ones
/// not produced by a given family are left empty.
#[derive(Debug, Clone, Default, PartialEq)]
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
}
