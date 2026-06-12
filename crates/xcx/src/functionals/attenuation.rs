// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Short-range (erf-screened) exchange attenuation — the Chai–Head-Gordon /
//! Tawada SR-LDA kernel used by the ωB97 family's semilocal exchange.
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/attenuation.mpl`
//! (`attenuation_erf`, cutoff/order from `check_attenuation.mpl`) +
//! `maple/lda_exc/lda_x_erf.mpl`. References: J. Toulouse, F. Colonna,
//! A. Savin, Int. J. Quantum Chem. 100, 1047 (2004); Y. Tawada et al.,
//! J. Chem. Phys. 120, 8425 (2004) (the implementation follows Tawada).
//!
//! The SR-erf LDA exchange per spin channel is the full-range LDA exchange
//! times the attenuation factor `F(a)`, `a = ω/(2 k_F,σ)`; libxc evaluates the
//! exact closed form below a cutoff and a large-`a` asymptotic series above it
//! (`enforce_smooth_lr(f, a, 1.35, 16)`), because the closed form cancels
//! catastrophically as `a → ∞` (`F(a) → 1/(36 a²)` from differences of
//! near-unit terms). xcx reproduces both branches — including the emitted
//! series coefficients and the branch-selection-on-the-real-part semantics —
//! so golden parity and finite `fxc` hold across the seam.
//!
//! **`erf` for dual numbers.** `num-dual` has no `erf`, so [`erf_dual`]
//! reconstructs it from the nilpotent structure of forward-mode duals: for any
//! first- or second-order dual `x = x₀ + δ` (`δ` nilpotent, `δ³ = 0`), the
//! truncated Taylor series `erf(x₀) + erf′(x₀)·δ + ½·erf″(x₀)·δ²` is **exact**
//! — no approximation beyond f64 `erf` itself (`libm::erf`, ≤ 1 ulp from the C
//! library libxc calls). `erf′(x) = 2/√π·e^{−x²}`, `erf″(x) = −2x·erf′(x)`.

use num_dual::DualNum;

/// `a_cnst/ω = (4/(9π))^(1/3)/2`: the prefactor mapping `rs/(1±z)^(1/3)` to the
/// attenuation argument `a = a_cnst·ω·rs/opz^(1/3)` (`lda_x_erf.mpl` `a_cnst`).
pub(crate) fn att_a_cnst(omega: f64) -> f64 {
    (4.0 / (9.0 * std::f64::consts::PI)).cbrt() * omega / 2.0
}

/// `erf(x)` for a (possibly dual) scalar via the exact nilpotent Taylor
/// reconstruction (see module docs). For plain `f64` this is just `libm::erf`.
pub(crate) fn erf_dual<N: DualNum<f64> + Copy>(x: N) -> N {
    let x0 = x.re();
    let d = x - N::from(x0); // nilpotent dual part: d³ = 0 exactly
    let e0 = libm::erf(x0);
    let d1 = 2.0 / std::f64::consts::PI.sqrt() * (-x0 * x0).exp();
    let d2 = -2.0 * x0 * d1;
    N::from(e0) + d * d1 + d * d * (0.5 * d2)
}

/// Branch cutoff of libxc's `enforce_smooth_lr(attenuation_erf0, a, 1.35, 16)`.
const A_CUTOFF: f64 = 1.35;

/// Large-`a` asymptotic series of `attenuation_erf0`, exactly the terms libxc's
/// maple2c output carries (order 16 in `1/a`, even powers only; the generated C
/// also carries `expm1(0) = 0` vestiges, dropped here as exact zeros).
const LARGE_A_COEFFS: [f64; 8] = [
    1.0 / 36.0,
    -1.0 / 960.0,
    1.0 / 26880.0,
    -1.0 / 829440.0,
    1.0 / 28385280.0,
    -1.0 / 1073479680.0,
    1.0 / 44590694400.0,
    -1.0 / 2.0214448128e13,
];

/// The erf-screened exchange attenuation `F(a)` (`attenuation.mpl`
/// `attenuation_erf`), AD-safe on both branches:
///
/// - `a < 1.35` — the exact Tawada closed form
///   `1 − 8/3·a·(√π·erf(1/(2a)) + 2a·(m − 2a²m − ½))`, `m = expm1(−1/(4a²))`.
///   As `a → 0⁺` the `e^{−1/(4a²)}` factors underflow cleanly to 0, so the
///   value → 1 and every AD derivative stays finite (`a > 0` always: it is
///   built from the floored density).
/// - `a ≥ 1.35` — the asymptotic series `Σ c_k/a^(2k)`, k = 1..8, evaluated by
///   Horner in `1/a²`; cancellation-free where the closed form is not.
///
/// Branch choice and the seam clamps follow libxc's emitted
/// `my_piecewise3` semantics on the **real part** (at `a == 1.35` exactly the
/// series is evaluated at the constant 1.35, derivative 0 through the clamp),
/// so derivatives match libxc's at and around the seam.
pub(crate) fn attenuation_erf<N: DualNum<f64> + Copy>(a: N) -> N {
    if a.re() >= A_CUTOFF {
        // libxc: f_large(m_max(a, cutoff)) — the max is the identity for
        // a > cutoff and pins the constant at the exact seam.
        let b = if a.re() > A_CUTOFF {
            a
        } else {
            N::from(A_CUTOFF)
        };
        let inv2 = (b * b).recip();
        let mut p = N::from(LARGE_A_COEFFS[7]);
        for &c in LARGE_A_COEFFS[..7].iter().rev() {
            p = p * inv2 + N::from(c);
        }
        p * inv2
    } else {
        // libxc: f(m_min(a, cutoff)) — the min is the identity on this branch.
        let a2 = a * a;
        let m = (-(a2 * 4.0).recip()).exp_m1(); // expm1(−1/(4a²))
        let inner = m - a2 * m * 2.0 - 0.5; // aux2 − aux3
        let bracket = erf_dual((a * 2.0).recip()) * std::f64::consts::PI.sqrt() + a * inner * 2.0;
        N::from(1.0) - a * bracket * (8.0 / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_dual::{first_derivative, second_derivative};

    /// Both branches agree to the series' truncation accuracy at the seam, and
    /// the function is smooth/finite across it (value, 1st, 2nd derivative).
    #[test]
    fn branches_agree_at_cutoff_and_are_finite() {
        // closed form vs series at the cutoff
        let small = {
            let a = A_CUTOFF - 1e-12;
            attenuation_erf(a)
        };
        let large = attenuation_erf(A_CUTOFF + 1e-12);
        assert!(
            (small - large).abs() < 1e-11,
            "seam mismatch: {small} vs {large}"
        );
        for &a in &[
            1e-6, 1e-3, 0.1, 0.5, 1.0, 1.349, 1.35, 1.351, 2.0, 10.0, 1e3,
        ] {
            let (v, d1, d2) = second_derivative(attenuation_erf, a);
            assert!(v.is_finite() && d1.is_finite() && d2.is_finite(), "a={a}");
            assert!(
                (0.0..=1.0 + 1e-12).contains(&v),
                "F({a}) = {v} out of [0,1]"
            );
        }
    }

    /// Known limits: F(a→0) → 1 (full-range LDA recovered), F(a→∞) ~ 1/(36a²).
    #[test]
    fn limits() {
        // F(a) = 1 − (8/√(9π))·a + O(a²) as a → 0 (slope −8/3·√π in this form)
        let a = 1e-8;
        let want = 1.0 - 8.0 / 3.0 * a * std::f64::consts::PI.sqrt();
        assert!((attenuation_erf(a) - want).abs() < 1e-13);
        let a = 1e4_f64;
        let want = 1.0 / (36.0 * a * a) - 1.0 / (960.0 * a.powi(4));
        let got = attenuation_erf(a);
        assert!((got - want).abs() <= 1e-12 * want.abs(), "{got} vs {want}");
    }

    /// `erf_dual` reproduces libm's erf and its analytic derivative.
    #[test]
    fn erf_dual_matches_value_and_derivative() {
        for &x in &[-2.0, -0.5, 0.0, 0.3, 1.0, 4.0] {
            let (v, d) = first_derivative(erf_dual, x);
            assert_eq!(v, libm::erf(x));
            let want = 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp();
            assert!((d - want).abs() <= 1e-15, "erf'({x}): {d} vs {want}");
        }
    }

    /// AD self-consistency: dual derivative of F vs central finite difference,
    /// on both branches.
    #[test]
    fn derivative_matches_finite_difference() {
        for &a in &[0.05, 0.3, 0.9, 1.2, 1.5, 3.0, 20.0] {
            let (_, d1) = first_derivative(attenuation_erf, a);
            let h = 1e-6 * a;
            let fd = (attenuation_erf(a + h) - attenuation_erf(a - h)) / (2.0 * h);
            assert!(
                (d1 - fd).abs() <= 1e-6 * d1.abs().max(1e-10),
                "a={a}: {d1} vs {fd}"
            );
        }
    }
}
