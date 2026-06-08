// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! RPBE exchange — `gga_x_rpbe` (libxc 117).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_x_rpbe.mpl`
//! (`rpbe_f`, `rpbe_values`) + `maple/util.mpl` (`gga_exchange`, `lda_x_spin`,
//! `X2S`).
//!
//! RPBE (Hammer, Hansen & Nørskov 1999) uses PBE's constants (κ = 0.8040,
//! μ = `MU_PBE`) but a **different functional form** — an *exponential* enhancement
//! rather than PBE's rational one:
//! ```text
//! F_x(s) = 1 + κ·(1 − exp(−μ·s²/κ)),   s = X2S·x.
//! ```
//! libxc writes it `rpbe_f0 = 1 + κ·(−xc_expm1(−μs²/κ))`; `expm1` keeps the
//! small-`s` limit (`F → 1`) cancellation-free. It is **naturally sqrt-free** — a
//! function of `s² = X2S²·x²` only, which the harness already carries squared
//! ([`GgaVars::xs0_sq`](crate::families::gga::GgaVars)) — and `exp` is entire, so
//! vxc *and* fxc are σ = 0-clean with no special handling, exactly like PBE-x. κ
//! and μ·X2S² are the very PBE-x literals ([`KAPPA`], [`MU_X2S2`]), reused rather
//! than re-declared (CLAUDE.md §2/§3); only the enhancement *form* is new.

use num_dual::DualNum;

use super::gga_x_pbe::{KAPPA, MU_X2S2};
use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};

/// RPBE exchange enhancement `F_x` as a function of the **squared** reduced
/// gradient `t = x²` (`s² = X2S²·t`). libxc's `rpbe_f0 = 1 + κ·(−expm1(−μs²/κ))`
/// `= 1 − κ·expm1(−μs²/κ) = 1 + κ(1 − exp(−μs²/κ))`. Sqrt-free (only `t = x²`
/// enters) and `exp` is entire, so `F(0) = 1` exactly and every derivative is
/// finite through σ = 0. `μs²/κ = (μ·X2S²)·t/κ` reuses PBE-x's [`MU_X2S2`] and
/// [`KAPPA`]. Provenance: ported-from-libxc (MPL-2.0), `maple/gga_exc/gga_x_rpbe.mpl`.
fn rpbe_enhancement<N: DualNum<f64> + Copy>(t: N) -> N {
    // arg = −μs²/κ = −(μ·X2S²/κ)·t; F = 1 − κ·expm1(arg)
    let arg = N::from(-MU_X2S2 / KAPPA) * t;
    N::from(1.0) - N::from(KAPPA) * arg.exp_m1()
}

pub(crate) struct GgaXRpbe {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaXRpbe {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaXRpbe),
                name: "gga_x_rpbe",
                family: Family::Gga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15, // libxc gga_x_rpbe
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaXRpbe {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // RPBE = per-channel LDA exchange × the exponential enhancement, screened
        // on the floored spin density (shared `gga_exchange` skeleton).
        gga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            rpbe_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functionals::gga_x_pbe::MU;
    use crate::reduced::consts::X2S;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn rpbe(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXRpbe, spin).unwrap()
    }

    /// The enhancement must equal the closed form `1 + κ(1 − exp(−μs²/κ))`
    /// (`s = X2S·x`, `t = x²`) with κ = 0.8040, μ = `MU_PBE`. `F(0) = 1` exactly;
    /// for small s it grows ~ μs² (like PBE-x) and asymptotes to `1 + κ`.
    #[test]
    fn enhancement_matches_closed_form() {
        let kappa = KAPPA;
        for &t in &[0.0_f64, 1e-8, 0.01, 0.5, 3.0, 100.0, 1e4] {
            let s2 = X2S * X2S * t;
            let want = 1.0 + kappa * (1.0 - (-MU * s2 / kappa).exp());
            let got: f64 = rpbe_enhancement(t);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "rpbe_enhancement({t}) = {got} vs closed form {want}"
            );
        }
        assert_eq!(rpbe_enhancement(0.0_f64), 1.0, "F(0) must be exactly 1");
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = rpbe(Spin::Unpolarized);
        let edens = |n: f64, s: f64| n * f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
        for &(n, s) in &[(0.5, 0.1), (2.0, 0.7), (0.1, 0.02), (10.0, 5.0)] {
            let out = f.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
            let hn = 1e-6 * n;
            let hs = 1e-6 * s;
            let fdn = (edens(n + hn, s) - edens(n - hn, s)) / (2.0 * hn);
            let fds = (edens(n, s + hs) - edens(n, s - hs)) / (2.0 * hs);
            assert!((out.vrho[0] - fdn).abs() <= 1e-6 * out.vrho[0].abs().max(1.0));
            assert!((out.vsigma[0] - fds).abs() <= 1e-6 * out.vsigma[0].abs().max(1.0));
        }
    }

    /// At σ = 0 the enhancement F_x → 1, so RPBE recovers Slater (lda_x) for energy
    /// and potential — the GGA→LDA limit.
    #[test]
    fn sigma_zero_recovers_lda_x() {
        let pu = rpbe(Spin::Unpolarized);
        let lu = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let p = pu.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap();
            let l = lu.eval(1, &XcInput::lda(&[n])).unwrap();
            assert!((p.exc[0] - l.exc[0]).abs() <= 1e-10 * l.exc[0].abs());
            assert!((p.vrho[0] - l.vrho[0]).abs() <= 1e-10 * l.vrho[0].abs());
        }
    }

    /// fxc must be finite even at exactly σ = 0 — the exp form is entire, so no
    /// special handling is needed (the sqrt-free σ = 0 cleanliness, like PBE-x).
    #[test]
    fn fxc_finite_at_sigma_zero() {
        let f = rpbe(Spin::Unpolarized);
        for &n in &[0.1, 1.0, 50.0] {
            let r = f.eval_fxc(1, &XcInput::gga(&[n], &[0.0])).unwrap();
            assert!(r.v2rho2[0].is_finite() && r.v2sigma2[0].is_finite());
            assert!(r.v2rhosigma[0].is_finite());
        }
    }

    /// Pure exchange has no σ_ab dependence: ∂e/∂σ_ab must be exactly zero.
    #[test]
    fn pol_no_sigma_ab_dependence() {
        let f = rpbe(Spin::Polarized);
        let out = f
            .eval(1, &XcInput::gga(&[0.6, 0.3], &[0.1, 0.05, 0.08]))
            .unwrap();
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
    }
}
