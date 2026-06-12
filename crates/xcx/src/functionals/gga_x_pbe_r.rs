// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Revised PBE exchange (revPBE) — `gga_x_pbe_r` (libxc 102).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_x_pbe.mpl`
//! (`pbe_f`, `pbe_r_values`) + `maple/util.mpl` (`gga_exchange`, `lda_x_spin`,
//! `X2S`).
//!
//! revPBE (Zhang & Yang 1998) is PBE exchange with a **single constant changed**:
//! the asymptotic enhancement bound κ rises from PBE's 0.8040 to 1.245, while
//! μ = `MU_PBE` is unchanged (libxc `pbe_r_values = {1.245, MU_PBE}`). The rational
//! enhancement `F_x` and the entire GGA-exchange skeleton are otherwise identical
//! to [`gga_x_pbe`](super::gga_x_pbe), so this reuses the shared, sqrt-free
//! [`pbe_enhancement`] with κ swapped — no forked math (CONTRIBUTING.md reuse rule;
//! recovery test [`tests::kappa_804_recovers_pbe_x`]).

use num_dual::DualNum;

use super::gga_x_pbe::{pbe_enhancement, MU_X2S2};
use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};

/// revPBE's enhancement bound κ (libxc `pbe_r_values` = `{1.245, MU_PBE}`). μ is
/// PBE's `MU_PBE`, carried via the shared [`MU_X2S2`] (`μ·X2S²`).
const KAPPA_R: f64 = 1.245;

pub(crate) struct GgaXPbeR {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaXPbeR {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaXPbeR),
                name: "gga_x_pbe_r",
                family: Family::Gga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15, // libxc gga_x_pbe_r (same as gga_x_pbe)
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaXPbeR {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // revPBE = PBE-x with κ = 1.245 (μ unchanged): same shared GGA-exchange
        // skeleton + shared rational enhancement, only the κ constant differs.
        gga_exchange(&v, self.info.dens_threshold, self.zeta_threshold, |t| {
            pbe_enhancement(t, KAPPA_R, MU_X2S2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functionals::gga_x_pbe::{KAPPA, MU};
    use crate::reduced::consts::X2S;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn revpbe(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXPbeR, spin).unwrap()
    }

    /// Reuse recovery (CONTRIBUTING.md reuse rule): revPBE swaps only κ in the shared
    /// [`pbe_enhancement`]. At PBE's κ = 0.8040 the shared enhancement must
    /// reproduce PBE-x's `pbe_f0 = 1 + κμs²/(κ + μs²)` exactly (`s = X2S·x`,
    /// `t = x²`), proving the κ parameterization didn't perturb PBE-x; and revPBE
    /// (κ = 1.245) must genuinely differ from PBE-x at finite σ.
    #[test]
    fn kappa_804_recovers_pbe_x() {
        for &t in &[0.0_f64, 1e-8, 0.01, 0.5, 3.0, 100.0, 1e4] {
            let s2 = X2S * X2S * t;
            let want = 1.0 + KAPPA * MU * s2 / (KAPPA + MU * s2);
            let got = pbe_enhancement(t, KAPPA, MU_X2S2);
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1.0),
                "pbe_enhancement({t}, κ=0.804) = {got} vs PBE-x pbe_f0 {want}"
            );
            // revPBE's larger κ raises F_x above PBE-x's (∂F/∂κ = μ²s⁴/(κ+μs²)² ≥ 0;
            // `>=` since for tiny t the difference underflows f64 against F ≈ 1).
            let got_r = pbe_enhancement(t, KAPPA_R, MU_X2S2);
            assert!(got_r >= want, "revPBE F_x must be ≥ PBE-x at t={t}");
        }
        // At a resolvable gradient revPBE strictly exceeds PBE-x (κ swapped, not μ).
        let t = 1.0_f64;
        assert!(
            pbe_enhancement(t, KAPPA_R, MU_X2S2) > pbe_enhancement(t, KAPPA, MU_X2S2),
            "revPBE F_x must strictly exceed PBE-x at t=1"
        );
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = revpbe(Spin::Unpolarized);
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

    /// At σ = 0 the enhancement F_x → 1 (κ-independent), so revPBE recovers Slater
    /// (lda_x) for energy and potential — the GGA→LDA limit.
    #[test]
    fn sigma_zero_recovers_lda_x() {
        let pu = revpbe(Spin::Unpolarized);
        let lu = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let p = pu.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap();
            let l = lu.eval(1, &XcInput::lda(&[n])).unwrap();
            assert!((p.exc[0] - l.exc[0]).abs() <= 1e-10 * l.exc[0].abs());
            assert!((p.vrho[0] - l.vrho[0]).abs() <= 1e-10 * l.vrho[0].abs());
        }
    }

    /// Pure exchange has no σ_ab dependence: ∂e/∂σ_ab must be exactly zero.
    #[test]
    fn pol_no_sigma_ab_dependence() {
        let f = revpbe(Spin::Polarized);
        let out = f
            .eval(1, &XcInput::gga(&[0.6, 0.3], &[0.1, 0.05, 0.08]))
            .unwrap();
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
    }
}
