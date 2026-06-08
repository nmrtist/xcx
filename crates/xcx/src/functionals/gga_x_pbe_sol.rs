// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PBE-for-solids exchange (PBEsol) — `gga_x_pbe_sol` (libxc 116).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_x_pbe.mpl`
//! (`pbe_f`, `pbe_sol_values`) + `maple/util.mpl` (`gga_exchange`, `lda_x_spin`,
//! `X2S`, `MU_GE`).
//!
//! PBEsol (Perdew et al. 2008) is PBE exchange with a **single constant changed**:
//! μ is restored to the gradient-expansion value `MU_GE = 10/81` (vs PBE's
//! `MU_PBE ≈ 0.2195`), while κ = 0.8040 is unchanged (libxc
//! `pbe_sol_values = {0.804, MU_GE}`). The rational enhancement `F_x` and the
//! GGA-exchange skeleton are otherwise identical to
//! [`gga_x_pbe`](super::gga_x_pbe), so this reuses the shared, sqrt-free
//! [`pbe_enhancement`] with μ swapped — no forked math (CLAUDE.md §2/§3 reuse rule;
//! recovery test [`tests::mu_pbe_recovers_pbe_x`]).

use num_dual::DualNum;

use super::gga_x_pbe::{pbe_enhancement, KAPPA};
use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::X2S;

/// PBEsol's gradient-expansion μ (libxc `MU_GE = 10/81`). κ is PBE's 0.8040
/// (shared [`KAPPA`]).
const MU_GE: f64 = 10.0 / 81.0;
/// `μ·X2S²` for PBEsol-x: the coefficient of the **squared** reduced gradient `x²`
/// in `κ + μs²` (`s = X2S·x`), sqrt-free, with `MU_GE`/`X2S` kept exact.
const MU_GE_X2S2: f64 = MU_GE * X2S * X2S;

// Compile-time guard for the reuse (CLAUDE.md §2/§3): PBEsol-x restores the smaller
// gradient-expansion μ, so its coefficient must sit strictly below PBE's MU_X2S2 —
// the entire point of the swap. (A would-be edit that equated them fails to compile.)
const _: () = assert!(MU_GE_X2S2 < super::gga_x_pbe::MU_X2S2);

pub(crate) struct GgaXPbeSol {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaXPbeSol {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaXPbeSol),
                name: "gga_x_pbe_sol",
                family: Family::Gga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15, // libxc gga_x_pbe_sol (same as gga_x_pbe)
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaXPbeSol {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // PBEsol-x = PBE-x with μ = 10/81 (κ unchanged): same shared GGA-exchange
        // skeleton + shared rational enhancement, only the μ constant differs.
        gga_exchange(&v, self.info.dens_threshold, self.zeta_threshold, |t| {
            pbe_enhancement(t, KAPPA, MU_GE_X2S2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functionals::gga_x_pbe::{MU, MU_X2S2};
    use crate::reduced::consts::X2S;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pbesol(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXPbeSol, spin).unwrap()
    }

    /// Reuse recovery (CLAUDE.md §2/§3): PBEsol-x swaps only μ in the shared
    /// [`pbe_enhancement`]. At PBE's μ = `MU_PBE` the shared enhancement must
    /// reproduce PBE-x's `pbe_f0 = 1 + κμs²/(κ + μs²)` exactly (`s = X2S·x`,
    /// `t = x²`), proving the μ parameterization didn't perturb PBE-x; and PBEsol-x
    /// (μ = 10/81 < `MU_PBE`) must genuinely differ from PBE-x at finite σ.
    #[test]
    fn mu_pbe_recovers_pbe_x() {
        for &t in &[0.0_f64, 1e-8, 0.01, 0.5, 3.0, 100.0, 1e4] {
            let s2 = X2S * X2S * t;
            let want = 1.0 + KAPPA * MU * s2 / (KAPPA + MU * s2);
            let got = pbe_enhancement(t, KAPPA, MU_X2S2);
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1.0),
                "pbe_enhancement({t}, μ=MU_PBE) = {got} vs PBE-x pbe_f0 {want}"
            );
            // PBEsol's smaller μ keeps F_x below PBE-x's (∂F/∂μ ≥ 0; `<=` since for
            // tiny t the difference underflows f64 against F ≈ 1).
            let got_sol = pbe_enhancement(t, KAPPA, MU_GE_X2S2);
            assert!(got_sol <= want, "PBEsol-x F_x must be ≤ PBE-x at t={t}");
        }
        // At a resolvable gradient PBEsol-x strictly trails PBE-x (μ swapped, not κ).
        let t = 1.0_f64;
        assert!(
            pbe_enhancement(t, KAPPA, MU_GE_X2S2) < pbe_enhancement(t, KAPPA, MU_X2S2),
            "PBEsol-x F_x must be strictly below PBE-x at t=1"
        );
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = pbesol(Spin::Unpolarized);
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

    /// At σ = 0 the enhancement F_x → 1 (μ-independent), so PBEsol-x recovers
    /// Slater (lda_x) for energy and potential — the GGA→LDA limit.
    #[test]
    fn sigma_zero_recovers_lda_x() {
        let pu = pbesol(Spin::Unpolarized);
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
        let f = pbesol(Spin::Polarized);
        let out = f
            .eval(1, &XcInput::gga(&[0.6, 0.3], &[0.1, 0.05, 0.08]))
            .unwrap();
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
    }
}
