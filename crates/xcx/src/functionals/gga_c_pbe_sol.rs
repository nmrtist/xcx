// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PBE-for-solids correlation (PBEsol) — `gga_c_pbe_sol` (libxc 133).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_c_pbe.mpl`
//! (`pbe_sol_values`) + `maple/lda_exc/lda_c_pw.mpl` + `maple/util.mpl`.
//!
//! PBEsol correlation (Perdew et al. 2008) is PBE correlation with a **single
//! constant changed**: β drops from PBE's 0.06672… to 0.046, while γ, the
//! `BB`/`tscale` folds, and the **modified-PW92 uniform limit**
//! (`pw92_ec(&A_MOD, FPP_VWN)`) are all unchanged (libxc
//! `pbe_sol_values = {0.046, γ, 1, 1}`). β enters only the gradient correction `H`,
//! so this reuses the shared [`pbe_c_energy`] with β swapped — no forked math
//! (CONTRIBUTING.md reuse rule; recovery test [`tests::beta_pbe_recovers_gga_c_pbe`]).

use num_dual::DualNum;

use super::gga_c_pbe::pbe_c_energy;
use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};

/// PBEsol correlation β (libxc `pbe_sol_values` `_beta` = 0.046). γ, BB, tscale,
/// and the modified-PW92 uniform limit are PBE's, shared inside [`pbe_c_energy`].
const BETA_SOL: f64 = 0.046;

// Compile-time guard for the reuse (CONTRIBUTING.md reuse rule): β is the only constant
// PBEsol-c swaps, and it lowers β below PBE-c's. (An edit equating them won't compile.)
const _: () = assert!(BETA_SOL < super::gga_c_pbe::BETA);

pub(crate) struct GgaCPbeSol {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaCPbeSol {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaCPbeSol),
                name: "gga_c_pbe_sol",
                family: Family::Gga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-12, // libxc gga_c_pbe_sol (same as gga_c_pbe)
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaCPbeSol {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // PBEsol-c = PBE-c with β = 0.046 (γ + modified-PW92 limit unchanged), via
        // the shared free function — the single source of the PBE-correlation math.
        pbe_c_energy(v.rs, v.z, v.xt2, self.zeta_threshold, BETA_SOL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functionals::gga_c_pbe::BETA;
    use crate::reduced::vars::{reduced_grad_sq, rs_from_n};
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pbesol(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaCPbeSol, spin).unwrap()
    }

    /// Reuse recovery (CONTRIBUTING.md reuse rule): PBEsol-c swaps only β in the shared
    /// [`pbe_c_energy`]. At PBE's β = `BETA` the shared free function must reproduce
    /// `gga_c_pbe`'s energy per particle exactly (the unpolarized harness returns
    /// `exc = f`), proving the β parameterization didn't perturb PBE-c; and PBEsol-c
    /// (β = 0.046) must genuinely *differ* from PBE-c at finite σ (β only enters the
    /// gradient correction `H`).
    #[test]
    fn beta_pbe_recovers_gga_c_pbe() {
        let pbe = Functional::new(FunctionalId::GgaCPbe, Spin::Unpolarized).unwrap();
        let sol = pbesol(Spin::Unpolarized);
        let zt = f64::EPSILON;
        for &(n, s) in &[(0.5, 0.1), (1.0, 0.3), (2.0, 1.5), (0.3, 0.02)] {
            let rs = rs_from_n(n);
            let xt2 = reduced_grad_sq(s, n); // unpolarized: σ_tot = σ, total n
            let got = pbe_c_energy(rs, 0.0_f64, xt2, zt, BETA);
            let want = pbe.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "pbe_c_energy(β=BETA) {got} vs gga_c_pbe exc {want} @ n={n}, σ={s}"
            );
            // PBEsol-c (β = 0.046) differs from PBE-c at finite σ.
            let solv = sol.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
            assert!(
                (solv - want).abs() > 1e-9 * want.abs(),
                "PBEsol-c unexpectedly equals PBE-c @ n={n}, σ={s}"
            );
        }
    }

    /// At σ = 0 the gradient correction `H → 0` (β-independent), so PBEsol-c
    /// recovers the **same** modified-PW92 uniform limit as PBE-c — β only enters
    /// `H`. So PBEsol-c(σ=0) == PBE-c(σ=0).
    #[test]
    fn sigma_zero_matches_pbe_c_uniform_limit() {
        let sol = pbesol(Spin::Unpolarized);
        let pbe = Functional::new(FunctionalId::GgaCPbe, Spin::Unpolarized).unwrap();
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let s = sol.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap().exc[0];
            let p = pbe.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap().exc[0];
            assert!(
                (s - p).abs() <= 1e-12 * p.abs(),
                "PBEsol-c(σ=0) {s} vs PBE-c(σ=0) {p} @ n={n}"
            );
        }
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

    /// Correlation depends on σ_ab through the total gradient x_t, so all three
    /// vsigma components are nonzero (unlike pure exchange).
    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = pbesol(Spin::Polarized);
        let (na, nb, saa, sab, sbb) = (0.6, 0.3, 0.1, 0.05, 0.08);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let edens = |r: [f64; 2], s: [f64; 3]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s)).unwrap();
        for (k, h) in [(0usize, 1e-6 * saa), (1, 1e-6 * sab), (2, 1e-6 * sbb)] {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp) - edens(r, sm)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-6 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
    }
}
