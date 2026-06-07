// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Slater (Dirac) exchange — `lda_x` (libxc 1).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/lda_exc/lda_x.mpl` + `util.mpl`.
//!
//! The energy per particle is the sum of the two spin-channel contributions,
//! `f = α·[ S(rs, z) + S(rs, −z) ]`, each screened independently, where `S` is
//! `lda_x_spin`. For `α = 1` this is Dirac/Slater exchange.

use num_dual::DualNum;

use crate::families::lda::{Lda, LdaEnergy, LdaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::vars::{lda_x_spin, screen_dens};

pub(crate) struct LdaX {
    info: FunctionalInfo,
    /// Xα scaling parameter; `1.0` for Dirac/Slater exchange.
    alpha: f64,
    zeta_threshold: f64,
}

impl LdaX {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::LdaX),
                name: "lda_x",
                family: Family::Lda,
                kind: Kind::Exchange,
                needs_sigma: false,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            alpha: 1.0,
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Lda(Self::new()))
    }
}

impl LdaEnergy for LdaX {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
        let dt = self.info.dens_threshold;
        let zt = self.zeta_threshold;
        // Up channel uses opz = 1+z, down channel uses omz = 1−z (both
        // cancellation-free), so vrho stays accurate at full polarization.
        let up = if screen_dens(v.na, dt) {
            N::from(0.0)
        } else {
            lda_x_spin(v.rs, v.opz, zt)
        };
        let dn = if screen_dens(v.nb, dt) {
            N::from(0.0)
        } else {
            lda_x_spin(v.rs, v.omz, zt)
        };
        (up + dn) * N::from(self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};
    use std::f64::consts::PI;

    /// Analytic unpolarized Slater exchange: ε_x = −(3/4)(3/π)^(1/3) n^(1/3).
    fn slater_eps(n: f64) -> f64 {
        -0.75 * (3.0 / PI).powf(1.0 / 3.0) * n.cbrt()
    }

    #[test]
    fn unpol_matches_analytic_slater() {
        let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        let rho = [1e-3, 0.1, 1.0, 7.3, 1e3];
        let out = f.eval(rho.len(), &XcInput::lda(&rho)).unwrap();
        for (i, &n) in rho.iter().enumerate() {
            let eps = slater_eps(n);
            let vx = 4.0 / 3.0 * eps; // v = d(n·ε)/dn = (4/3)·ε for ε ∝ n^(1/3)
            assert!(
                (out.exc[i] - eps).abs() <= 1e-12 * eps.abs(),
                "exc(n={n}) = {} vs {eps}",
                out.exc[i]
            );
            assert!(
                (out.vrho[i] - vx).abs() <= 1e-12 * vx.abs(),
                "vrho(n={n}) = {} vs {vx}",
                out.vrho[i]
            );
        }
    }

    #[test]
    fn unpol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        let edens = |x: f64| x * f.eval(1, &XcInput::lda(&[x])).unwrap().exc[0];
        for &n in &[0.05, 0.5, 3.0] {
            let h = 1e-6 * n;
            let fd = (edens(n + h) - edens(n - h)) / (2.0 * h);
            let v = f.eval(1, &XcInput::lda(&[n])).unwrap().vrho[0];
            assert!(
                (v - fd).abs() <= 1e-6 * v.abs().max(1.0),
                "n={n}: {v} vs fd {fd}"
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        let po = Functional::new(FunctionalId::LdaX, Spin::Polarized).unwrap();
        let n = 0.7;
        let ou = up.eval(1, &XcInput::lda(&[n])).unwrap();
        let op = po.eval(1, &XcInput::lda(&[n / 2.0, n / 2.0])).unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-13 * ou.exc[0].abs());
        // both spin potentials equal the unpolarized one at z = 0
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-12 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-12 * ou.vrho[0].abs());
    }

    #[test]
    fn edge_energy_and_derivatives_finite() {
        let f = Functional::new(FunctionalId::LdaX, Spin::Polarized).unwrap();
        // full polarization (z=+1, z=-1), tiny, large, and sub-threshold channels
        let rho = [
            1.0, 0.0, // z = +1
            0.0, 1.0, // z = -1
            1e-12, 1e-13, // small
            1e3, 1e2, // large
            5e-16, 5e-16, // both channels individually sub-threshold
        ];
        let out = f.eval(5, &XcInput::lda(&rho)).unwrap();
        for v in out.exc.iter().chain(&out.vrho) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
