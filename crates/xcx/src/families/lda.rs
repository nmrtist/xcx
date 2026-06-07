// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! LDA family: energy trait, autodiff harness, object-safe wrapper.

use nalgebra::SVector;
use num_dual::{gradient, DualNum, DualSVec64};

use super::{check_len, XcEval};
use crate::error::XcError;
use crate::func::{FunctionalInfo, Spin};
use crate::io::{XcInput, XcResult};
use crate::reduced::vars;

/// Reduced variables passed to an LDA energy expression. `opz`/`omz` are the
/// cancellation-free spin factors `1 ± z` (`= 2·n_{a,b}/n`), so functionals stay
/// accurate at full spin polarization.
#[derive(Clone, Copy)]
pub(crate) struct LdaVars<N> {
    /// Wigner–Seitz radius.
    pub rs: N,
    /// Relative spin polarization `z = (n_a − n_b)/n`.
    pub z: N,
    /// `1 + z`, computed cancellation-free as `2·n_a/n`.
    pub opz: N,
    /// `1 − z`, computed cancellation-free as `2·n_b/n`.
    pub omz: N,
    /// Spin-up density (floored), for direct per-channel screening like libxc.
    pub na: N,
    /// Spin-down density (floored).
    pub nb: N,
}

/// An LDA functional's energy per particle, written generically over a
/// dual-number scalar. Mirrors libxc's `f := (rs, z) -> ...`.
pub(crate) trait LdaEnergy: Send + Sync {
    fn info(&self) -> &FunctionalInfo;
    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N;
}

/// Object-safe wrapper turning any [`LdaEnergy`] into an [`XcEval`].
pub(crate) struct Lda<F: LdaEnergy>(pub F);

impl<F: LdaEnergy> XcEval for Lda<F> {
    fn info(&self) -> &FunctionalInfo {
        self.0.info()
    }

    fn eval(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        match spin {
            Spin::Unpolarized => self.eval_unpol(np, input),
            Spin::Polarized => self.eval_pol(np, input),
        }
    }
}

impl<F: LdaEnergy> Lda<F> {
    fn eval_unpol(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        check_len(input.rho, np)?;
        let thr = self.0.info().dens_threshold;
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; np];
        for i in 0..np {
            let n = input.rho[i];
            if n < thr || n.is_nan() {
                continue; // strictly below threshold or non-finite: outputs stay 0
            }
            let nf = n.max(thr); // libxc floors the density to dens_threshold
            let (e, g) = gradient(
                |v: SVector<DualSVec64<1>, 1>| {
                    let n = v[0];
                    let rs = vars::rs_from_n(n);
                    let one = DualSVec64::<1>::from(1.0);
                    let zero = DualSVec64::<1>::from(0.0);
                    let half = n * DualSVec64::<1>::from(0.5);
                    n * self.0.f(LdaVars {
                        rs,
                        z: zero,
                        opz: one,
                        omz: one,
                        na: half,
                        nb: half,
                    })
                },
                &SVector::<f64, 1>::from([nf]),
            );
            exc[i] = e / nf;
            vrho[i] = g[0];
        }
        Ok(XcResult {
            exc,
            vrho,
            ..Default::default()
        })
    }

    fn eval_pol(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        check_len(input.rho, 2 * np)?;
        let thr = self.0.info().dens_threshold;
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; 2 * np];
        for i in 0..np {
            let na = input.rho[2 * i];
            let nb = input.rho[2 * i + 1];
            let n = na + nb;
            if n < thr || n.is_nan() {
                continue;
            }
            // libxc floors each spin density to dens_threshold and differentiates
            // w.r.t. the floored value (it does not chain through the max), so seed
            // the gradient at the floored densities. This is what makes the z=±1
            // exact-boundary points match (libxc evaluates n_b = 0 as 1e-15).
            let na_f = na.max(thr);
            let nb_f = nb.max(thr);
            let n_f = na_f + nb_f;
            let (e, g) = gradient(
                |v: SVector<DualSVec64<2>, 2>| {
                    let na = v[0];
                    let nb = v[1];
                    let n = na + nb;
                    let rs = vars::rs_from_n(n);
                    let z = (na - nb) / n;
                    let opz = (na + na) / n; // 1 + z, cancellation-free
                    let omz = (nb + nb) / n; // 1 − z, cancellation-free
                    n * self.0.f(LdaVars {
                        rs,
                        z,
                        opz,
                        omz,
                        na,
                        nb,
                    })
                },
                &SVector::<f64, 2>::from([na_f, nb_f]),
            );
            exc[i] = e / n_f;
            vrho[2 * i] = g[0];
            vrho[2 * i + 1] = g[1];
        }
        Ok(XcResult {
            exc,
            vrho,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::{Family, FunctionalId, Kind};

    struct DummyLda(FunctionalInfo);
    impl LdaEnergy for DummyLda {
        fn info(&self) -> &FunctionalInfo {
            &self.0
        }
        fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
            -v.rs.recip() // arbitrary smooth function for harness validation
        }
    }

    fn dummy() -> Lda<DummyLda> {
        Lda(DummyLda(FunctionalInfo {
            id: Some(FunctionalId::LdaX),
            name: "dummy_lda",
            family: Family::Lda,
            kind: Kind::Exchange,
            needs_sigma: false,
            needs_lapl: false,
            needs_tau: false,
            dens_threshold: 1e-15,
            hybrid: None,
        }))
    }

    #[test]
    fn unpol_runs_finite() {
        let f = dummy();
        let rho = [0.1, 0.5, 1.0];
        let out = f.eval(Spin::Unpolarized, 3, &XcInput::lda(&rho)).unwrap();
        assert_eq!(out.exc.len(), 3);
        assert_eq!(out.vrho.len(), 3);
        assert!(out.exc.iter().chain(&out.vrho).all(|v| v.is_finite()));
    }

    #[test]
    fn pol_runs_finite() {
        let f = dummy();
        let rho = [0.05, 0.05, 0.3, 0.2, 0.6, 0.4];
        let out = f.eval(Spin::Polarized, 3, &XcInput::lda(&rho)).unwrap();
        assert_eq!(out.vrho.len(), 6);
        assert!(out.vrho.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn below_threshold_is_zero() {
        let f = dummy();
        let rho = [0.0, 1e-300];
        let out = f.eval(Spin::Unpolarized, 2, &XcInput::lda(&rho)).unwrap();
        assert_eq!(out.exc, vec![0.0, 0.0]);
        assert_eq!(out.vrho, vec![0.0, 0.0]);
    }
}
