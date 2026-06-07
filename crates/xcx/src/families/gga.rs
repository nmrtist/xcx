// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! GGA family: energy trait, autodiff harness, object-safe wrapper.
//!
//! The harness mirrors libxc's `work_gga` exactly (golden-locked): screen on the
//! total density (strict `<`), floor each spin density to `dens_threshold` and
//! each σ_aa/σ_bb to `sigma_threshold²`, clamp σ_ab to `[−s_ave, +s_ave]` with
//! `s_ave = ½(σ_aa+σ_bb)` (floored), then seed forward-AD at those floored values
//! so `vrho`/`vsigma` are derivatives w.r.t. the floored inputs — as libxc does.

use nalgebra::SVector;
use num_dual::{gradient, DualNum, DualSVec64};

use super::{check_len, XcEval};
use crate::error::XcError;
use crate::func::{FunctionalInfo, Spin};
use crate::io::{XcInput, XcResult};
use crate::reduced::vars;

/// Reduced variables passed to a GGA energy expression. As in [`super::lda`],
/// `opz`/`omz` are the cancellation-free spin factors `1 ± z` (`= 2·n_{a,b}/n`),
/// matching libxc's exchange arithmetic at full spin polarization. `xt2` is the
/// **squared** total reduced gradient (sqrt-free for AD-safety at σ_tot = 0);
/// `xs0`/`xs1` are the per-spin reduced gradients (`√` of *floored* spin σ, > 0).
#[derive(Clone, Copy)]
pub(crate) struct GgaVars<N> {
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
    /// Squared total reduced gradient `x_t² = (σ_aa+2σ_ab+σ_bb)/n^(8/3)`. Squared
    /// (not the magnitude) so forward-AD stays finite when σ_tot → 0: the σ_ab
    /// clamp can make σ_tot exactly 0, where `√σ_tot`'s derivative diverges.
    pub xt2: N,
    /// Spin-up reduced gradient `√σ_aa/n_a^(4/3)`.
    pub xs0: N,
    /// Spin-down reduced gradient `√σ_bb/n_b^(4/3)`.
    pub xs1: N,
}

/// GGA-exchange skeleton shared by every GGA exchange functional — the Rust
/// mirror of libxc's `util.mpl` `gga_exchange(func, rs, z, xs0, xs1)`. Each spin
/// channel is the LDA exchange of that channel (`lda_x_spin`, cancellation-free
/// `opz`/`omz`) times an enhancement factor `F_x` of that channel's reduced
/// gradient, screened independently on the floored spin density — exactly as
/// `lda_x`. A functional supplies only `enhancement`, the dimensionless
/// `F_x(x_σ)`; PBE-x and B88 therefore share this one screen + `lda_x_spin` +
/// per-channel-sum source rather than each forking it (reuse rule — the
/// enhancement is the sole parameter). Provenance: ported-from-libxc (MPL-2.0),
/// `maple/util.mpl` `gga_exchange`.
pub(crate) fn gga_exchange<N, F>(
    v: &GgaVars<N>,
    dens_threshold: f64,
    zeta_threshold: f64,
    enhancement: F,
) -> N
where
    N: DualNum<f64> + Copy,
    F: Fn(N) -> N,
{
    let up = if vars::screen_dens(v.na, dens_threshold) {
        N::from(0.0)
    } else {
        vars::lda_x_spin(v.rs, v.opz, zeta_threshold) * enhancement(v.xs0)
    };
    let dn = if vars::screen_dens(v.nb, dens_threshold) {
        N::from(0.0)
    } else {
        vars::lda_x_spin(v.rs, v.omz, zeta_threshold) * enhancement(v.xs1)
    };
    up + dn
}

/// A GGA functional's energy per particle, written generically over a
/// dual-number scalar. Mirrors libxc's `f := (rs, z, xt, xs0, xs1) -> ...`.
pub(crate) trait GgaEnergy: Send + Sync {
    fn info(&self) -> &FunctionalInfo;
    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N;
}

/// Object-safe wrapper turning any [`GgaEnergy`] into an [`XcEval`].
pub(crate) struct Gga<F: GgaEnergy>(pub F);

impl<F: GgaEnergy> XcEval for Gga<F> {
    fn info(&self) -> &FunctionalInfo {
        self.0.info()
    }

    fn eval(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let sigma = input.sigma.ok_or(XcError::MissingInput("sigma"))?;
        match spin {
            Spin::Unpolarized => self.eval_unpol(np, input.rho, sigma),
            Spin::Polarized => self.eval_pol(np, input.rho, sigma),
        }
    }
}

impl<F: GgaEnergy> Gga<F> {
    /// libxc's σ floor: `sigma_threshold² = (dens_threshold^(4/3))²`.
    fn sigma_floor(&self) -> f64 {
        let st = self.0.info().dens_threshold.powf(4.0 / 3.0);
        st * st
    }

    fn eval_unpol(&self, np: usize, rho: &[f64], sigma: &[f64]) -> Result<XcResult, XcError> {
        check_len(rho, np)?;
        check_len(sigma, np)?;
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; np];
        let mut vsigma = vec![0.0; np];
        for i in 0..np {
            let n = rho[i];
            if n < thr || n.is_nan() {
                continue;
            }
            let nf = n.max(thr);
            let sf = sigma[i].max(sfloor); // libxc floors σ to sigma_threshold²
            let (e, g) = gradient(
                |v: SVector<DualSVec64<2>, 2>| {
                    let n = v[0];
                    let s = v[1];
                    let two = DualSVec64::<2>::from(2.0);
                    let four = DualSVec64::<2>::from(4.0);
                    let one = DualSVec64::<2>::from(1.0);
                    let rs = vars::rs_from_n(n);
                    let half = n / two;
                    let xt2 = vars::reduced_grad_sq(s, n);
                    // unpolarized: n_a = n/2, σ_aa = σ/4 per spin channel
                    let xs = vars::reduced_grad(s / four, half);
                    n * self.0.f(GgaVars {
                        rs,
                        z: DualSVec64::<2>::from(0.0),
                        opz: one,
                        omz: one,
                        na: half,
                        nb: half,
                        xt2,
                        xs0: xs,
                        xs1: xs,
                    })
                },
                &SVector::<f64, 2>::from([nf, sf]),
            );
            exc[i] = e / nf;
            vrho[i] = g[0];
            vsigma[i] = g[1];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            ..Default::default()
        })
    }

    fn eval_pol(&self, np: usize, rho: &[f64], sigma: &[f64]) -> Result<XcResult, XcError> {
        check_len(rho, 2 * np)?;
        check_len(sigma, 3 * np)?;
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; 2 * np];
        let mut vsigma = vec![0.0; 3 * np];
        for i in 0..np {
            let na = rho[2 * i];
            let nb = rho[2 * i + 1];
            let n = na + nb;
            if n < thr || n.is_nan() {
                continue;
            }
            // Floor densities and σ_aa/σ_bb, then clamp σ_ab to [−s_ave, +s_ave]
            // with s_ave from the floored σ_aa/σ_bb — libxc's exact work_gga steps.
            let na_f = na.max(thr);
            let nb_f = nb.max(thr);
            let saa_f = sigma[3 * i].max(sfloor);
            let sbb_f = sigma[3 * i + 2].max(sfloor);
            let s_ave = 0.5 * (saa_f + sbb_f);
            let sab = sigma[3 * i + 1];
            let sab = if sab >= -s_ave { sab } else { -s_ave };
            let sab_c = if sab <= s_ave { sab } else { s_ave };
            let n_f = na_f + nb_f;
            let (e, g) = gradient(
                |v: SVector<DualSVec64<5>, 5>| {
                    let na = v[0];
                    let nb = v[1];
                    let saa = v[2];
                    let sab = v[3];
                    let sbb = v[4];
                    let n = na + nb;
                    let rs = vars::rs_from_n(n);
                    let z = (na - nb) / n;
                    let opz = (na + na) / n; // 1 + z, cancellation-free
                    let omz = (nb + nb) / n; // 1 − z, cancellation-free
                    let sigma_tot = saa + sab + sab + sbb; // σ_aa + 2σ_ab + σ_bb
                    let xt2 = vars::reduced_grad_sq(sigma_tot, n);
                    let xs0 = vars::reduced_grad(saa, na);
                    let xs1 = vars::reduced_grad(sbb, nb);
                    n * self.0.f(GgaVars {
                        rs,
                        z,
                        opz,
                        omz,
                        na,
                        nb,
                        xt2,
                        xs0,
                        xs1,
                    })
                },
                &SVector::<f64, 5>::from([na_f, nb_f, saa_f, sab_c, sbb_f]),
            );
            exc[i] = e / n_f;
            vrho[2 * i] = g[0];
            vrho[2 * i + 1] = g[1];
            vsigma[3 * i] = g[2];
            vsigma[3 * i + 1] = g[3];
            vsigma[3 * i + 2] = g[4];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::{Family, FunctionalId, Kind};

    struct DummyGga(FunctionalInfo);
    impl GgaEnergy for DummyGga {
        fn info(&self) -> &FunctionalInfo {
            &self.0
        }
        fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
            // depends on rs and the reduced gradients so vrho & vsigma are nonzero
            let c = N::from(1e-3);
            -v.rs.recip() + (v.xs0 * v.xs0 + v.xs1 * v.xs1) * c
        }
    }

    fn dummy() -> Gga<DummyGga> {
        Gga(DummyGga(FunctionalInfo {
            id: Some(FunctionalId::GgaXPbe),
            name: "dummy_gga",
            family: Family::Gga,
            kind: Kind::Exchange,
            needs_sigma: true,
            needs_lapl: false,
            needs_tau: false,
            dens_threshold: 1e-15,
            hybrid: None,
        }))
    }

    #[test]
    fn missing_sigma_errors() {
        let f = dummy();
        let rho = [0.3];
        let err = f
            .eval(Spin::Unpolarized, 1, &XcInput::lda(&rho))
            .unwrap_err();
        assert_eq!(err, XcError::MissingInput("sigma"));
    }

    #[test]
    fn unpol_runs_finite() {
        let f = dummy();
        let rho = [0.1, 0.5];
        let sigma = [0.01, 0.2];
        let out = f
            .eval(Spin::Unpolarized, 2, &XcInput::gga(&rho, &sigma))
            .unwrap();
        assert_eq!(out.vsigma.len(), 2);
        assert!(out
            .exc
            .iter()
            .chain(&out.vrho)
            .chain(&out.vsigma)
            .all(|v| v.is_finite()));
    }

    #[test]
    fn pol_runs_finite() {
        let f = dummy();
        let rho = [0.05, 0.05, 0.3, 0.2];
        let sigma = [0.01, 0.005, 0.01, 0.2, 0.1, 0.15];
        let out = f
            .eval(Spin::Polarized, 2, &XcInput::gga(&rho, &sigma))
            .unwrap();
        assert_eq!(out.vrho.len(), 4);
        assert_eq!(out.vsigma.len(), 6);
        assert!(out.vrho.iter().chain(&out.vsigma).all(|v| v.is_finite()));
    }

    /// Full polarization (n_b = 0) must stay finite: libxc floors n_b to
    /// dens_threshold so the minority reduced gradient never divides by zero.
    #[test]
    fn full_polarization_is_finite() {
        let f = dummy();
        let rho = [1.0, 0.0];
        let sigma = [0.1, 0.0, 0.0];
        let out = f
            .eval(Spin::Polarized, 1, &XcInput::gga(&rho, &sigma))
            .unwrap();
        assert!(out
            .exc
            .iter()
            .chain(&out.vrho)
            .chain(&out.vsigma)
            .all(|v| v.is_finite()));
    }
}
