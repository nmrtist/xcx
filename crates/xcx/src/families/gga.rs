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
use num_dual::{gradient, hessian, Dual2SVec64, DualNum, DualSVec64};

use super::{check_len, XcEval};
use crate::error::XcError;
use crate::func::{FunctionalInfo, Spin};
use crate::io::{XcInput, XcResult};
use crate::reduced::vars;

/// Reduced variables passed to a GGA energy expression. As in [`super::lda`],
/// `opz`/`omz` are the cancellation-free spin factors `1 ± z` (`= 2·n_{a,b}/n`),
/// matching libxc's exchange arithmetic at full spin polarization. All reduced
/// gradients are carried **squared and sqrt-free** — the total `xt2` and the
/// per-spin `xs0_sq`/`xs1_sq` — so forward-AD's *second* derivatives stay
/// accurate as σ → 0 (the `√σ` trap, divergence #4; see [`vars::reduced_grad_sq`]).
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
    /// Squared spin-up reduced gradient `x_{s0}² = σ_aa/n_a^(8/3)`, sqrt-free (so
    /// the second derivative `v2sigma2` is accurate at small σ — divergence #4).
    /// Enhancements consume this squared form directly.
    pub xs0_sq: N,
    /// Squared spin-down reduced gradient `x_{s1}² = σ_bb/n_b^(8/3)`, sqrt-free.
    pub xs1_sq: N,
}

/// GGA-exchange skeleton shared by every GGA exchange functional — the Rust
/// mirror of libxc's `util.mpl` `gga_exchange(func, rs, z, xs0, xs1)`. Each spin
/// channel is the LDA exchange of that channel (`lda_x_spin`, cancellation-free
/// `opz`/`omz`) times an enhancement factor `F_x` of that channel's reduced
/// gradient, screened independently on the floored spin density — exactly as
/// `lda_x`. A functional supplies only `enhancement`, the dimensionless `F_x`
/// **as a function of the squared reduced gradient `x_σ²`** (sqrt-free, divergence
/// #4): both PBE-x and B88 need only `x²` analytically (PBE-x is rational in `s²`;
/// B88's `x·asinh x` is a power series in `x²` near 0), so passing `x²` keeps
/// `v2sigma2` accurate at small σ. PBE-x and B88 share this one screen +
/// `lda_x_spin` + per-channel-sum source rather than each forking it (reuse rule —
/// the enhancement is the sole parameter). Provenance: ported-from-libxc
/// (MPL-2.0), `maple/util.mpl` `gga_exchange`.
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
        vars::lda_x_spin(v.rs, v.opz, zeta_threshold) * enhancement(v.xs0_sq)
    };
    let dn = if vars::screen_dens(v.nb, dens_threshold) {
        N::from(0.0)
    } else {
        vars::lda_x_spin(v.rs, v.omz, zeta_threshold) * enhancement(v.xs1_sq)
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

    fn eval_fxc(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let sigma = input.sigma.ok_or(XcError::MissingInput("sigma"))?;
        match spin {
            Spin::Unpolarized => self.eval_fxc_unpol(np, input.rho, sigma),
            Spin::Polarized => self.eval_fxc_pol(np, input.rho, sigma),
        }
    }
}

impl<F: GgaEnergy> Gga<F> {
    /// libxc's σ floor: `sigma_threshold² = (dens_threshold^(4/3))²`.
    fn sigma_floor(&self) -> f64 {
        let st = self.0.info().dens_threshold.powf(4.0 / 3.0);
        st * st
    }

    /// Unpolarized energy density `e = n·f` at one point, generic over the dual
    /// scalar `N` so the *same* expression feeds both the gradient (vxc) and the
    /// Hessian (fxc) harness. Seed vector is `[n, σ]` (floored). Per the
    /// unpolarized convention `n_a = n/2`, `σ_aa = σ/4` per spin channel.
    fn energy_unpol<N: DualNum<f64> + Copy>(&self, x: &SVector<N, 2>) -> N {
        let n = x[0];
        let s = x[1];
        let rs = vars::rs_from_n(n);
        let half = n / N::from(2.0);
        let xt2 = vars::reduced_grad_sq(s, n);
        // Per the unpolarized convention each channel has n_σ = n/2, σ_σσ = σ/4;
        // both channels share the same squared reduced gradient (sqrt-free).
        let xs_sq = vars::reduced_grad_sq(s / N::from(4.0), half);
        n * self.0.f(GgaVars {
            rs,
            z: N::from(0.0),
            opz: N::from(1.0),
            omz: N::from(1.0),
            na: half,
            nb: half,
            xt2,
            xs0_sq: xs_sq,
            xs1_sq: xs_sq,
        })
    }

    /// Polarized energy density `e = n·f` at one point, generic over `N`. Seed
    /// vector is `[n_a, n_b, σ_aa, σ_ab, σ_bb]` (floored/clamped by the caller).
    fn energy_pol<N: DualNum<f64> + Copy>(&self, x: &SVector<N, 5>) -> N {
        let na = x[0];
        let nb = x[1];
        let saa = x[2];
        let sab = x[3];
        let sbb = x[4];
        let n = na + nb;
        let rs = vars::rs_from_n(n);
        let z = (na - nb) / n;
        let opz = (na + na) / n; // 1 + z, cancellation-free
        let omz = (nb + nb) / n; // 1 − z, cancellation-free
        let sigma_tot = saa + sab + sab + sbb; // σ_aa + 2σ_ab + σ_bb
        let xt2 = vars::reduced_grad_sq(sigma_tot, n);
        // Per-spin squared reduced gradients, sqrt-free (σ floored > 0 by caller).
        let xs0_sq = vars::reduced_grad_sq(saa, na);
        let xs1_sq = vars::reduced_grad_sq(sbb, nb);
        n * self.0.f(GgaVars {
            rs,
            z,
            opz,
            omz,
            na,
            nb,
            xt2,
            xs0_sq,
            xs1_sq,
        })
    }

    /// Floor and clamp the polarized inputs exactly as libxc's `work_gga` does,
    /// returning the seed vector `[n_a, n_b, σ_aa, σ_ab, σ_bb]` and the floored
    /// total density `n_f` (the `exc` denominator). Shared by the vxc and fxc
    /// polarized harnesses so both seed at identical points.
    fn seed_pol(&self, na: f64, nb: f64, saa: f64, sab: f64, sbb: f64) -> (SVector<f64, 5>, f64) {
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let na_f = na.max(thr);
        let nb_f = nb.max(thr);
        let saa_f = saa.max(sfloor);
        let sbb_f = sbb.max(sfloor);
        let s_ave = 0.5 * (saa_f + sbb_f);
        let sab = if sab >= -s_ave { sab } else { -s_ave };
        let sab_c = if sab <= s_ave { sab } else { s_ave };
        (
            SVector::<f64, 5>::from([na_f, nb_f, saa_f, sab_c, sbb_f]),
            na_f + nb_f,
        )
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
                |v: SVector<DualSVec64<2>, 2>| self.energy_unpol(&v),
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
            let (seed, n_f) =
                self.seed_pol(na, nb, sigma[3 * i], sigma[3 * i + 1], sigma[3 * i + 2]);
            let (e, g) = gradient(|v: SVector<DualSVec64<5>, 5>| self.energy_pol(&v), &seed);
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

    /// Second-order (`fxc`) unpolarized harness: same screening/flooring/seeding
    /// as [`eval_unpol`](Self::eval_unpol), evaluated through num-dual's `hessian`
    /// (which also returns value + gradient, so `exc`/`vrho`/`vsigma` come out for
    /// free). The 2×2 Hessian over `[n, σ]` maps to `v2rho2 = ∂²e/∂n²`,
    /// `v2rhosigma = ∂²e/∂n∂σ`, `v2sigma2 = ∂²e/∂σ²` (one component each).
    fn eval_fxc_unpol(&self, np: usize, rho: &[f64], sigma: &[f64]) -> Result<XcResult, XcError> {
        check_len(rho, np)?;
        check_len(sigma, np)?;
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; np];
        let mut vsigma = vec![0.0; np];
        let mut v2rho2 = vec![0.0; np];
        let mut v2rhosigma = vec![0.0; np];
        let mut v2sigma2 = vec![0.0; np];
        for i in 0..np {
            let n = rho[i];
            if n < thr || n.is_nan() {
                continue; // below threshold: energy and every derivative stay 0
            }
            let nf = n.max(thr);
            let sf = sigma[i].max(sfloor);
            let (e, g, h) = hessian(
                |v: SVector<Dual2SVec64<2>, 2>| self.energy_unpol(&v),
                &SVector::<f64, 2>::from([nf, sf]),
            );
            exc[i] = e / nf;
            vrho[i] = g[0];
            vsigma[i] = g[1];
            v2rho2[i] = h[(0, 0)];
            v2rhosigma[i] = h[(0, 1)];
            v2sigma2[i] = h[(1, 1)];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            v2rho2,
            v2rhosigma,
            v2sigma2,
            ..Default::default()
        })
    }

    /// Second-order (`fxc`) polarized harness. The 5×5 Hessian over
    /// `[n_a, n_b, σ_aa, σ_ab, σ_bb]` is packed into libxc's xc.h ordering:
    /// `v2rho2 = [aa, ab, bb]`; `v2rhosigma = [a·aa, a·ab, a·bb, b·aa, b·ab,
    /// b·bb]` (ρ-spin major); `v2sigma2 = [aa·aa, aa·ab, aa·bb, ab·ab, ab·bb,
    /// bb·bb]` (symmetric upper triangle).
    fn eval_fxc_pol(&self, np: usize, rho: &[f64], sigma: &[f64]) -> Result<XcResult, XcError> {
        check_len(rho, 2 * np)?;
        check_len(sigma, 3 * np)?;
        let thr = self.0.info().dens_threshold;
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; 2 * np];
        let mut vsigma = vec![0.0; 3 * np];
        let mut v2rho2 = vec![0.0; 3 * np];
        let mut v2rhosigma = vec![0.0; 6 * np];
        let mut v2sigma2 = vec![0.0; 6 * np];
        for i in 0..np {
            let na = rho[2 * i];
            let nb = rho[2 * i + 1];
            let n = na + nb;
            if n < thr || n.is_nan() {
                continue;
            }
            let (seed, n_f) =
                self.seed_pol(na, nb, sigma[3 * i], sigma[3 * i + 1], sigma[3 * i + 2]);
            let (e, g, h) = hessian(|v: SVector<Dual2SVec64<5>, 5>| self.energy_pol(&v), &seed);
            exc[i] = e / n_f;
            vrho[2 * i] = g[0];
            vrho[2 * i + 1] = g[1];
            vsigma[3 * i] = g[2];
            vsigma[3 * i + 1] = g[3];
            vsigma[3 * i + 2] = g[4];
            // density-density block (indices 0=n_a, 1=n_b)
            v2rho2[3 * i] = h[(0, 0)];
            v2rho2[3 * i + 1] = h[(0, 1)];
            v2rho2[3 * i + 2] = h[(1, 1)];
            // density-sigma block (ρ-spin major: a×{aa,ab,bb}, b×{aa,ab,bb})
            v2rhosigma[6 * i] = h[(0, 2)];
            v2rhosigma[6 * i + 1] = h[(0, 3)];
            v2rhosigma[6 * i + 2] = h[(0, 4)];
            v2rhosigma[6 * i + 3] = h[(1, 2)];
            v2rhosigma[6 * i + 4] = h[(1, 3)];
            v2rhosigma[6 * i + 5] = h[(1, 4)];
            // sigma-sigma block (symmetric upper triangle over {aa,ab,bb})
            v2sigma2[6 * i] = h[(2, 2)];
            v2sigma2[6 * i + 1] = h[(2, 3)];
            v2sigma2[6 * i + 2] = h[(2, 4)];
            v2sigma2[6 * i + 3] = h[(3, 3)];
            v2sigma2[6 * i + 4] = h[(3, 4)];
            v2sigma2[6 * i + 5] = h[(4, 4)];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            v2rho2,
            v2rhosigma,
            v2sigma2,
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
            // depends on rs and the squared reduced gradients so vrho & vsigma are
            // nonzero (the harness now carries the per-spin gradients squared).
            let c = N::from(1e-3);
            -v.rs.recip() + (v.xs0_sq + v.xs1_sq) * c
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
