// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! ωB97M-V — `hyb_mgga_xc_wb97m_v` (libxc 531): range-separated hybrid
//! meta-GGA with VV10 nonlocal correlation. N. Mardirossian & M. Head-Gordon,
//! *J. Chem. Phys.* **144**, 214110 (2016).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/hyb_mgga_xc_wb97mv.c`
//! (`par_wb97m_v`) + `maple/mgga_exc/hyb_mgga_xc_wb97mv.mpl` + `maple/b97mv.mpl`
//! + `maple/lda_exc/lda_x_erf.mpl` + `maple/attenuation.mpl`.
//!
//! ## Scope fence
//!
//! xcx evaluates only the semilocal part (SR-erf-attenuated {w, u}-series
//! exchange + B97M-V-form correlation). The host assembles, from metadata,
//! - exact exchange `EXX(r₁₂) = α + β·erf(ω·r₁₂)` with
//!   `ω = 0.30, α = 0.15, β = 0.85` (libxc `cam_alpha = 1.0`,
//!   `cam_beta = −(1 − 0.15)` translated to the xcx frozen convention), and
//! - the VV10 nonlocal term with `b = 6.0, C = 0.01` (libxc `nlc_b`/`nlc_C`).
//!
//! ## Form
//!
//! The Mardirossian–Head-Gordon {w_x, u_x} 2D expansion of
//! [`super::mgga_xc_b97mv`] with ωB97M-V's term tables (3 exchange, 5
//! same-spin, 6 opposite-spin terms), the exchange channel multiplied by the
//! shared SR-erf attenuation `F(a_σ)` ([`attenuation_erf`],
//! `a_σ = (4/(9π))^(1/3)·(ω/2)·rs/(1±z)^(1/3)`) and screened per channel like
//! libxc's `screen_dens_zeta` (`wb97mv_f` in the maple). Correlation is the
//! shared [`b97mv_correlation`] (modified-PW92 Stoll split), sqrt-free
//! throughout.

use num_dual::DualNum;

use super::attenuation::{att_a_cnst, attenuation_erf};
use super::mgga_xc_b97mv::{b97mv_correlation, b97mv_g, b97mv_wx_ss, B97MvTerm, GAMMA_X};
use crate::families::mgga::{Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{CamParams, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind, Vv10Params};
use crate::reduced::vars;

/// Range separation ω (libxc `_omega` = 0.3).
pub(crate) const OMEGA: f64 = 0.3;
/// Global (short-range) exact-exchange fraction α in the xcx CAM convention
/// `EXX(r₁₂) = α + β·erf(ω·r₁₂)` (libxc: `cam_alpha + cam_beta = 0.15`).
pub(crate) const CAM_ALPHA: f64 = 0.15;
/// Long-range increment β: `α + β = 1` (100% LR exact exchange).
pub(crate) const CAM_BETA: f64 = 1.0 - CAM_ALPHA;

// ωB97M-V term tables (libxc `par_wb97m_v`; rows are (c, w-pow, u-pow) exactly
// as `b97mv_par_x/ss/os` in `maple/mgga_exc/hyb_mgga_xc_wb97mv.mpl`; the
// maple's zero-padded rows are dropped — exact zeros).
const PAR_X: [B97MvTerm; 3] = [(0.85, 0, 0), (1.007, 0, 1), (0.259, 1, 0)];
const PAR_SS: [B97MvTerm; 5] = [
    (0.443, 0, 0),
    (-1.437, 0, 4),
    (-4.535, 1, 0),
    (-3.39, 2, 0),
    (4.278, 4, 3),
];
const PAR_OS: [B97MvTerm; 6] = [
    (1.0, 0, 0),
    (1.358, 1, 0),
    (2.924, 2, 0),
    (-8.812, 2, 1),
    (-1.39, 6, 0),
    (9.142, 6, 1),
];

pub(crate) struct HybMggaXcWb97mV {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl HybMggaXcWb97mV {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::HybMggaXcWb97mV),
                name: "hyb_mgga_xc_wb97m_v",
                family: Family::HybMgga,
                kind: Kind::ExchangeCorrelation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-13, // libxc hyb_mgga_xc_wb97m_v threshold
                hybrid: Some(HybridInfo {
                    exx_fraction: CAM_ALPHA,
                    cam: Some(CamParams {
                        omega: OMEGA,
                        alpha: CAM_ALPHA,
                        beta: CAM_BETA,
                    }),
                    vv10: Some(Vv10Params { b: 6.0, c: 0.01 }),
                }),
            },
            zeta_threshold: f64::EPSILON,
        }))
    }
}

impl MggaEnergy for HybMggaXcWb97mV {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let thr = self.info.dens_threshold;
        let zt = self.zeta_threshold;
        let a_cnst = att_a_cnst(OMEGA);

        // SR-erf {w, u}-series exchange, per channel (maple `wb97mv_f`,
        // screened by `screen_dens_zeta`).
        let up_screened = v.na.re() <= thr || v.opz.re() <= zt;
        let dn_screened = v.nb.re() <= thr || v.omz.re() <= zt;
        let x_chan = |opzv: N, xs_sq: N, t: N| {
            let a = N::from(a_cnst) * v.rs / opzv.cbrt();
            vars::lda_x_spin(v.rs, opzv, zt)
                * attenuation_erf(a)
                * b97mv_g(GAMMA_X, &PAR_X, xs_sq, b97mv_wx_ss(t))
        };
        let x_up = if up_screened {
            N::from(0.0)
        } else {
            x_chan(v.opz, v.xs0_sq, v.t0)
        };
        let x_dn = if dn_screened {
            N::from(0.0)
        } else {
            x_chan(v.omz, v.xs1_sq, v.t1)
        };

        x_up + x_dn + b97mv_correlation(&v, thr, zt, &PAR_SS, &PAR_OS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::{DispersionModel, Rung};
    use crate::{Functional, Spin, XcInput};

    fn wb97mv(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybMggaXcWb97mV, spin).unwrap()
    }

    /// Acceptance metadata: RangeSeparatedHybrid rung, verified CAM params
    /// (ω = 0.30, α = 0.15, β = 0.85), exx_fraction == α, VV10 b = 6.0 /
    /// C = 0.01, level-4 grid-sensitive grid, canonical VV10 pairing.
    #[test]
    fn metadata_cam_vv10_grid() {
        let f = wb97mv(Spin::Unpolarized);
        let info = f.info();
        assert_eq!(info.rung(), Rung::RangeSeparatedHybrid);
        let h = info.hybrid.unwrap();
        let cam = h.cam.unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.3, 0.15, 1.0 - 0.15));
        assert_eq!(f.exx_fraction(), 0.15);
        let vv10 = h.vv10.unwrap();
        assert_eq!((vv10.b, vv10.c), (6.0, 0.01));
        let g = info.grid();
        assert_eq!((g.level, g.grid_sensitive), (4, true));
        let d = info.dispersion().unwrap();
        assert_eq!(d.model, DispersionModel::Vv10);
        assert_eq!(d.param_set, "hyb_mgga_xc_wb97m_v");
        assert!(info.needs_tau && info.needs_sigma);
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = wb97mv(Spin::Unpolarized);
        let edens = |n: f64, s: f64, tau: f64| {
            n * f
                .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                .unwrap()
                .exc[0]
        };
        for &(n, s, tau) in &[
            (0.5, 0.1, 0.3),
            (2.0, 0.7, 1.5),
            (0.3, 0.02, 0.2),
            (5.0, 3.0, 8.0),
            (1.0, 0.4, 0.06),   // τ ≈ τ_W
            (1e-2, 1e-6, 1e-3), // low density: large attenuation argument a
        ] {
            let out = f
                .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                .unwrap();
            let (hn, hs, ht) = (1e-6 * n, 1e-6 * s, 1e-6 * tau);
            let fdn = (edens(n + hn, s, tau) - edens(n - hn, s, tau)) / (2.0 * hn);
            let fds = (edens(n, s + hs, tau) - edens(n, s - hs, tau)) / (2.0 * hs);
            let fdt = (edens(n, s, tau + ht) - edens(n, s, tau - ht)) / (2.0 * ht);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-5 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s} t={tau}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-5 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s} t={tau}: {} vs {fds}",
                out.vsigma[0]
            );
            assert!(
                (out.vtau[0] - fdt).abs() <= 1e-5 * out.vtau[0].abs().max(1.0),
                "vtau n={n} s={s} t={tau}: {} vs {fdt}",
                out.vtau[0]
            );
        }
    }

    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = wb97mv(Spin::Polarized);
        let (na, nb, saa, sab, sbb, ta, tb) = (0.6, 0.3, 0.1, 0.05, 0.08, 0.4, 0.25);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let t = [ta, tb];
        let edens = |r: [f64; 2], s: [f64; 3], t: [f64; 2]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s).with_tau(&t)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s).with_tau(&t)).unwrap();
        for (k, h) in [(0usize, 1e-6 * na), (1, 1e-6 * nb)] {
            let (mut rp, mut rm) = (r, r);
            rp[k] += h;
            rm[k] -= h;
            let fd = (edens(rp, s, t) - edens(rm, s, t)) / (2.0 * h);
            assert!(
                (out.vrho[k] - fd).abs() <= 1e-5 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        assert_eq!(out.vsigma[1], 0.0, "ωB97M-V vsigma_ab must be 0");
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp, t) - edens(r, sm, t)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-5 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
        for (k, h) in [(0usize, 1e-6 * ta), (1, 1e-6 * tb)] {
            let (mut tp, mut tm) = (t, t);
            tp[k] += h;
            tm[k] -= h;
            let fd = (edens(r, s, tp) - edens(r, s, tm)) / (2.0 * h);
            assert!(
                (out.vtau[k] - fd).abs() <= 1e-5 * out.vtau[k].abs().max(1.0),
                "vtau[{k}]: {} vs {fd}",
                out.vtau[k]
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = wb97mv(Spin::Unpolarized);
        let po = wb97mv(Spin::Polarized);
        let (n, s, tau) = (0.8, 0.3, 0.6);
        let ou = up
            .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
            .unwrap();
        let op = po
            .eval(
                1,
                &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0])
                    .with_tau(&[tau / 2.0, tau / 2.0]),
            )
            .unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-11 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-10 * ou.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = wb97mv(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-10, 1e-11, 1.0, 1.0, 1000.0, 500.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1e6, 5e5, 8e5, //
        ];
        let tau = [0.5, 0.0, 0.0, 0.5, 1e-12, 1e-13, 0.5, 0.5, 5e4, 3e4];
        let out = f
            .eval(5, &XcInput::gga(&rho, &sigma).with_tau(&tau))
            .unwrap();
        for v in out
            .exc
            .iter()
            .chain(&out.vrho)
            .chain(&out.vsigma)
            .chain(&out.vtau)
        {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
