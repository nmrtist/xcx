// Copyright (c) 2026 Jiekang Tian and the xcx authors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ωB97M(2) — range-separated double-hybrid meta-GGA (xcx-private id 100004;
//! **not in libxc**). N. Mardirossian & M. Head-Gordon, *J. Chem. Phys.*
//! **148**, 241736 (2018).
//!
//! Provenance: clean-room (from the publication). The functional form is
//! exactly ωB97M-V's machinery — SR-erf-attenuated {w_x, u_x} 2D-series
//! exchange plus the B97M-V-form Stoll-split correlation, shared with
//! [`super::hyb_mgga_xc_wb97m_v`] / [`super::mgga_xc_b97mv`] (paper Eqs. 3–4:
//! "Definitions for the terms … can be found in Section V of Ref. 31
//! [ωB97M-V]", so the γ's and {w, u} variables are ωB97M-V's) — with the
//! 14-parameter coefficient set of paper **Table II**, selected
//! combinatorially in the presence of PT2 correlation:
//!
//! - exchange `c_x,ij`: 00 = 0.37806, 01 = 0.13642, 20 = 0.28193,
//!   30 = −0.21886, 41 = 0.70767 (UEG constraint `c_x + c_x,00 = 1`),
//! - same-spin correlation `c_css,ij`: 00 = 0.54846, 10 = −1.17724,
//!   20 = −3.67267,
//! - opposite-spin correlation `c_cos,ij`: 00 = 0.46152, 01 = −1.94794,
//!   02 = 3.24910, 20 = 2.30490, 22 = −2.26280,
//! - SR exact exchange `c_x = 0.62194` (ω = 0.3, 100% LR ⇒ CAM in the xcx
//!   convention `EXX(r₁₂) = α + β·erf(ω·r₁₂)`: α = 0.62194, β = 0.37806),
//! - nonlocal correlation: `c_VV10 = 0.65904`, `c_PT2 = 0.34096`, constrained
//!   `c_PT2 + c_VV10 = 1`; PT2 is a single canonical-MP2 coefficient, so
//!   `c_os = c_ss = 0.34096`.
//!
//! xcx evaluates only the SR-attenuated semilocal part; the host adds the
//! range-separated exact exchange (CAM metadata), the PT2 correlation scaled
//! by `double_hybrid()` (`c_os = c_ss = c_PT2`), and the VV10 term built from
//! the exposed `Vv10Params { b: 6.0, c: 0.01 }` (ωB97M-V's, per Ref. 31)
//! **scaled by `c_VV10 = 1 − c_PT2`** — the paper's constraint makes the
//! scale derivable from the metadata.

use num_dual::DualNum;

use super::attenuation::{att_a_cnst, attenuation_erf};
use super::mgga_xc_b97mv::{b97mv_correlation, b97mv_g, b97mv_wx_ss, B97MvTerm, GAMMA_X};
use crate::families::mgga::{Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{CamParams, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind, Vv10Params};
use crate::reduced::vars;

/// Range separation ω (paper: "for the range-separation parameter, ω, a value
/// of 0.3 is used", as in ωB97M-V).
const OMEGA: f64 = 0.3;
/// Short-range exact-exchange fraction α = c_x (paper Table II), constrained
/// by the UEG limit `c_x + c_x,00 = 1`.
const CAM_ALPHA: f64 = 0.62194;
/// Long-range increment β: α + β = 1 (100% LR exact exchange).
const CAM_BETA: f64 = 1.0 - CAM_ALPHA;

// ωB97M(2) term tables (paper Table II; rows are (c, w-pow, u-pow) in the
// b97mv-series layout shared with ωB97M-V).
const PAR_X: [B97MvTerm; 5] = [
    (0.37806, 0, 0),
    (0.13642, 0, 1),
    (0.28193, 2, 0),
    (-0.21886, 3, 0),
    (0.70767, 4, 1),
];
const PAR_SS: [B97MvTerm; 3] = [(0.54846, 0, 0), (-1.17724, 1, 0), (-3.67267, 2, 0)];
const PAR_OS: [B97MvTerm; 5] = [
    (0.46152, 0, 0),
    (-1.94794, 0, 1),
    (3.24910, 0, 2),
    (2.30490, 2, 0),
    (-2.26280, 2, 2),
];

pub(crate) struct HybMggaXcWb97m2 {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl HybMggaXcWb97m2 {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::HybMggaXcWb97m2),
                name: "hyb_mgga_xc_wb97m_2",
                family: Family::HybMgga,
                kind: Kind::ExchangeCorrelation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-13, // matches the ωB97M-V harness threshold
                hybrid: Some(HybridInfo {
                    exx_fraction: CAM_ALPHA,
                    cam: Some(CamParams {
                        omega: OMEGA,
                        alpha: CAM_ALPHA,
                        beta: CAM_BETA,
                    }),
                    // ωB97M(2) keeps VV10 (b/C from ωB97M-V, Ref. 31), scaled
                    // by c_VV10 = 1 − c_PT2 = 0.65904 (paper's constraint —
                    // the host derives the scale from `double_hybrid()`).
                    vv10: Some(Vv10Params { b: 6.0, c: 0.01 }),
                }),
            },
            zeta_threshold: f64::EPSILON,
        }))
    }
}

impl MggaEnergy for HybMggaXcWb97m2 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let thr = self.info.dens_threshold;
        let zt = self.zeta_threshold;
        let a_cnst = att_a_cnst(OMEGA);

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
    use crate::func::Rung;
    use crate::{Functional, Spin, XcInput};

    fn wb97m2(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybMggaXcWb97m2, spin).unwrap()
    }

    /// Metadata: DoubleHybrid rung (despite CAM), ω = 0.30 / α = 0.62194 /
    /// β = 0.37806 (paper Table II, `c_x + c_x,00 = 1`), PT2
    /// c_os = c_ss = c_PT2 = 0.34096, VV10 retained (b 6.0, C 0.01; host
    /// scales by 1 − c_PT2 = 0.65904), level-4 grid.
    #[test]
    fn metadata_cam_double_hybrid() {
        let f = wb97m2(Spin::Unpolarized);
        let info = f.info();
        assert_eq!(info.rung(), Rung::DoubleHybrid);
        let h = info.hybrid.unwrap();
        let cam = h.cam.unwrap();
        assert_eq!(
            (cam.omega, cam.alpha, cam.beta),
            (0.3, 0.62194, 1.0 - 0.62194)
        );
        assert_eq!(f.exx_fraction(), 0.62194);
        let vv10 = h.vv10.unwrap();
        assert_eq!((vv10.b, vv10.c), (6.0, 0.01));
        let p = info.double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.34096, 0.34096));
        // the paper's constraint c_PT2 + c_VV10 = 1 makes the VV10 scale
        // derivable: c_VV10 = 1 − c_os = 0.65904
        assert!((1.0 - p.c_os - 0.65904).abs() < 1e-12);
        let d = info.dispersion().unwrap();
        assert_eq!(d.model, crate::func::DispersionModel::Vv10);
        assert_eq!(d.param_set, "hyb_mgga_xc_wb97m_2");
        let g = info.grid();
        assert_eq!((g.level, g.grid_sensitive), (4, true));
        assert!(info.needs_tau && info.needs_sigma);
    }

    /// Must differ from ωB97M-V (refit coefficients) while sharing the form.
    #[test]
    fn differs_from_wb97m_v() {
        let a = wb97m2(Spin::Unpolarized);
        let b = Functional::new(FunctionalId::HybMggaXcWb97mV, Spin::Unpolarized).unwrap();
        let inp = XcInput::gga(&[0.7], &[0.3]).with_tau(&[0.5]);
        let ea = a.eval(1, &inp).unwrap().exc[0];
        let eb = b.eval(1, &inp).unwrap().exc[0];
        assert!((ea - eb).abs() > 1e-6 * ea.abs(), "{ea} vs {eb}");
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = wb97m2(Spin::Unpolarized);
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
            (1e-2, 1e-6, 1e-3), // low density: large attenuation argument
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
        let f = wb97m2(Spin::Polarized);
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
        assert_eq!(out.vsigma[1], 0.0, "ωB97M(2) vsigma_ab must be 0");
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
        let up = wb97m2(Spin::Unpolarized);
        let po = wb97m2(Spin::Polarized);
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
        let f = wb97m2(Spin::Polarized);
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
