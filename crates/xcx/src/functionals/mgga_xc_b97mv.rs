// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! B97M-V — `mgga_xc_b97m_v` (libxc 254): the Mardirossian–Head-Gordon
//! combinatorially-optimized meta-GGA with VV10 nonlocal correlation.
//! N. Mardirossian & M. Head-Gordon, *J. Chem. Phys.* **142**, 074111 (2015).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/mgga_xc_b97mv.c`
//! (`par_b97m_v`) + `maple/mgga_exc/mgga_xc_b97mv.mpl` + `maple/b97mv.mpl` +
//! `maple/lda_exc/lda_x.mpl` + `maple/lda_exc/lda_c_pw.mpl` (**modified** PW92
//! set — `b97mv.mpl` defines `lda_c_pw_modified_params`) + `maple/util.mpl`
//! (`lda_stoll_par`/`lda_stoll_perp`).
//!
//! ## Scope fence
//!
//! xcx evaluates the semilocal meta-GGA only; the host adds the VV10 nonlocal
//! term from the exposed `Vv10Params { b: 6.0, c: 0.01 }` (libxc
//! `nlc_b`/`nlc_C`). No exact exchange (pure meta-GGA, rung `MetaGga`).
//!
//! ## Form — the Mardirossian–Head-Gordon {w, u} 2D expansion
//!
//! Every ingredient is a truncated 2D power series over the meta-GGA
//! inhomogeneity variables (`b97mv.mpl` `b97mv_g`):
//! ```text
//! g(γ, terms, x², t…) = Σ_k c_k · w^{i_k} · u^{j_k}
//! u = γ·x²/(1 + γ·x²)                                   (sqrt-free in x²)
//! w_ss(t)      = (K − t)/(K + t)                       (K = K_FACTOR_C; t = τ/n^{5/3})
//! w_os(t₀,t₁)  = [t₀(K − t₁) + t₁(K − t₀)]/[t₀(K + t₁) + t₁(K + t₀)]
//! ```
//! `w_os` is the cancellation-free symmetric regrouping of the maple's
//! `(K(t₀+t₁) − 2t₀t₁)/(K(t₀+t₁) + 2t₀t₁)` (algebraically identical; each
//! `(K − tᵢ)` factor hits zero on its own at the UEG/iso-orbital limit instead
//! of cancelling 2K² against itself). Exchange applies the series per channel
//! on top of `lda_x_spin`; correlation applies it on the Stoll split of
//! modified PW92 (same-spin per channel; opposite-spin with the combined
//! squared gradient `(x₀² + x₁²)/2`). Both are shared with ωB97M-V
//! ([`super::hyb_mgga_xc_wb97m_v`]), which differs only in the term tables and
//! the SR-erf attenuation on exchange.

use num_dual::DualNum;

use super::mgga_c_m06_l::f_pw;
use crate::families::mgga::{mgga_exchange, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, HybridInfo, Kind, Vv10Params};
use crate::reduced::consts::K_FACTOR_C;

/// One term of the {w, u} expansion: `(coefficient, w-power, u-power)`.
pub(crate) type B97MvTerm = (f64, i32, i32);

// Shared B97M-V/ωB97M-V γ's (`b97mv_gamma_x/ss/os`, identical in both).
pub(crate) const GAMMA_X: f64 = 0.004;
pub(crate) const GAMMA_SS: f64 = 0.2;
pub(crate) const GAMMA_OS: f64 = 0.006;

// B97M-V term tables (libxc `par_b97m_v`; rows are (c, w-pow, u-pow) exactly
// as `b97mv_par_x/ss/os` in `maple/mgga_exc/mgga_xc_b97mv.mpl`).
const PAR_X: [B97MvTerm; 5] = [
    (1.000, 0, 0),
    (1.308, 0, 1),
    (1.901, 0, 2),
    (0.416, 1, 0),
    (3.070, 1, 1),
];
const PAR_SS: [B97MvTerm; 5] = [
    (1.000, 0, 0),
    (-1.855, 0, 2),
    (-5.668, 1, 0),
    (-20.497, 3, 2),
    (-20.364, 4, 2),
];
const PAR_OS: [B97MvTerm; 5] = [
    (1.000, 0, 0),
    (1.573, 0, 1),
    (-6.298, 0, 3),
    (2.535, 1, 0),
    (-6.427, 3, 2),
];

/// Same-spin kinetic inhomogeneity `w_ss(t) = (K − t)/(K + t)` ∈ (−1, 1].
/// `K + t > 0` always (t ≥ 0), so this is pole-free.
pub(crate) fn b97mv_wx_ss<N: DualNum<f64> + Copy>(t: N) -> N {
    let k = N::from(K_FACTOR_C);
    (k - t) / (k + t)
}

/// Opposite-spin kinetic inhomogeneity in the symmetric, cancellation-free
/// regrouping (see module docs). Denominator `t₀(K+t₁) + t₁(K+t₀) > 0` for
/// `t > 0` (τ floored by the harness).
pub(crate) fn b97mv_wx_os<N: DualNum<f64> + Copy>(t0: N, t1: N) -> N {
    let k = N::from(K_FACTOR_C);
    (t0 * (k - t1) + t1 * (k - t0)) / (t0 * (k + t1) + t1 * (k + t0))
}

/// The Mardirossian–Head-Gordon 2D series `Σ c·w^i·u^j` (`b97mv.mpl`
/// `b97mv_g`), taking the **squared** reduced gradient (`u = γw_x/(1+γw_x)`,
/// sqrt-free, bounded in [0, 1)) and the precomputed kinetic variable `w`.
/// Integer powers via `powi` keep AD exact at `w = 0`/negative `w`.
pub(crate) fn b97mv_g<N: DualNum<f64> + Copy>(gamma: f64, terms: &[B97MvTerm], x_sq: N, w: N) -> N {
    let gw = N::from(gamma) * x_sq;
    let u = gw / (N::from(1.0) + gw);
    let mut acc = N::from(0.0);
    for &(c, wi, uj) in terms {
        let mut term = N::from(c);
        if wi != 0 {
            term *= w.powi(wi);
        }
        if uj != 0 {
            term *= u.powi(uj);
        }
        acc += term;
    }
    acc
}

/// The B97M-V-family correlation: the {w, u} series applied to the Stoll
/// decomposition of **modified** PW92 (`b97mv.mpl` `b97mv_fpar` + `b97mv_fos`).
/// Shared by B97M-V and ωB97M-V (term tables injected). Same-spin channels are
/// screened like libxc's `screen_dens_zeta`; the opposite-spin series takes the
/// combined squared gradient `(x₀² + x₁²)/2` (the maple's `√(x₀²+x₁²)/√2`
/// squared — sqrt-free) and `w_os(t₀, t₁)`.
pub(crate) fn b97mv_correlation<N: DualNum<f64> + Copy>(
    v: &MggaVars<N>,
    dens_threshold: f64,
    zeta_threshold: f64,
    par_ss: &[B97MvTerm],
    par_os: &[B97MvTerm],
) -> N {
    let zt = zeta_threshold;
    let up_screened = v.na.re() <= dens_threshold || v.opz.re() <= zt;
    let dn_screened = v.nb.re() <= dens_threshold || v.omz.re() <= zt;
    let par_up = if up_screened {
        N::from(0.0)
    } else {
        v.opz / N::from(2.0)
            * f_pw(
                v.rs * (N::from(2.0) / v.opz).powf(1.0 / 3.0),
                N::from(1.0),
                zt,
            )
    };
    let par_dn = if dn_screened {
        N::from(0.0)
    } else {
        v.omz / N::from(2.0)
            * f_pw(
                v.rs * (N::from(2.0) / v.omz).powf(1.0 / 3.0),
                N::from(-1.0),
                zt,
            )
    };
    let perp = f_pw(v.rs, v.z, zt) - par_up - par_dn;

    let up = if up_screened {
        N::from(0.0)
    } else {
        par_up * b97mv_g(GAMMA_SS, par_ss, v.xs0_sq, b97mv_wx_ss(v.t0))
    };
    let dn = if dn_screened {
        N::from(0.0)
    } else {
        par_dn * b97mv_g(GAMMA_SS, par_ss, v.xs1_sq, b97mv_wx_ss(v.t1))
    };
    let cross = perp
        * b97mv_g(
            GAMMA_OS,
            par_os,
            (v.xs0_sq + v.xs1_sq) / N::from(2.0),
            b97mv_wx_os(v.t0, v.t1),
        );
    up + dn + cross
}

pub(crate) struct MggaXcB97mV {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaXcB97mV {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaXcB97mV),
                name: "mgga_xc_b97m_v",
                family: Family::Mgga,
                kind: Kind::ExchangeCorrelation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-15, // libxc mgga_xc_b97m_v threshold
                // Pure meta-GGA (exx = 0, no CAM): `hybrid` carries only the
                // VV10 parameters the host needs for the nonlocal term.
                // `rung()` still reports `MetaGga` (exx-/cam-free).
                hybrid: Some(HybridInfo {
                    exx_fraction: 0.0,
                    cam: None,
                    vv10: Some(Vv10Params { b: 6.0, c: 0.01 }),
                }),
            },
            zeta_threshold: f64::EPSILON,
        }))
    }
}

impl MggaEnergy for MggaXcB97mV {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let thr = self.info.dens_threshold;
        let zt = self.zeta_threshold;
        // Exchange: per-channel LDA-x × the {w, u} series (`b97mv_f_aux`; the
        // `f_lda_x` channel screen is the shared `mgga_exchange` skeleton's).
        let ex = mgga_exchange(&v, thr, zt, |x_sq, t| {
            b97mv_g(GAMMA_X, &PAR_X, x_sq, b97mv_wx_ss(t))
        });
        ex + b97mv_correlation(&v, thr, zt, &PAR_SS, &PAR_OS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::{DispersionModel, Rung};
    use crate::{Functional, Spin, XcInput};

    fn b97mv(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaXcB97mV, spin).unwrap()
    }

    /// Acceptance metadata: pure meta-GGA rung (despite carrying VV10 params),
    /// zero EXX, no CAM, VV10 b = 6.0 / C = 0.01, level-4 grid-sensitive grid,
    /// canonical VV10 dispersion pairing.
    #[test]
    fn metadata_mgga_rung_with_vv10() {
        let f = b97mv(Spin::Unpolarized);
        let info = f.info();
        assert_eq!(info.rung(), Rung::MetaGga);
        assert_eq!(f.exx_fraction(), 0.0);
        let h = info.hybrid.unwrap();
        assert!(h.cam.is_none());
        let vv10 = h.vv10.unwrap();
        assert_eq!((vv10.b, vv10.c), (6.0, 0.01));
        let g = info.grid();
        assert_eq!((g.level, g.grid_sensitive), (4, true));
        let d = info.dispersion().unwrap();
        assert_eq!(d.model, DispersionModel::Vv10);
        assert_eq!(d.param_set, "mgga_xc_b97m_v");
    }

    /// UEG limits of the inhomogeneity variables: u(0) = 0 and w_ss(K) = 0, so
    /// only the (0,0) terms survive — exchange → LDA-x, correlation → PW92.
    #[test]
    fn ueg_limit_series_collapse() {
        assert_eq!(
            b97mv_g(GAMMA_X, &PAR_X, 0.0_f64, b97mv_wx_ss(K_FACTOR_C)),
            1.0
        );
        assert_eq!(b97mv_wx_os(K_FACTOR_C, K_FACTOR_C), 0.0);
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = b97mv(Spin::Unpolarized);
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
            (1.0, 0.4, 0.06), // τ ≈ τ_W
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
        let f = b97mv(Spin::Polarized);
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
        assert_eq!(out.vsigma[1], 0.0, "B97M-V vsigma_ab must be 0");
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
        let up = b97mv(Spin::Unpolarized);
        let po = b97mv(Spin::Polarized);
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
        let f = b97mv(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-10, 1e-11, 1.0, 1.0, 100.0, 50.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1.0, 0.5, 0.8, //
        ];
        let tau = [0.5, 0.0, 0.0, 0.5, 1e-12, 1e-13, 0.5, 0.5, 50.0, 30.0];
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
