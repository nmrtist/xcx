// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PW6B95 — `hyb_mgga_xc_pw6b95` (libxc 451), 28% EXX. Zhao & Truhlar,
//! J. Phys. Chem. A 109, 5656 (2005) (the "6-parameter" PW91/B95 hybrid).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/hyb_mgga_xc_b88b95.c`
//! (`xc_hyb_mgga_xc_pw6b95_init`) + `maple/gga_exc/gga_x_pw91.mpl` +
//! `maple/mgga_exc/mgga_c_bc95.mpl` + `maple/util.mpl`
//! (`gga_exchange`, `lda_stoll_par`/`lda_stoll_perp`, `Fermi_D`).
//!
//! libxc builds PW6B95 as the mix `0.72·GGA_X_MPW91(bt = 0.00538,
//! α = 1.7382/X2S², expo = 3.8901) + 1.0·MGGA_C_BC95(c_ss = 0.03668,
//! c_opp = 0.00262)`, with `cam_alpha = 0.28` exact exchange. xcx mirrors that
//! exactly: two *internal* component evaluators (not registered ids — their
//! parameters are PW6B95's, not the standalone functionals') combined with
//! [`crate::func::mixed_eval`], each screening at its own libxc threshold
//! (mPW91 `1e-15` via `work_gga`, B95 `1e-14` via `work_mgga`) just as libxc's
//! `xc_mix` does. The host adds the 28% EXX from
//! [`HybridInfo::exx_fraction`](crate::func::HybridInfo).
//!
//! ## mPW91 exchange part (in the squared reduced gradient `w = x²`)
//!
//! libxc's `pw91_f(x) = 1 + num(s)/den(s)`, `s = X2S·x`, with
//! `mpw91_set_ext_params` mapping `(bt, α, expo)` to the seven PW91 parameters.
//! Substituting those (the `X2S` powers cancel algebraically) gives the
//! **sqrt-free** form actually implemented here:
//! ```text
//! num(w) = [bt − (bt − β)·e^(−1.7382·w)]·w/X_FACTOR_C − F6·w^(expo/2)
//! den(w) = 1 + 6·bt·g(w) + F6·w^(expo/2)
//! ```
//! with `β = 5·(36π)^(−5/3)`, `F6 = 1e-6/X_FACTOR_C`, and `g(w) = x·asinh(x)`
//! the shared series-protected [`b88_g`] (libxc's `s·a·asinh(b·s)` term has
//! `b = 1/X2S`, so it reduces to `6·bt·x·asinh(x)` — the same `x·asinh(x)`
//! kernel B88 uses, reused rather than forked). `w^(expo/2)` (expo = 3.8901)
//! is a plain `powf`: σ is floored by the harness so `w > 0` and its
//! derivatives stay finite.
//!
//! ## B95 correlation part (Becke 1996, PW6B95 parameters)
//!
//! The Stoll decomposition of the modified-PW92 LSDA (`f_pw`, the same shared
//! uniform limit M06-L correlation uses — libxc's bc95 maple defines
//! `lda_c_pw_modified_params`), times sqrt-free kinetic/gradient factors:
//! ```text
//! same-spin σσ: stoll_par(±z) · t_σ·Fermi_D(x_σ², t_σ)/(K·(1 + c_ss·x_σ²)²)
//! opposite-spin: stoll_perp(z) / (1 + c_opp·(x₀² + x₁²))
//! ```
//! `Fermi_D` is the shared FHC-clamped factor from M06-L correlation; every
//! gradient enters squared. The Laplacian is unused.

use num_dual::DualNum;

use super::gga_x_b88::b88_g;
use super::mgga_c_m06_l::{f_pw, fermi_d};
use crate::error::XcError;
use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::mgga::{Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{mixed_eval, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind};
use crate::reduced::consts::{K_FACTOR_C, X_FACTOR_C};

// PW6B95 mixing (libxc `xc_hyb_mgga_xc_pw6b95_init`): 0.72·mPW91-x + 1.0·B95-c,
// 28% exact exchange.
const EXX_FRACTION: f64 = 0.28;
const MPW91_WEIGHT: f64 = 1.0 - EXX_FRACTION;

// PW6B95's mPW91 exchange parameters (`par_x_pw91` = {bt, α·X2S⁻², expo}; α and
// expo are used in `w`-space where the X2S powers cancel — see module docs).
const BT: f64 = 0.00538;
const ALPHA_W: f64 = 1.7382;
const EXPO: f64 = 3.8901;

// PW6B95's B95 correlation parameters (`par_c_bc95` = {c_ss, c_opp}).
const C_SS: f64 = 0.03668;
const C_OPP: f64 = 0.00262;

/// mPW91-form exchange enhancement `F(w)` in the squared reduced gradient
/// `w = x_σ²` (see module docs for the algebra mapping libxc's `s`-space form),
/// parameterized by `(bt, alpha_w, expo)` so PW6B95 (bt = 0.00538, 1.7382,
/// 3.8901) and PWPB95's reoptimized exchange can share one source.
fn mpw91_enhancement<N: DualNum<f64> + Copy>(w: N, bt: f64, alpha_w: f64, expo: f64) -> N {
    // β = 5·(36π)^(−5/3): the gradient-expansion constant `mpw91_set_ext_params`
    // folds into the PW91 `d` parameter.
    let beta = 5.0 * (36.0 * std::f64::consts::PI).powf(-5.0 / 3.0);
    let f6 = 1e-6 / X_FACTOR_C;
    let pow_term = w.powf(expo / 2.0) * f6; // f·s^expo, X2S^expo cancelled
    let num = (N::from(bt) - N::from(bt - beta) * (-w * alpha_w).exp()) * w / N::from(X_FACTOR_C)
        - pow_term;
    let den = N::from(1.0) + b88_g(w) * (6.0 * bt) + pow_term;
    N::from(1.0) + num / den
}

/// Internal mPW91-form exchange component (not a registered id: each consumer —
/// PW6B95, PWPB95 — carries its own parameter set, distinct from the standalone
/// `gga_x_mpw91`'s bt = 0.00426, expo = 3.72).
struct GgaXMpw91 {
    info: FunctionalInfo,
    zeta_threshold: f64,
    bt: f64,
    alpha_w: f64,
    expo: f64,
}

/// Boxed mPW91-form exchange with the given parameters (`bt`, `alpha_w` the
/// exponential coefficient in `w`-space, `expo` the large-gradient damping
/// power). `dens_threshold` 1e-15 (libxc `gga_x_mpw91`).
pub(crate) fn mpw91_x_component(
    name: &'static str,
    bt: f64,
    alpha_w: f64,
    expo: f64,
) -> Box<dyn XcEval> {
    Box::new(Gga(GgaXMpw91 {
        info: FunctionalInfo {
            id: None,
            name,
            family: Family::Gga,
            kind: Kind::Exchange,
            needs_sigma: true,
            needs_lapl: false,
            needs_tau: false,
            dens_threshold: 1e-15, // libxc gga_x_mpw91 threshold
            hybrid: None,
        },
        zeta_threshold: f64::EPSILON,
        bt,
        alpha_w,
        expo,
    }))
}

impl GgaEnergy for GgaXMpw91 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        gga_exchange(&v, self.info.dens_threshold, self.zeta_threshold, |w| {
            mpw91_enhancement(w, self.bt, self.alpha_w, self.expo)
        })
    }
}

/// Internal B95 correlation component (not a registered id: each consumer —
/// PW6B95, PWPB95 — carries its own `(c_ss, c_opp)`, distinct from the
/// standalone `mgga_c_bc95`'s c_ss = 0.038, c_opp = 0.0031).
struct MggaCBc95 {
    info: FunctionalInfo,
    zeta_threshold: f64,
    c_ss: f64,
    c_opp: f64,
}

/// Boxed B95 correlation with the given `(c_ss, c_opp)`. `dens_threshold`
/// 1e-14 (libxc `mgga_c_bc95`).
pub(crate) fn bc95_c_component(name: &'static str, c_ss: f64, c_opp: f64) -> Box<dyn XcEval> {
    Box::new(Mgga(MggaCBc95 {
        info: FunctionalInfo {
            id: None,
            name,
            family: Family::Mgga,
            kind: Kind::Correlation,
            needs_sigma: true,
            needs_lapl: false,
            needs_tau: true,
            dens_threshold: 1e-14, // libxc mgga_c_bc95 threshold
            hybrid: None,
        },
        zeta_threshold: f64::EPSILON,
        c_ss,
        c_opp,
    }))
}

impl MggaEnergy for MggaCBc95 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let zt = self.zeta_threshold;
        let thr = self.info.dens_threshold;
        let MggaVars {
            rs,
            z,
            opz,
            omz,
            na,
            nb,
            xs0_sq,
            xs1_sq,
            t0,
            t1,
            ..
        } = v;

        // Stoll same-spin (parallel) LDA correlation per channel, screened like
        // libxc's `screen_dens_zeta` (shared pattern with M06-L correlation).
        let up_screened = na.re() <= thr || opz.re() <= zt;
        let dn_screened = nb.re() <= thr || omz.re() <= zt;
        let par_up = if up_screened {
            N::from(0.0)
        } else {
            opz / N::from(2.0) * f_pw(rs * (N::from(2.0) / opz).powf(1.0 / 3.0), N::from(1.0), zt)
        };
        let par_dn = if dn_screened {
            N::from(0.0)
        } else {
            omz / N::from(2.0) * f_pw(rs * (N::from(2.0) / omz).powf(1.0 / 3.0), N::from(-1.0), zt)
        };
        let perp = f_pw(rs, z, zt) - par_up - par_dn;

        // bc95_gpar(x², t) = t·Fermi_D(x², t)/(K·(1 + c_ss·x²)²), sqrt-free; the
        // 8t denominator of Fermi_D is > 0 (τ floored) and the FHC clamp keeps
        // Fermi_D ∈ [0, 1].
        let gpar = |w: N, t: N| {
            let d = N::from(1.0) + w * self.c_ss;
            t * fermi_d(w, t) / (d * d * K_FACTOR_C)
        };
        let up = if up_screened {
            N::from(0.0)
        } else {
            par_up * gpar(xs0_sq, t0)
        };
        let dn = if dn_screened {
            N::from(0.0)
        } else {
            par_dn * gpar(xs1_sq, t1)
        };

        // bc95_gperp = 1/(1 + c_opp·(x₀² + x₁²)) — denominator ≥ 1, pole-free.
        let cross = perp / (N::from(1.0) + (xs0_sq + xs1_sq) * self.c_opp);

        up + dn + cross
    }
}

/// Build PW6B95 as the libxc mix `0.72·mPW91-x(PW6) + 1.0·B95-c(PW6)`; each
/// component screens at its own threshold inside [`mixed_eval`], exactly as
/// libxc's `xc_mix` does.
pub(crate) fn pw6b95() -> Result<Box<dyn XcEval>, XcError> {
    let info = FunctionalInfo {
        id: Some(FunctionalId::HybMggaXcPw6b95),
        name: "hyb_mgga_xc_pw6b95",
        family: Family::HybMgga,
        kind: Kind::ExchangeCorrelation,
        needs_sigma: true,
        needs_lapl: false,
        needs_tau: true,
        dens_threshold: 1e-14, // libxc hyb_mgga_xc_pw6b95 threshold
        hybrid: Some(HybridInfo {
            exx_fraction: EXX_FRACTION,
            cam: None,
            vv10: None,
        }),
    };
    Ok(mixed_eval(
        vec![
            (
                MPW91_WEIGHT,
                mpw91_x_component("gga_x_mpw91 (PW6B95 parameters)", BT, ALPHA_W, EXPO),
            ),
            (
                1.0,
                bc95_c_component("mgga_c_bc95 (PW6B95 parameters)", C_SS, C_OPP),
            ),
        ],
        info,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pw6(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybMggaXcPw6b95, spin).unwrap()
    }

    /// Metadata: hybrid meta-GGA XC with 28% EXX (Zhao & Truhlar 2005).
    #[test]
    fn metadata_reports_28_percent_exx() {
        let f = pw6(Spin::Unpolarized);
        assert_eq!(f.exx_fraction(), 0.28);
        assert_eq!(f.info().name, "hyb_mgga_xc_pw6b95");
        assert!(f.info().needs_tau && f.info().needs_sigma);
    }

    /// mPW91 enhancement sanity: F(0) = 1 exactly (LDA limit; num(0) = 0), and
    /// F grows with w at small w (the gradient expansion has positive c − slope).
    #[test]
    fn mpw91_enhancement_lda_limit_and_slope() {
        assert_eq!(mpw91_enhancement(0.0_f64, BT, ALPHA_W, EXPO), 1.0);
        let f1 = mpw91_enhancement(0.1_f64, BT, ALPHA_W, EXPO);
        assert!(f1 > 1.0, "F(0.1) = {f1} must exceed 1");
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = pw6(Spin::Unpolarized);
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
        let f = pw6(Spin::Polarized);
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
        // neither component uses σ_ab ⇒ vsigma_ab = 0
        assert_eq!(out.vsigma[1], 0.0, "PW6B95 vsigma_ab must be 0");
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
        let up = pw6(Spin::Unpolarized);
        let po = pw6(Spin::Polarized);
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
        let f = pw6(Spin::Polarized);
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
