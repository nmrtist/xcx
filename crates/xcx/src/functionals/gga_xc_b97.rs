// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The Becke 1997 GGA exchange–correlation power series — `gga_xc_b97`
//! family, including the named **B97-3c** xc part (`gga_xc_b97_3c`,
//! libxc 327). A. D. Becke, *J. Chem. Phys.* **107**, 8554 (1997);
//! J. G. Brandenburg, C. Bannwarth, A. Hansen & S. Grimme, *J. Chem. Phys.*
//! **148**, 064104 (2018) (the B97-3c refit, Table I).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/gga_xc_b97.c`
//! (`b97_3c_values`) + `maple/gga_exc/gga_xc_b97.mpl` + `maple/b97.mpl` +
//! `maple/lda_exc/lda_x.mpl` + `maple/lda_exc/lda_c_pw.mpl` (**standard**
//! PW92 set) + `maple/util.mpl` (`lda_stoll_par`/`lda_stoll_perp`).
//!
//! ## Form
//!
//! Each ingredient is the parent LDA times a truncated power series in the
//! bounded variable `u = γ·s²/(1 + γ·s²)` (`b97.mpl` `b97_g`, the shared
//! sqrt-free [`b97_g`]; every gradient enters **squared**):
//! - exchange, per spin channel: `lda_x_spin(rs, 1±z) · g(γ_x, c_x, x_σ²)`;
//! - same-spin correlation: `stoll_par(±z)·g(γ_ss, c_ss, x_σ²)` on the Stoll
//!   split of *standard* PW92;
//! - opposite-spin correlation: `stoll_perp(z)·g(γ_os, c_os, (x₀²+x₁²)/2)`
//!   (the maple's `√(x₀²+x₁²)/√2` cancels inside `u` — sqrt-free).
//!
//! The γ's are Becke's originals (γ_x = 0.004, γ_ss = 0.2, γ_os = 0.006),
//! fixed across the whole B97 family; only the series coefficients vary, so
//! [`B97Series`] is **generic over the coefficient count** and future B97
//! variants are pure data. This is exactly the unattenuated counterpart of
//! ωB97X-V's form ([`super::hyb_gga_xc_wb97x_v`]), sharing the [`b97_g`] /
//! [`pw92_ec`] / `lda_x_spin` implementations rather than forking them.
//!
//! ## B97-3c
//!
//! The B97-3c xc part (Brandenburg et al. 2018, Table I) is the B97 series
//! refit truncated at **three terms per series** (k = 0..2), with **no exact
//! exchange** (a_x = 0 — pure GGA, rung [`crate::Rung::Gga`]). The composite
//! method's D3(BJ)/ATM dispersion and SRB short-range-bond corrections are the
//! *host's* job — xcx only evaluates the xc functional; `dispersion()` is
//! `None`.

use num_dual::DualNum;

use super::lda_c_pw::{pw92_ec, A_STD, FZ20_STD};
use super::mgga_c_m06_l::b97_g;
use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::vars;

/// Becke's exchange γ (B97 paper; libxc `gga_xc_b97.mpl` literal 0.004).
pub(crate) const GAMMA_X: f64 = 0.004;
/// Becke's same-spin correlation γ (0.2).
pub(crate) const GAMMA_SS: f64 = 0.2;
/// Becke's opposite-spin correlation γ (0.006).
pub(crate) const GAMMA_OS: f64 = 0.006;

// B97-3c series coefficients: Brandenburg, Bannwarth, Hansen & Grimme,
// J. Chem. Phys. 148, 064104 (2018), Table I (= libxc `b97_3c_values`,
// truncated at three terms per series; exact exchange a_x = 0).
const B97_3C_C_X: [f64; 3] = [1.076616, -0.469912, 3.322442];
const B97_3C_C_SS: [f64; 3] = [0.543788, -1.444420, 1.637436];
const B97_3C_C_OS: [f64; 3] = [0.635047, 5.532103, -15.301575];

/// A B97-type GGA xc power series (Becke 1997 form) with arbitrary series
/// coefficients — named functionals (B97-3c) and the public
/// [`crate::Functional::b97_xc`] constructor both build this one evaluator,
/// differing only in data.
pub(crate) struct B97Series {
    info: FunctionalInfo,
    zeta_threshold: f64,
    c_x: Vec<f64>,
    c_ss: Vec<f64>,
    c_os: Vec<f64>,
}

impl B97Series {
    fn boxed_with(
        info: FunctionalInfo,
        c_x: &[f64],
        c_ss: &[f64],
        c_os: &[f64],
    ) -> Box<dyn XcEval> {
        Box::new(Gga(Self {
            info,
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
            c_x: c_x.to_vec(),
            c_ss: c_ss.to_vec(),
            c_os: c_os.to_vec(),
        }))
    }
}

/// Metadata shared by every B97-series GGA: pure xc GGA, σ required, libxc's
/// `gga_xc_b97`-family density threshold (1e-14).
fn series_info(id: Option<FunctionalId>, name: &'static str) -> FunctionalInfo {
    FunctionalInfo {
        id,
        name,
        family: Family::Gga,
        kind: Kind::ExchangeCorrelation,
        needs_sigma: true,
        needs_lapl: false,
        needs_tau: false,
        dens_threshold: 1e-14, // libxc gga_xc_b97 family threshold
        hybrid: None,
    }
}

/// The named **B97-3c** xc part (`gga_xc_b97_3c`, libxc 327).
pub(crate) fn b97_3c() -> Box<dyn XcEval> {
    B97Series::boxed_with(
        series_info(Some(FunctionalId::GgaXcB973c), "gga_xc_b97_3c"),
        &B97_3C_C_X,
        &B97_3C_C_SS,
        &B97_3C_C_OS,
    )
}

/// A user-parameterized B97 series (behind [`crate::Functional::b97_xc`]):
/// same form, caller-supplied coefficients, no id.
pub(crate) fn b97_series(c_x: &[f64], c_ss: &[f64], c_os: &[f64]) -> Box<dyn XcEval> {
    B97Series::boxed_with(series_info(None, "gga_xc_b97_series"), c_x, c_ss, c_os)
}

impl GgaEnergy for B97Series {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        let zt = self.zeta_threshold;
        let thr = self.info.dens_threshold;
        let GgaVars {
            rs,
            z,
            opz,
            omz,
            na,
            nb,
            xs0_sq,
            xs1_sq,
            ..
        } = v;

        // Channel screens: libxc `screen_dens_zeta` (spin density at the floor,
        // or the channel fully unpopulated) — as in ωB97X-V.
        let up_screened = na.re() <= thr || opz.re() <= zt;
        let dn_screened = nb.re() <= thr || omz.re() <= zt;

        // --- B97 exchange, per channel (`b97_fpar(f_lda_x, …)`) ---
        let x_up = if up_screened {
            N::from(0.0)
        } else {
            vars::lda_x_spin(rs, opz, zt) * b97_g(GAMMA_X, &self.c_x, xs0_sq)
        };
        let x_dn = if dn_screened {
            N::from(0.0)
        } else {
            vars::lda_x_spin(rs, omz, zt) * b97_g(GAMMA_X, &self.c_x, xs1_sq)
        };

        // --- B97 correlation on the Stoll split of *standard* PW92 ---
        let f_pw = |rs: N, zz: N| pw92_ec(rs, zz, zt, &A_STD, FZ20_STD);
        let par_up = if up_screened {
            N::from(0.0)
        } else {
            opz / N::from(2.0) * f_pw(rs * (N::from(2.0) / opz).powf(1.0 / 3.0), N::from(1.0))
        };
        let par_dn = if dn_screened {
            N::from(0.0)
        } else {
            omz / N::from(2.0) * f_pw(rs * (N::from(2.0) / omz).powf(1.0 / 3.0), N::from(-1.0))
        };
        let perp = f_pw(rs, z) - par_up - par_dn;

        let c_up = if up_screened {
            N::from(0.0)
        } else {
            par_up * b97_g(GAMMA_SS, &self.c_ss, xs0_sq)
        };
        let c_dn = if dn_screened {
            N::from(0.0)
        } else {
            par_dn * b97_g(GAMMA_SS, &self.c_ss, xs1_sq)
        };
        // opposite-spin gradient: maple's √(x₀²+x₁²)/√2 squared = (x₀²+x₁²)/2
        let cross = perp * b97_g(GAMMA_OS, &self.c_os, (xs0_sq + xs1_sq) / N::from(2.0));

        x_up + x_dn + c_up + c_dn + cross
    }
}

#[cfg(test)]
mod tests {
    use crate::func::Rung;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn b97_3c(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXcB973c, spin).unwrap()
    }

    /// Metadata: B97-3c is a *pure* GGA (no exact exchange — the
    /// refit sets a_x = 0): rung Gga, exx 0, no hybrid record, standard level-3
    /// grid, and `dispersion() == None` (the host supplies the 3c D3/SRB terms).
    #[test]
    fn metadata_pure_gga_no_dispersion() {
        let f = b97_3c(Spin::Unpolarized);
        let info = f.info();
        assert_eq!(info.rung(), Rung::Gga);
        assert_eq!(f.exx_fraction(), 0.0);
        assert!(info.hybrid.is_none());
        assert_eq!(info.dispersion(), None);
        let g = info.grid();
        assert_eq!((g.level, g.grid_sensitive), (3, false));
        assert!(info.needs_sigma && !info.needs_tau);
        assert_eq!(info.name, "gga_xc_b97_3c");
        assert_eq!(FunctionalId::GgaXcB973c.as_u32(), 327);
        assert_eq!(
            FunctionalId::from_name("b97-3c"),
            Some(FunctionalId::GgaXcB973c)
        );
    }

    /// Exact law (uniform-gas / series-collapse): with every series equal to
    /// the constant 1, the B97 form reduces *identically* to LDA-x + standard
    /// PW92 correlation (the Stoll split recombines exactly), at any σ.
    #[test]
    fn unit_series_recovers_lda_x_plus_pw92() {
        let one = [1.0_f64];
        for &spin in &[Spin::Unpolarized, Spin::Polarized] {
            let b97 = Functional::b97_xc(&one, &one, &one, spin);
            let ldax = Functional::new(FunctionalId::LdaX, spin).unwrap();
            let pw = Functional::new(FunctionalId::LdaCPw, spin).unwrap();
            let (rho, sigma): (&[f64], &[f64]) = match spin {
                Spin::Unpolarized => (&[0.7], &[0.4]),
                Spin::Polarized => (&[0.6, 0.3], &[0.1, 0.05, 0.08]),
            };
            let got = b97.eval(1, &XcInput::gga(rho, sigma)).unwrap().exc[0];
            let want = ldax.eval(1, &XcInput::lda(rho)).unwrap().exc[0]
                + pw.eval(1, &XcInput::lda(rho)).unwrap().exc[0];
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "{spin:?}: unit B97 series {got} vs lda_x + pw92 {want}"
            );
        }
    }

    /// s → 0 limit: u → 0 collapses each series to its constant term, so
    /// B97-3c(σ=0) = c_x0·LDA-x + c_ss0·(par split) + c_os0·perp; cross-check
    /// against the constant-truncated series built through the public
    /// constructor (also pins the constructor's coefficient routing).
    #[test]
    fn sigma_zero_collapses_to_constant_series() {
        let f = b97_3c(Spin::Polarized);
        let g = Functional::b97_xc(&[1.076616], &[0.543788], &[0.635047], Spin::Polarized);
        let rho = [0.6, 0.3];
        let sigma = [0.0, 0.0, 0.0];
        let a = f.eval(1, &XcInput::gga(&rho, &sigma)).unwrap();
        let b = g.eval(1, &XcInput::gga(&rho, &sigma)).unwrap();
        assert_eq!(a.exc[0], b.exc[0]);
        assert_eq!(a.vrho, b.vrho);
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = b97_3c(Spin::Unpolarized);
        let edens = |n: f64, s: f64| n * f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
        for &(n, s) in &[(0.5, 0.1), (2.0, 0.7), (0.3, 0.02), (5.0, 3.0), (0.1, 0.01)] {
            let out = f.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
            let (hn, hs) = (1e-6 * n, 1e-6 * s);
            let fdn = (edens(n + hn, s) - edens(n - hn, s)) / (2.0 * hn);
            let fds = (edens(n, s + hs) - edens(n, s - hs)) / (2.0 * hs);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-5 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-5 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s}: {} vs {fds}",
                out.vsigma[0]
            );
        }
    }

    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = b97_3c(Spin::Polarized);
        let (na, nb, saa, sab, sbb) = (0.6, 0.3, 0.1, 0.05, 0.08);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let edens = |r: [f64; 2], s: [f64; 3]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s)).unwrap();
        for (k, h) in [(0usize, 1e-6 * na), (1, 1e-6 * nb)] {
            let (mut rp, mut rm) = (r, r);
            rp[k] += h;
            rm[k] -= h;
            let fd = (edens(rp, s) - edens(rm, s)) / (2.0 * h);
            assert!(
                (out.vrho[k] - fd).abs() <= 1e-5 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        // σ_ab never enters (per-spin gradients only) ⇒ vsigma_ab = 0.
        assert_eq!(out.vsigma[1], 0.0);
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp) - edens(r, sm)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-5 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
    }

    /// Spin-scaling consistency: the polarized evaluation at zero polarization
    /// must reproduce the unpolarized one (energy, vrho, and fxc diagonal).
    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = b97_3c(Spin::Unpolarized);
        let po = b97_3c(Spin::Polarized);
        let (n, s) = (0.8, 0.3);
        let ou = up.eval_fxc(1, &XcInput::gga(&[n], &[s])).unwrap();
        let op = po
            .eval_fxc(
                1,
                &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0]),
            )
            .unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-11 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
        // unpolarized v2rho2 = ½(v2rho2_aa + v2rho2_ab)·… spot-check finiteness
        for v in op.v2rho2.iter().chain(&op.v2rhosigma).chain(&op.v2sigma2) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn edge_outputs_finite() {
        let f = b97_3c(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-10, 1e-11, 1.0, 1.0, 1000.0, 500.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1e6, 5e5, 8e5, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
