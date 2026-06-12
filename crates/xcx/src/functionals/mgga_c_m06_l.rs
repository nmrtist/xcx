// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Minnesota M06-L correlation — `mgga_c_m06_l` (libxc 233).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_c_m06l.mpl` +
//! `maple/mgga_exc/mgga_c_m05.mpl` + `maple/mgga_exc/mgga_c_vsxc.mpl` +
//! `maple/b97.mpl` (`b97_g`) + `maple/gvt4.mpl` (`gtv4`) +
//! `maple/lda_exc/lda_c_pw.mpl` (`f_pw`) + `maple/util.mpl` (`lda_stoll_par`/
//! `lda_stoll_perp`, `Fermi_D`/`Fermi_D_corrected`, `K_FACTOR_C`). Coefficient
//! tables from `src/mgga_c_m06l.c` `m06l_values`.
//!
//! M06-L correlation is the sum of an **M05** part and a **VSXC** part
//! (`mgga_c_m06l.mpl`: `m06l_f = m05_f + vsxc_f`). Both decompose the uniform-gas
//! correlation into opposite-spin (perpendicular) and same-spin (parallel) pieces
//! via the **Stoll decomposition** of the modified-PW92 LSDA energy `f_pw`:
//! ```text
//! stoll_par(rs, z) = (1+z)/2 · f_pw(rs·(2/(1+z))^(1/3), 1)   [same-spin σσ, screened]
//! stoll_perp(rs, z) = f_pw(rs, z) − stoll_par(z) − stoll_par(−z)   [opposite-spin αβ]
//! ```
//! Each piece is multiplied by a kinetic enhancement:
//! ```text
//! M05  same-spin:  stoll_par(±z) · b97_g(γ_ss, c_ss, x_σ) · Fermi_D_corrected(x_σ, t_σ)
//! M05  opp-spin:   stoll_perp(z) · b97_g(γ_ab, c_ab, √(x₀²+x₁²))
//! VSXC same-spin:  stoll_par(±z) · gtv4(α_ss, d_ss, x_σ, 2(t_σ − K)) · Fermi_D(x_σ, t_σ)
//! VSXC opp-spin:   stoll_perp(z) · gtv4(α_ab, d_ab, √(x₀²+x₁²), 2(t₀+t₁ − 2K))
//! ```
//! with `t_σ = τ_σ/n_σ^(5/3)`, `K = K_FACTOR_C`, `b97_g` the B97 GGA-correlation
//! factor, and `Fermi_D = (8t − x²)/(8t)` the (FHC-clamped, ≥ 0) Fermi-hole
//! curvature. Grouped per channel, the same `stoll_par(±z)` multiplies both the
//! M05 and VSXC same-spin terms, and `stoll_perp` both opposite-spin terms.
//!
//! **Reuse (CONTRIBUTING.md).** `f_pw` is the **shared** [`pw92_ec`] with the
//! *modified* PW92 parametrization ([`A_MOD`] + exact `f''(0)` = [`FPP_VWN`]) —
//! the same uniform limit PBE-C and r2SCAN-c use (libxc's M06-L `func_aux` is
//! `XC_LDA_C_PW_MOD`); the [`gtv4`] working term is shared with M06-L exchange.
//! Recovery test [`tests::c_uses_modified_pw92`] pins `f_pw` to the modified set
//! (and confirms it differs from standard `lda_c_pw`).
//!
//! **AD-safety.** Every reduced gradient enters **squared** (`w_σ = x_σ²`, even in
//! the opposite-spin `√(x₀²+x₁²)` — `b97_g`/`gtv4` use only `x²`, `x⁴`, so the
//! `√` cancels and we pass `x₀²+x₁²` directly, sqrt-free). `b97_g` and `gtv4` are
//! smooth rationals; `Fermi_D`'s `8t` denominator is `> 0` (τ floored), and the
//! FHC clamp keeps `x² ≤ 8t` so `Fermi_D ∈ [0, 1]`; `Fermi_D_corrected`'s
//! `−expm1(−4t²/c²)` is smooth (`→ 1` for any non-tiny t). M06-L correlation
//! depends only on the **per-spin** reduced gradients, not the total `x_t`. The
//! Laplacian is unused (`needs_lapl = false`).

use num_dual::DualNum;

use super::lda_c_pw::{pw92_ec, A_MOD};
use crate::families::mgga::{gtv4, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::{FPP_VWN, K_FACTOR_C};

/// The M06-family correlation parameter set (libxc `mgga_c_m06l_params`): the
/// whole family (M06-L, M06-2X, …) shares one functional form and differs only
/// in these coefficients (`src/mgga_c_m06l.c` value tables), so the energy
/// expression below is the single parameterized source for all of them.
pub(crate) struct M06CParams {
    pub gamma_ss: f64,
    pub gamma_ab: f64,
    pub alpha_ss: f64,
    pub alpha_ab: f64,
    pub css: [f64; 5],
    pub cab: [f64; 5],
    pub dss: [f64; 6],
    pub dab: [f64; 6],
}

// M06-L correlation coefficients (libxc `m06l_values`, `src/mgga_c_m06l.c`).
const M06_L_PAR: M06CParams = M06CParams {
    gamma_ss: 0.06,
    gamma_ab: 0.0031,
    alpha_ss: 0.00515088,
    alpha_ab: 0.00304966,
    css: [
        5.349466e-01,
        5.396620e-01,
        -3.161217e+01,
        5.149592e+01,
        -2.919613e+01,
    ],
    cab: [
        6.042374e-01,
        1.776783e+02,
        -2.513252e+02,
        7.635173e+01,
        -1.255699e+01,
    ],
    dss: [
        4.650534e-01,
        1.617589e-01,
        1.833657e-01,
        4.692100e-04,
        -4.990573e-03,
        0.0,
    ],
    dab: [
        3.957626e-01,
        -5.614546e-01,
        1.403963e-02,
        9.831442e-04,
        -3.577176e-03,
        0.0,
    ],
};
/// `params_a_Fermi_D_cnst`: the `Fermi_D_corrected` decay constant (`m06l_values`).
const FERMI_D_CNST: f64 = 1e-10;

/// Modified-PW92 LSDA correlation `f_pw(rs, z)` — the **shared** [`pw92_ec`] with
/// the modified parametrization ([`A_MOD`] + [`FPP_VWN`]). The single source of
/// M06-L's uniform limit; isolated as a named fn so the reuse recovery test can
/// pin it.
pub(crate) fn f_pw<N: DualNum<f64> + Copy>(rs: N, z: N, zeta_threshold: f64) -> N {
    pw92_ec(rs, z, zeta_threshold, &A_MOD, FPP_VWN)
}

/// B97 GGA-correlation factor `b97_g(γ, c, x) = Σ_{i=0}^{4} c[i]·u^i`, `u =
/// γ·x²/(1 + γ·x²)` (`b97.mpl` `b97_g`), via Horner. Takes the **squared** reduced
/// gradient `w = x²` directly (sqrt-free): `u = γw/(1 + γw) ∈ [0, 1)` for `w ≥ 0`,
/// so the polynomial is bounded and smooth.
pub(crate) fn b97_g<N: DualNum<f64> + Copy>(gamma: f64, c: &[f64; 5], w: N) -> N {
    let gw = N::from(gamma) * w;
    let u = gw / (N::from(1.0) + gw);
    let mut p = N::from(c[4]);
    for &ci in c[..4].iter().rev() {
        p = p * u + N::from(ci);
    }
    p
}

/// Fermi-hole curvature `Fermi_D(x, t) = (8t − x²)/(8t)` (`util.mpl` `Fermi_D`),
/// taking the **squared** reduced gradient `w = x²`. libxc's single-fraction form
/// is cancellation-free at the iso-orbital limit `x²/(8t) → 1`; the harness's FHC
/// clamp keeps `w ≤ 8t`, so `Fermi_D ∈ [0, 1]` (and is exactly 0 on the clamp
/// boundary). The `8t` denominator is `> 0` (τ floored to `1e-20`).
pub(crate) fn fermi_d<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    let eight_t = N::from(8.0) * t;
    (eight_t - w) / eight_t
}

/// `Fermi_D_corrected(x, t) = Fermi_D(x, t)·(−expm1(−4t²/c²))` (`util.mpl`,
/// `c = FERMI_D_CNST`). The factor `−expm1(−4t²/c²) = 1 − exp(−4t²/c²) ∈ [0, 1]`
/// saturates to 1 for any non-tiny `t` (`c = 1e-10` is minuscule) and → 0 as
/// `t → 0`; `exp` of a large negative argument underflows cleanly to 0 (no
/// overflow), so the term and its derivatives stay finite.
fn fermi_d_corrected<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    let arg = N::from(-4.0 / (FERMI_D_CNST * FERMI_D_CNST)) * (t * t);
    fermi_d(w, t) * (-arg.exp_m1())
}

/// The shared M06-family correlation evaluator: one energy expression, the
/// parameter set injected per id (M06-L here; M06-2X in
/// [`super::mgga_c_m06_2x`]).
pub(crate) struct MggaCM06 {
    info: FunctionalInfo,
    zeta_threshold: f64,
    par: &'static M06CParams,
}

impl MggaCM06 {
    /// Wrap a parameter set + metadata into a boxed evaluator. `zeta_threshold`
    /// is libxc's default (DBL_EPSILON) for the whole family.
    pub(crate) fn boxed(info: FunctionalInfo, par: &'static M06CParams) -> Box<dyn XcEval> {
        Box::new(Mgga(Self {
            info,
            zeta_threshold: f64::EPSILON,
            par,
        }))
    }
}

pub(crate) struct MggaCM06L;

impl MggaCM06L {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        MggaCM06::boxed(
            FunctionalInfo {
                id: Some(FunctionalId::MggaCM06L),
                name: "mgga_c_m06_l",
                family: Family::Mgga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-12, // libxc mgga_c_m06l threshold
                hybrid: None,
            },
            &M06_L_PAR,
        )
    }
}

impl MggaEnergy for MggaCM06 {
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
        let k = N::from(K_FACTOR_C);

        // Stoll same-spin (parallel) LDA correlation per channel: (1±z)/2 ·
        // f_pw(rs·(2/(1±z))^(1/3), ±1). Screened (→ 0) when the spin density is at
        // the floor or the channel is fully unpopulated (libxc `screen_dens_zeta`).
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
        // Stoll opposite-spin (perpendicular) LDA correlation (no screen of its own).
        let perp = f_pw(rs, z, zt) - par_up - par_dn;

        // Same-spin channels: stoll_par · (M05 b97·Fermi_corrected + VSXC gtv4·Fermi).
        let p = self.par;
        let up = if up_screened {
            N::from(0.0)
        } else {
            par_up
                * (b97_g(p.gamma_ss, &p.css, xs0_sq) * fermi_d_corrected(xs0_sq, t0)
                    + gtv4(p.alpha_ss, &p.dss, xs0_sq, N::from(2.0) * (t0 - k))
                        * fermi_d(xs0_sq, t0))
        };
        let dn = if dn_screened {
            N::from(0.0)
        } else {
            par_dn
                * (b97_g(p.gamma_ss, &p.css, xs1_sq) * fermi_d_corrected(xs1_sq, t1)
                    + gtv4(p.alpha_ss, &p.dss, xs1_sq, N::from(2.0) * (t1 - k))
                        * fermi_d(xs1_sq, t1))
        };

        // Opposite-spin channel: stoll_perp · (M05 b97_ab + VSXC gtv4_ab), no Fermi
        // factor. b97_g/gtv4 take the combined squared gradient x₀²+x₁² (sqrt-free).
        let w_ab = xs0_sq + xs1_sq;
        let zz_ab = N::from(2.0) * (t0 + t1 - N::from(2.0) * k);
        let cross =
            perp * (b97_g(p.gamma_ab, &p.cab, w_ab) + gtv4(p.alpha_ab, &p.dab, w_ab, zz_ab));

        up + dn + cross
    }
}

#[cfg(test)]
mod tests {
    use super::{f_pw, FPP_VWN};
    use crate::functionals::lda_c_pw::{pw92_ec, A_MOD};
    use crate::reduced::vars::rs_from_n;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn m06l(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaCM06L, spin).unwrap()
    }

    /// Reuse recovery (CONTRIBUTING.md reuse rule): M06-L correlation's LSDA piece `f_pw` must be
    /// the **modified** PW92 (`pw92_ec(&A_MOD, FPP_VWN)`, libxc `XC_LDA_C_PW_MOD`),
    /// the same shared uniform limit PBE-C/r2SCAN-c use — *not* a fork and *not* the
    /// standard `lda_c_pw` set. A swap to standard params would shift `f_pw` by
    /// ~1e-5, which this catches.
    #[test]
    fn c_uses_modified_pw92() {
        let zt = f64::EPSILON;
        let lda = Functional::new(FunctionalId::LdaCPw, Spin::Unpolarized).unwrap();
        for &n in &[0.1_f64, 1.0, 7.3, 100.0] {
            let rs = rs_from_n(n);
            // f_pw equals the shared modified pw92_ec (same function, same params).
            let want = pw92_ec(rs, 0.0_f64, zt, &A_MOD, FPP_VWN);
            let got = f_pw(rs, 0.0_f64, zt);
            assert!(
                (got - want).abs() <= 1e-14 * want.abs().max(1.0),
                "n={n}: f_pw {got} vs shared modified pw92_ec {want}"
            );
            // and differs from the standard lda_c_pw set by ~1e-5 (modified ≠ std).
            let std = lda.eval(1, &XcInput::lda(&[n])).unwrap().exc[0];
            assert!(
                (got - std).abs() > 1e-7 * std.abs(),
                "n={n}: f_pw unexpectedly equals standard lda_c_pw"
            );
        }
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = m06l(Spin::Unpolarized);
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
        let f = m06l(Spin::Polarized);
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
        // correlation depends on σ_aa and σ_bb (per spin); σ_ab only enters the
        // total gradient, which M06-L correlation does not use ⇒ vsigma_ab = 0.
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
        assert_eq!(out.vsigma[1], 0.0, "M06-L correlation vsigma_ab must be 0");
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
        let up = m06l(Spin::Unpolarized);
        let po = m06l(Spin::Polarized);
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
        let f = m06l(Spin::Polarized);
        let rho = [
            1.0, 0.0, // full polarization
            0.0, 1.0, //
            1e-10, 1e-11, // small densities
            1.0, 1.0, //
            100.0, 50.0, //
        ];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1.0, 0.5, 0.8, //
        ];
        let tau = [
            0.5, 0.0, //
            0.0, 0.5, //
            1e-12, 1e-13, //
            0.5, 0.5, //
            50.0, 30.0, //
        ];
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
