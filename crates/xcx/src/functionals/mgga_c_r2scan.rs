// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Re-regularized SCAN correlation — `mgga_c_r2scan` (libxc 498).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_c_r2scan.mpl` +
//! `maple/mgga_exc/mgga_c_rscan.mpl` + `maple/mgga_exc/mgga_c_scan.mpl` +
//! `maple/lda_exc/lda_c_pw.mpl` + `maple/util.mpl`.
//!
//! r2SCAN correlation interpolates, via the same switch `f(α)` as the exchange
//! ([`rscan_f_alpha`], correlation constants), between a slowly-varying branch
//! `ε_c¹` (α ≈ 1) and a single-orbital branch `ε_c⁰` (α = 0):
//! `ε_c = ε_c¹ + f(α)·(ε_c⁰ − ε_c¹)` (eqn S24/S35). The iso-orbital indicator
//! `α = (τ̄ − x_t²/8)/(K_FACTOR_C·d_s(z) + η·x_t²/8)` (eqn S6) carries r2SCAN's
//! regularized denominator (η = 0.001), so it is a smooth rational function of the
//! reduced variables — no 0/0.
//!
//! **Reuse (CONTRIBUTING.md reuse rule).** The uniform limit `ε_c¹ = f_pw + f_H` is built on
//! the **shared** [`pw92_ec`] with the *modified* PW92 parametrization
//! ([`A_MOD`] + [`FPP_VWN`]) — the *same* `f_pw` libxc's r2SCAN uses (both
//! `lda_c_pw_params` and `lda_c_pw_modified_params` are defined, so modified wins),
//! and the same uniform limit PBE-C uses. `f_H`'s `mgamma`/`mbeta` reuse PBE-C's
//! [`GAMMA`]/[`BETA`] literals; `t²` is the shared [`tt_sq`]; `φ`, `f_ζ`, `t_total`,
//! `(1±z)^p`, `1−z^12` are the shared reduced-variable helpers. The switch `f(α)`
//! is the shared [`rscan_f_alpha`] with the correlation `(c1, c2, d)` and
//! coefficients. A recovery test pins the σ → 0 limit to the shared `f_pw`.
//!
//! **Embedded rs-derivatives (the only non-obvious mechanic).** The
//! gradient-expansion-restoring term `DC2` (eqn S34) contains the analytic
//! `∂/∂rs` of the two LSDA energies (`r2scan_delsda0`/`delsda1`). Rather than
//! hand-derive `∂ε_c/∂rs` (and risk a second, drifting analytic derivative), xcx
//! evaluates each LSDA energy over a *nested* forward dual seeded on `rs`
//! ([`d_drs`]) and reads off `.eps` — the exact same `∂/∂rs` libxc's maple emits,
//! reusing [`pw92_ec`]/`scan_eclda0` verbatim. The outer gradient/Hessian (vxc/fxc)
//! then differentiates the whole expression — embedded derivative included —
//! correctly, because the nested dual carries the outer sensitivities through
//! `rs`/`z`.
//!
//! **AD-safety.** Every reduced gradient enters squared/sqrt-free (`s² = XT2S²·x_t²`,
//! `t² = tt_sq`); the `f_H`/`ε_c⁰` gradient corrections use libxc's `expm1`/`log1p`
//! forms so the `w1 → 0` (low-density) and small-gradient limits stay
//! cancellation-free; the `w1` factor is divided out of `y − dy` once
//! (`(Y0 − DY0)/w1`, algebraically identical, finite deeper than libxc's separate
//! `Y0/w1 − DY0/w1`). The switch-seam derivative jumps are libxc's exact behavior.
//! The Laplacian is unused (`needs_lapl = false`). r2SCAN correlation depends only
//! on the **total** reduced gradient `x_t²` (not the per-spin `xs0/xs1`).

use num_dual::{Dual, DualNum};

use super::gga_c_pbe::{BETA, GAMMA};
use super::lda_c_pw::{pw92_ec, A_MOD};
use crate::families::mgga::{rscan_f_alpha, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::{FPP_VWN, K_FACTOR_C, XT2S};
use crate::reduced::vars::{f_zeta, mphi, one_minus_z_pow12, opz_pow, t_total, tt_sq};

/// r2SCAN correlation regularization parameter η (libxc `r2scan_values`).
const ETA: f64 = 0.001;
/// `r2scan_x` / `r2scan_dy` Gaussian width dp2 and its 4th power (eqn S34).
const DP2: f64 = 0.361;
const DP2_4: f64 = DP2 * DP2 * DP2 * DP2;
/// `χ∞` of `scan_one_minus_g_infty` (mgga_c_scan.mpl `scan_chi_infty`).
const CHI_INFTY: f64 = 0.128_025_852_626_258_15;
/// SCAN ε_c^LDA0 parameters (mgga_c_scan.mpl): `−b1c/(1 + b2c√rs + b3c·rs)`.
const B1C: f64 = 0.0285764;
const B2C: f64 = 0.0889;
const B3C: f64 = 0.125541;
/// `scan_Gc` spin-decay constant (mgga_c_scan.mpl `scan_G_cnst`).
const G_CNST: f64 = 2.363;
/// Correlation switch `f(α)` constants (baked in mgga_c_scan.mpl: c1/c2/d).
const FA_C1: f64 = 0.64;
const FA_C2: f64 = 1.5;
const FA_D: f64 = 0.7;
/// `XT2S²`: `s² = XT2S²·x_t²` (total-density reduced gradient, sqrt-free).
const XT2S2: f64 = XT2S * XT2S;

/// Coefficients of the rSCAN **correlation** switch `f(α)` polynomial (libxc
/// `rscan_fc`), reversed order `[a⁷, …, a⁰]` (`RSCAN_FC[7]` is the constant term).
/// Provenance: `mgga_c_rscan.mpl` `rscan_fc`.
const RSCAN_FC: [f64; 8] = [
    -0.051848879792,
    0.516884468372,
    -1.915710236206,
    3.061560252175,
    -1.535685604549,
    -0.4352,
    -0.64,
    1.0,
];
/// `Σ_{i=1}^{7} i·rscan_fc[8−i]` (eqn S27 `r2scan_dfc2`), spelled out 0-indexed
/// (`rscan_fc[8−i]` 1-based ⇒ `RSCAN_FC[7−i]` 0-based).
const DFC2: f64 = RSCAN_FC[6]
    + 2.0 * RSCAN_FC[5]
    + 3.0 * RSCAN_FC[4]
    + 4.0 * RSCAN_FC[3]
    + 5.0 * RSCAN_FC[2]
    + 6.0 * RSCAN_FC[1]
    + 7.0 * RSCAN_FC[0];

/// `(value, ∂value/∂rs)` of `g(rs, z)` via a *nested* forward dual seeded on `rs`
/// (eps = 1) with `z` held constant (eps = 0). Reads off the exact analytic
/// `∂g/∂rs` libxc's maple emits, reusing `g` verbatim — and, because the seeded
/// `rs`/`z` are themselves the outer dual `N` (carrying `d/d(n,σ,τ)`), the returned
/// derivative is itself differentiable by the outer AD (so vxc/fxc are correct).
fn d_drs<N, G>(rs: N, z: N, g: G) -> (N, N)
where
    N: DualNum<f64> + Copy,
    G: Fn(Dual<N, f64>, Dual<N, f64>) -> Dual<N, f64>,
{
    let out = g(Dual::new(rs, N::from(1.0)), Dual::new(z, N::from(0.0)));
    (out.re, out.eps)
}

/// SCAN ε_c^LDA0: `−b1c/(1 + b2c·√rs + b3c·rs)` (mgga_c_scan.mpl `scan_eclda0`).
fn scan_eclda0<N: DualNum<f64> + Copy>(rs: N) -> N {
    -N::from(B1C) / (N::from(1.0) + N::from(B2C) * rs.sqrt() + N::from(B3C) * rs)
}

/// SCAN spin-interpolation `G_c(z) = (1 − G·(2^(1/3)−1)·f_ζ(z))·(1 − z^12)`
/// (mgga_c_scan.mpl `scan_Gc`). Vanishes at full polarization via `1 − z^12`.
fn scan_gc<N: DualNum<f64> + Copy>(z: N, zeta_threshold: f64) -> N {
    let cbrt2_m1 = 2.0_f64.cbrt() - 1.0;
    (N::from(1.0) - N::from(G_CNST * cbrt2_m1) * f_zeta(z, zeta_threshold)) * one_minus_z_pow12(z)
}

/// SCAN ε_c⁰ gradient piece `H0(rs, s²)` (mgga_c_scan.mpl `scan_H0`): the
/// `b1c·log1p(expm1(−eclda0/b1c)·(1 − g∞))` form, with `1 − g∞ =
/// −expm1(−¼·log1p(4·χ∞·s²))`. Takes the **squared** reduced gradient `s²`.
fn scan_h0<N: DualNum<f64> + Copy>(rs: N, s2: N) -> N {
    let one_minus_g_infty = -(N::from(-0.25) * (N::from(4.0 * CHI_INFTY) * s2).ln_1p()).exp_m1();
    let eclda0 = scan_eclda0(rs);
    let inner = (-eclda0 / N::from(B1C)).exp_m1() * one_minus_g_infty;
    N::from(B1C) * inner.ln_1p()
}

/// SCAN single-orbital correlation ε_c⁰ `= (ε_c^LDA0(rs) + H0(rs, s²))·G_c(z)`
/// (mgga_c_scan.mpl `scan_e0`), the α = 0 branch of r2SCAN-c.
fn scan_e0<N: DualNum<f64> + Copy>(rs: N, z: N, s2: N, zeta_threshold: f64) -> N {
    (scan_eclda0(rs) + scan_h0(rs, s2)) * scan_gc(z, zeta_threshold)
}

pub(crate) struct MggaCR2scan {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaCR2scan {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaCR2scan),
                name: "mgga_c_r2scan",
                family: Family::Mgga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-15, // libxc mgga_c_r2scan threshold
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self::new()))
    }
}

impl MggaEnergy for MggaCR2scan {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let zt = self.zeta_threshold;
        let MggaVars {
            rs, z, xt2, t0, t1, ..
        } = v;

        // Reduced gradients (sqrt-free): SCAN-family s² and PBE t².
        let mphi_z = mphi(z, zt);
        let mphi3 = mphi_z * mphi_z * mphi_z;
        let s2 = N::from(XT2S2) * xt2;
        let t2 = tt_sq(rs, xt2, mphi_z);

        // --- ε_c¹ = f_pw + f_H (slowly-varying / α ≈ 1 branch) ---
        let fpw = pw92_ec(rs, z, zt, &A_MOD, FPP_VWN);
        // w1 = expm1(−f_pw/(mgamma·φ³)); f_pw < 0 ⇒ argument > 0 ⇒ w1 > 0.
        let w1 = (-fpw / (N::from(GAMMA) * mphi3)).exp_m1();

        // Embedded ∂/∂rs of the two LSDA energies (eqn S34), via nested duals.
        let (elsda0, delsda0) = d_drs(rs, z, |r, zz| scan_eclda0(r) * scan_gc(zz, zt));
        let (elsda1, delsda1) = d_drs(rs, z, |r, zz| pw92_ec(r, zz, zt, &A_MOD, FPP_VWN));
        // r2scan_d(z) = (½)·[(1+z)^(5/3) + (1−z)^(5/3)] (eqn S28).
        let r2d = (opz_pow(N::from(1.0) + z, 5.0 / 3.0, zt)
            + opz_pow(N::from(1.0) - z, 5.0 / 3.0, zt))
            / N::from(2.0);
        // DC2 numerator (eqn S34).
        let dc2 =
            N::from(20.0) * rs * (delsda0 - delsda1) - N::from(45.0 * ETA) * (elsda0 - elsda1);
        // mbeta(rs) = β·(1 + 0.1 rs)/(1 + 0.1778 rs) (eqn S33, Hu–Langreth).
        let mbeta = N::from(BETA) * (N::from(1.0) + N::from(0.1) * rs)
            / (N::from(1.0) + N::from(0.1778) * rs);

        // y − dy with the common 1/w1 factored out once: (Y0 − DY0)/w1.
        let y0 = mbeta * t2 / N::from(GAMMA);
        let dy0 = N::from(DFC2) / (N::from(27.0) * N::from(GAMMA) * r2d * mphi3)
            * dc2
            * s2
            * (-(s2 * s2) / N::from(DP2_4)).exp(); // exp(−s⁴/dp2⁴)
        let ydiff = (y0 - dy0) / w1;
        // 1 − g = −expm1(−¼·log1p(4·(y − dy))); f_H = mgamma·φ³·log1p(w1·(1 − g)).
        let one_minus_g = -(N::from(-0.25) * (N::from(4.0) * ydiff).ln_1p()).exp_m1();
        let fh = N::from(GAMMA) * mphi3 * (w1 * one_minus_g).ln_1p();
        let ec1 = fpw + fh;

        // --- ε_c⁰ (single-orbital / α = 0 branch) ---
        let ec0 = scan_e0(rs, z, s2, zt);

        // --- regularized iso-orbital indicator α (eqn S6) and the switch ---
        let tau_bar = t_total(z, t0, t1);
        let alpha = (tau_bar - xt2 / N::from(8.0))
            / (N::from(K_FACTOR_C) * t_total(z, N::from(1.0), N::from(1.0))
                + N::from(ETA) * xt2 / N::from(8.0));
        let falpha = rscan_f_alpha(alpha, FA_C1, FA_C2, FA_D, &RSCAN_FC);

        ec1 + falpha * (ec0 - ec1)
    }
}

#[cfg(test)]
mod tests {
    use super::{d_drs, scan_eclda0, A_MOD, FPP_VWN};
    use crate::functionals::lda_c_pw::pw92_ec;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn r2scan(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaCR2scan, spin).unwrap()
    }

    /// The nested-dual embedded `∂/∂rs` (the novel mechanic) matches a central
    /// finite difference of the *same* shared `pw92_ec` — validating both that the
    /// nested dual computes the right derivative and that `f_pw` is the modified
    /// PW92 being reused (a swap to standard PW92 would shift this). This is the
    /// reuse recovery guard for the embedded-derivative term.
    #[test]
    fn embedded_drs_matches_fd_of_shared_pw92() {
        let zt = f64::EPSILON;
        for &(rs, z) in &[(1.0, 0.0), (3.7, 0.3), (0.5, -0.6), (20.0, 0.0)] {
            let (val, der) = d_drs(rs, z, |r, zz| pw92_ec(r, zz, zt, &A_MOD, FPP_VWN));
            // value equals the shared pw92_ec at (rs, z)
            let direct = pw92_ec(rs, z, zt, &A_MOD, FPP_VWN);
            assert!((val - direct).abs() <= 1e-14 * direct.abs().max(1.0));
            // derivative equals a central FD of pw92_ec in rs
            let h = 1e-6 * rs;
            let fd = (pw92_ec(rs + h, z, zt, &A_MOD, FPP_VWN)
                - pw92_ec(rs - h, z, zt, &A_MOD, FPP_VWN))
                / (2.0 * h);
            assert!(
                (der - fd).abs() <= 1e-6 * der.abs().max(1.0),
                "d(f_pw)/drs @(rs={rs}, z={z}): nested-dual {der} vs FD {fd}"
            );
        }
    }

    /// Independent FD of `scan_eclda0` (guards the closed `−b1c/(1+b2c√rs+b3c rs)`).
    #[test]
    fn scan_eclda0_fd() {
        for &rs in &[0.5_f64, 1.0, 5.0, 50.0] {
            let (val, der) = d_drs(rs, 0.0, |r, _z| scan_eclda0(r));
            let h = 1e-6 * rs;
            let fd = (scan_eclda0(rs + h) - scan_eclda0(rs - h)) / (2.0 * h);
            assert!((der - fd).abs() <= 1e-6 * der.abs().max(1.0));
            assert!(val < 0.0); // ε_c^LDA0 is negative
        }
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = r2scan(Spin::Unpolarized);
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
            (1.0, 0.4, 4.6), // α ≈ 1
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
        let f = r2scan(Spin::Polarized);
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
        for (k, h) in [(0usize, 1e-6 * saa), (1, 1e-6 * sab), (2, 1e-6 * sbb)] {
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
        let up = r2scan(Spin::Unpolarized);
        let po = r2scan(Spin::Polarized);
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
        let f = r2scan(Spin::Polarized);
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
