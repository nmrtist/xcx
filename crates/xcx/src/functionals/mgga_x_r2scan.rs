// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Re-regularized SCAN exchange — `mgga_x_r2scan` (libxc 497).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_x_r2scan.mpl` +
//! `maple/mgga_exc/mgga_x_scan.mpl` + `maple/util.mpl` (`mgga_exchange`,
//! `lda_x_spin`, `X2S`, `K_FACTOR_C`, `MU_GE`).
//!
//! Like TPSS/PBE exchange, r2SCAN is per-spin LDA exchange × an enhancement
//! factor `F_x(x_σ², t_σ)`, so it reuses the shared [`mgga_exchange`] skeleton. The
//! enhancement interpolates, via a switch function `f(α)`, between the
//! single-orbital limit (`α = 0`, `h0x`) and the slowly-varying limit (`h1x`):
//! `F_x = (h1x + f(α)·(h0x − h1x))·g_x`. r2SCAN is the **regularized-restored**
//! SCAN: its iso-orbital indicator `α = (t − x²/8)/(K_FACTOR_C + η·x²/8)` carries a
//! regularized denominator (η = 0.001) so it is a smooth rational function of the
//! reduced variables — no 0/0 — and `f(α)` is a degree-7 polynomial on `α ∈ [0,
//! 2.5]` with smooth `exp` tails (the rSCAN switch). This removes SCAN's
//! near-singular `α ≈ 1` switch derivatives; what remains is the **switch-class AD
//! hazard** (CLAUDE.md §3), handled by keeping every reduced gradient squared and
//! using cancellation-free algebraic forms so both vxc and fxc stay accurate.
//!
//! **Sqrt-free / AD-safe organization.** The harness seeds the *squared* reduced
//! gradient `w = x²` (`reduced_grad_sq`) and the reduced KE `t = τ_σ/n_σ^(5/3)`
//! (`reduced_tau`), both sqrt-free. Everything here consumes `w` directly: the
//! reduced gradient enters as `p = X2S²·w = s²`, `α` as `w/8`, and the only `√`-ish
//! term — the gradient damping `g_x = −expm1(−a₁/√s) = −expm1(−A·w^(−1/4))` — is
//! wrapped in an `exp` that drives it (and all its derivatives) smoothly to its
//! `w → 0` limit (`g_x → 1`), so no `1/√σ` blows up. The slowly-varying factor
//! `h1x` is written in the cancellation-free `(1+k1) − k1²/(k1+y)` form (the PBE-x
//! rational-enhancement lesson, divergence #2) so its AD derivative does not cancel
//! at large `y`. The Laplacian is unused (`needs_lapl = false`).

use num_dual::DualNum;

use crate::families::mgga::{mgga_exchange, rscan_f_alpha, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::{K_FACTOR_C, X2S};

// r2SCAN exchange parameters (libxc `par_r2scan`): the rSCAN switch constants
// c1/c2/d, the `h1x` rational parameter k1, the α-denominator regularization η, and
// the `r2scan_x` Gaussian width dp2.
const C1: f64 = 0.667;
const C2: f64 = 0.8;
const D: f64 = 1.24;
const K1: f64 = 0.065;
const ETA: f64 = 0.001;
const DP2: f64 = 0.361;

/// `MU_GE = 10/81`: the second-order gradient-expansion coefficient of exchange.
const MU_GE: f64 = 10.0 / 81.0;
/// `X2S²`: coefficient of `w = x²` in the reduced gradient `p = X2S²·w = s²`.
const X2S2: f64 = X2S * X2S;
/// SCAN's two interpolation endpoints (`mgga_x_scan.mpl`): `h0x` the single-orbital
/// (α = 0) enhancement bound, `a1` the gradient-damping decay constant.
const SCAN_H0X: f64 = 1.174;
const SCAN_A1: f64 = 4.9479;
/// `dp2⁴`, the denominator of the Gaussian in the `r2scan_x` argument (eqn S10).
const DP2_4: f64 = DP2 * DP2 * DP2 * DP2;

/// Coefficients of the rSCAN exchange switch `f(α)` polynomial, libxc's `rscan_fx`,
/// in **reversed** order `[a⁷, a⁶, …, a¹, a⁰]` (so `RSCAN_FX[7]` is the constant
/// term). Used both by the degree-7 polynomial branch and by the gradient-expansion
/// constants `Cn`/`C2` (eqns S11–S12). Provenance: `mgga_x_rscan.mpl` `rscan_fx`.
const RSCAN_FX: [f64; 8] = [
    -0.023185843322,
    0.234528941479,
    -0.887998041597,
    1.451297044490,
    -0.663086601049,
    -0.4445555,
    -0.667,
    1.0,
];

/// `Cn = 20/27 + η·5/3` (eqn S11): the gradient-expansion-restoring prefactor.
const CN: f64 = 20.0 / 27.0 + ETA * 5.0 / 3.0;
/// `Σ_{i=1}^{8} i·rscan_fx[9−i]` (the inner sum of eqn S12), spelled out 0-indexed
/// (`rscan_fx[9−i]` 1-based ⇒ `RSCAN_FX[8−i]` 0-based).
const C2_SUM: f64 = RSCAN_FX[7]
    + 2.0 * RSCAN_FX[6]
    + 3.0 * RSCAN_FX[5]
    + 4.0 * RSCAN_FX[4]
    + 5.0 * RSCAN_FX[3]
    + 6.0 * RSCAN_FX[2]
    + 7.0 * RSCAN_FX[1]
    + 8.0 * RSCAN_FX[0];
/// `Cn·C2` with `C2 = −C2_SUM·(1 − h0x)` (eqn S12): the constant coefficient of the
/// `s²·exp(−s⁴/dp2⁴)` Gaussian-damped term in the `r2scan_x` argument (eqn S10).
const CN_C2: f64 = CN * (-C2_SUM * (1.0 - SCAN_H0X));

/// r2SCAN exchange enhancement `F_x` as a function of the **squared** reduced
/// gradient `w = x²` and reduced kinetic-energy density `t = τ_σ/n_σ^(5/3)`
/// (`mgga_x_r2scan.mpl` `r2scan_f`):
/// `F_x = (h1x + f(α)·(h0x − h1x))·g_x`, with
/// `p = X2S²·w = s²`, `α = (t − w/8)/(K_FACTOR_C + η·w/8)` (regularized),
/// `y = (Cn·C2·exp(−p²/dp2⁴) + MU_GE)·p`, `h1x = (1+k1) − k1²/(k1+y)`,
/// `g_x = −expm1(−a₁/√s)`. See the module docs for the AD-safety rationale.
fn r2scan_x_enhancement<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    let p = N::from(X2S2) * w; // scan_p(x) = X2S²·x² = s²

    // r2scan_x argument (eqn S10): the Gaussian-damped gradient-expansion form.
    let y = (N::from(CN_C2) * (-(p * p) / N::from(DP2_4)).exp() + N::from(MU_GE)) * p;
    // scan_h1x in the cancellation-free (1+k1) − k1²/(k1+y) form (PBE-x, div #2).
    let h1 = N::from(1.0 + K1) - N::from(K1 * K1) / (N::from(K1) + y);

    // Regularized iso-orbital indicator (eqn S6): smooth rational, no 0/0.
    let alpha = (t - w / N::from(8.0)) / (N::from(K_FACTOR_C) + N::from(ETA) * w / N::from(8.0));
    let falpha = rscan_f_alpha(alpha, C1, C2, D, &RSCAN_FX);

    // Gradient damping g_x = −expm1(−a₁/√s) with √s = (X2S²·w)^(1/4) = X2S^(1/2)·w^(1/4).
    // Written as −expm1(−A·w^(−1/4)): the exp drives g_x → 1 (and its derivatives → 0)
    // smoothly as w → 0, so no 1/√σ singularity reaches vxc/fxc.
    let gx_a = SCAN_A1 / X2S.sqrt();
    let gx = -(-N::from(gx_a) * w.powf(-0.25)).exp_m1();

    (h1 + falpha * (N::from(SCAN_H0X) - h1)) * gx
}

pub(crate) struct MggaXR2scan {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaXR2scan {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaXR2scan),
                name: "mgga_x_r2scan",
                family: Family::Mgga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-11, // libxc mgga_x_r2scan threshold (not 1e-15)
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self::new()))
    }
}

impl MggaEnergy for MggaXR2scan {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        // meta-GGA exchange = per-channel LDA exchange × r2SCAN enhancement, screened
        // on the floored spin density (shared `mgga_exchange` skeleton).
        mgga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            r2scan_x_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CN, CN_C2};
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn r2scan(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaXR2scan, spin).unwrap()
    }

    /// The derived gradient-expansion constants (eqns S11–S12) match an independent
    /// computation from the primitive coefficients (guards the reversed-index logic).
    #[test]
    fn gradient_expansion_constants() {
        assert!((CN - (20.0 / 27.0 + 0.001 * 5.0 / 3.0)).abs() < 1e-15);
        // C2 = −(Σ i·ff[9−i])·(1−h0x); Cn·C2 reference value.
        let ff = [
            -0.023185843322,
            0.234528941479,
            -0.887998041597,
            1.451297044490,
            -0.663086601049,
            -0.4445555,
            -0.667,
            1.0,
        ];
        let mut s = 0.0;
        for i in 1..=8 {
            s += (i as f64) * ff[8 - i]; // ff[9−i] 1-based = ff[8−i] 0-based
        }
        let c2 = -s * (1.0 - 1.174);
        assert!((CN_C2 - CN * c2).abs() < 1e-14);
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
            (0.1, 0.02, 0.05),
            (10.0, 5.0, 20.0),
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
                (out.vrho[0] - fdn).abs() <= 1e-6 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s} t={tau}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-6 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s} t={tau}: {} vs {fds}",
                out.vsigma[0]
            );
            assert!(
                (out.vtau[0] - fdt).abs() <= 1e-6 * out.vtau[0].abs().max(1.0),
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
                (out.vrho[k] - fd).abs() <= 1e-6 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp, t) - edens(r, sm, t)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-6 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
        for (k, h) in [(0usize, 1e-6 * ta), (1, 1e-6 * tb)] {
            let (mut tp, mut tm) = (t, t);
            tp[k] += h;
            tm[k] -= h;
            let fd = (edens(r, s, tp) - edens(r, s, tm)) / (2.0 * h);
            assert!(
                (out.vtau[k] - fd).abs() <= 1e-6 * out.vtau[k].abs().max(1.0),
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
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-12 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-11 * ou.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = r2scan(Spin::Polarized);
        let rho = [
            1.0, 0.0, // full polarization
            0.0, 1.0, //
            1e-10, 1e-11, // small densities
            1.0, 1.0, //
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0
            0.0, 0.0, 0.0, //
            1e-20, 0.0, 1e-22, //
            1e6, 1e6, 1e6, // large σ (τ < τ_W ⇒ α < 0 before the FHC clamp)
            1.0, 0.5, 0.8, //
        ];
        let tau = [
            0.5, 0.0, // τ → floor on minority
            0.0, 0.5, //
            1e-15, 1e-16, //
            0.1, 0.1, // τ ≪ τ_W with large σ
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
