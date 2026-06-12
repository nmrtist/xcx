// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Becke 88 exchange — `gga_x_b88` (libxc 106).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_x_b88.mpl` +
//! `maple/util.mpl` (`gga_exchange`, `lda_x_spin`, `xc_asinh`).
//!
//! Like PBE exchange, B88 is the LDA exchange of each spin channel times an
//! enhancement factor `F_x(x_σ)` — so it reuses the shared `gga_exchange`
//! skeleton (screen + `lda_x_spin` + per-channel sum). Only the enhancement
//! differs: B88's `F_x` uses the reduced gradient `x` directly (no `X2S`
//! prefactor, unlike PBE's `s = X2S·x`). It consumes the **squared** gradient
//! `t = x²` (sqrt-free) and routes the `x·asinh(x)` term through [`b88_g`], a
//! power series in `t` near 0, so the second derivative `v2sigma2` stays accurate
//! as σ → 0 (divergence #4; see [`crate::reduced::vars::reduced_grad_sq`]).

use num_dual::DualNum;

use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::X_FACTOR_C;

// B88 parameters (libxc `b88_values = {0.0042, 6.0}`): `beta` is the gradient-
// expansion coefficient (as `beta/X_FACTOR_C`); `gamma = 6` fixes the large-`x`
// asymptotics of E_x.
const BETA: f64 = 0.0042;
const GAMMA: f64 = 6.0;

/// Switch point (in `t = x²`) between the power series for `g(t) = x·asinh(x)`
/// and the direct `√t·asinh(√t)`. Below it the direct form's `√t` would carry a
/// `~t^(-3/2)` second derivative that destroys fxc accuracy; above it `√t` is
/// bounded away from 0 and harmless. At `t = 0.1` the two forms agree to ~4e-19.
const T_SWITCH: f64 = 0.1;

/// Taylor coefficients of `g(t) = √t·asinh(√t) = Σ_{k≥0} G[k]·t^(k+1)` about
/// `t = 0` (so `g = t − t²/6 + 3t³/40 − 5t⁴/112 + …`). `G[k] = (−1)^k (2k)! /
/// (4^k (k!)² (2k+1))` (the `asinh` series, shifted one power). 16 terms make the
/// series and the direct form agree to ~4e-19 (value) and ~4e-15 (second
/// derivative) at `T_SWITCH = 0.1` — far under the 1e-10 fxc tolerance. Verified
/// against the recurrence `G[k] = G[k−1]·(−(2k−1)²/(2k(2k+1)))` in the unit test.
const G_COEFFS: [f64; 16] = [
    1.0,
    -0.166_666_666_666_666_66,
    0.075,
    -0.044_642_857_142_857_144,
    0.030_381_944_444_444_444,
    -0.022_372_159_090_909_092,
    0.017_352_764_423_076_924,
    -0.013_964_843_75,
    0.011_551_800_896_139_705,
    -0.009_761_609_529_194_078,
    0.008_390_335_809_616_815,
    -0.007_312_525_873_598_845_4,
    0.006_447_210_311_889_649,
    -0.005_740_037_670_841_924,
    0.005_153_309_682_319_905,
    -0.004_660_143_486_915_096,
];

/// `g(t) = √t·asinh(√t) = x·asinh(x)` with `t = x²`. **Analytic in `t`** (its
/// Taylor series has only integer powers of `t`), which is why B88's enhancement
/// is analytic in `t` and its `v2sigma2` is finite at σ = 0. Below `T_SWITCH`
/// evaluate the series (sqrt-free → forward-AD's second derivative stays accurate
/// at small σ, divergence #4); at/above it use the direct form, where `√t` is far
/// from 0 and its derivatives are harmless. `g(0) = 0` exactly (the series has no
/// constant term), so `F_x(σ = 0) = 1` (the LDA limit) is preserved exactly.
pub(crate) fn b88_g<N: DualNum<f64> + Copy>(t: N) -> N {
    if t.re() < T_SWITCH {
        // Horner on the inner polynomial Σ G[k]·t^k, then ×t to get Σ G[k]·t^(k+1).
        let mut p = N::from(G_COEFFS[G_COEFFS.len() - 1]);
        for &c in G_COEFFS[..G_COEFFS.len() - 1].iter().rev() {
            p = p * t + N::from(c);
        }
        p * t
    } else {
        let x = t.sqrt();
        x * x.asinh()
    }
}

/// B88 exchange enhancement `F_x` as a function of the **squared** reduced
/// gradient `t = x²`: `F = 1 + (β/X_FACTOR_C)·t / (1 + γβ·g(t))`, with
/// `g(t) = √t·asinh(√t) = x·asinh(x)` (libxc's `b88_f`; the maple feeds the
/// magnitude `x = √σ/n^(4/3)` straight in, no `X2S`).
///
/// Written as `1 + m1` directly (matching libxc's `b88_f := 1 + b88_f_m1`): we
/// need `F`, not `F − 1`, so the energy never forms a `(1 + tiny) − 1`
/// cancellation. Passing `t` (not the magnitude `x`) and routing `x·asinh(x)`
/// through [`b88_g`] keeps the AD `v2sigma2` accurate as σ → 0: the maple forms
/// `x = √σ/n^(4/3)` whose second derivative diverges `~ σ^(-3/2)`, destroying the
/// finite-limit cancellation in f64 (divergence #4). This is an algebraic
/// identity for the energy and `vxc`; only `fxc` changes, becoming clean.
fn b88_enhancement<N: DualNum<f64> + Copy>(t: N) -> N {
    let denom = N::from(1.0) + N::from(GAMMA * BETA) * b88_g(t); // 1 + γβ·g(t)
    N::from(1.0) + N::from(BETA / X_FACTOR_C) * t / denom
}

pub(crate) struct GgaXB88 {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaXB88 {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaXB88),
                name: "gga_x_b88",
                family: Family::Gga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaXB88 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // GGA exchange = per-channel LDA exchange × B88 enhancement, screened on
        // the floored spin density (shared `gga_exchange` skeleton; the
        // enhancement is this functional's only contribution).
        gga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            b88_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{b88_g, G_COEFFS, T_SWITCH};
    use crate::{Functional, FunctionalId, Spin, XcInput};

    /// The hardcoded `g(t)` Taylor coefficients must equal the closed-form
    /// recurrence `G[k] = G[k−1]·(−(2k−1)²/(2k(2k+1)))`, `G[0] = 1` — so the
    /// literals are provably the `√t·asinh(√t)` series (cf. the consts.rs pattern).
    #[test]
    fn g_coeffs_match_recurrence() {
        assert_eq!(G_COEFFS[0], 1.0);
        let mut prev = 1.0_f64;
        for (k, &g) in G_COEFFS.iter().enumerate().skip(1) {
            let kk = k as f64;
            prev *= -((2.0 * kk - 1.0).powi(2)) / (2.0 * kk * (2.0 * kk + 1.0));
            assert!(
                (g - prev).abs() <= 1e-15 * prev.abs(),
                "G[{k}] = {g} vs recurrence {prev}"
            );
        }
    }

    /// The series branch (`t < T_SWITCH`) and the direct `√t·asinh(√t)` must
    /// agree across the switch, and `g(0) = 0` exactly (so `F_x(σ=0) = 1`).
    #[test]
    fn b88_g_series_matches_direct() {
        assert_eq!(b88_g(0.0_f64), 0.0, "g(0) must be exactly 0");
        for &t in &[1e-8_f64, 1e-3, 0.05, 0.09, T_SWITCH * (1.0 - 1e-12)] {
            let direct = t.sqrt() * t.sqrt().asinh();
            assert!(
                (b88_g(t) - direct).abs() <= 1e-13 * direct.abs(),
                "b88_g({t}) series {} vs direct {direct}",
                b88_g(t)
            );
        }
        // and the direct branch agrees with itself just above the switch
        let t = T_SWITCH * (1.0 + 1e-12);
        assert!((b88_g(t) - t.sqrt() * t.sqrt().asinh()).abs() <= 1e-15);
    }

    fn b88(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXB88, spin).unwrap()
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = b88(Spin::Unpolarized);
        let edens = |n: f64, s: f64| n * f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
        for &(n, s) in &[(0.5, 0.1), (2.0, 0.7), (0.1, 0.02), (10.0, 5.0)] {
            let out = f.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
            let hn = 1e-6 * n;
            let hs = 1e-6 * s;
            let fdn = (edens(n + hn, s) - edens(n - hn, s)) / (2.0 * hn);
            let fds = (edens(n, s + hs) - edens(n, s - hs)) / (2.0 * hs);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-6 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-6 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s}: {} vs {fds}",
                out.vsigma[0]
            );
        }
    }

    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = b88(Spin::Polarized);
        let (na, nb, saa, sab, sbb) = (0.6, 0.3, 0.1, 0.05, 0.08);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let edens = |r: [f64; 2], s: [f64; 3]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s)).unwrap();
        for (k, h) in [(0usize, 1e-6 * na), (1, 1e-6 * nb)] {
            let mut rp = r;
            let mut rm = r;
            rp[k] += h;
            rm[k] -= h;
            let fd = (edens(rp, s) - edens(rm, s)) / (2.0 * h);
            assert!(
                (out.vrho[k] - fd).abs() <= 1e-6 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp) - edens(r, sm)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-6 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
        // Pure exchange has no σ_ab dependence: ∂e/∂σ_ab must be exactly zero.
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
    }

    /// At σ = 0 the enhancement F_x → 1, so B88 exchange must recover Slater
    /// (lda_x) — the GGA→LDA limit — for both energy and potential.
    #[test]
    fn sigma_zero_recovers_lda_x() {
        let pu = b88(Spin::Unpolarized);
        let lu = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let p = pu.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap();
            let l = lu.eval(1, &XcInput::lda(&[n])).unwrap();
            assert!(
                (p.exc[0] - l.exc[0]).abs() <= 1e-10 * l.exc[0].abs(),
                "exc n={n}: {} vs {}",
                p.exc[0],
                l.exc[0]
            );
            assert!(
                (p.vrho[0] - l.vrho[0]).abs() <= 1e-10 * l.vrho[0].abs(),
                "vrho n={n}: {} vs {}",
                p.vrho[0],
                l.vrho[0]
            );
        }
        let pp = b88(Spin::Polarized);
        let lp = Functional::new(FunctionalId::LdaX, Spin::Polarized).unwrap();
        let p = pp
            .eval(1, &XcInput::gga(&[0.6, 0.3], &[0.0, 0.0, 0.0]))
            .unwrap();
        let l = lp.eval(1, &XcInput::lda(&[0.6, 0.3])).unwrap();
        assert!((p.exc[0] - l.exc[0]).abs() <= 1e-10 * l.exc[0].abs());
        assert!((p.vrho[0] - l.vrho[0]).abs() <= 1e-10 * l.vrho[0].abs());
        assert!((p.vrho[1] - l.vrho[1]).abs() <= 1e-10 * l.vrho[1].abs());
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = b88(Spin::Unpolarized);
        let po = b88(Spin::Polarized);
        let (n, s) = (0.8, 0.3);
        let ou = up.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
        let op = po
            .eval(
                1,
                &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0]),
            )
            .unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-12 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-11 * ou.vrho[0].abs());
    }

    #[test]
    fn edge_outputs_finite() {
        let f = b88(Spin::Polarized);
        let rho = [
            1.0, 0.0, // ζ = +1, full polarization
            0.0, 1.0, // ζ = −1
            1e-12, 1e-13, // small densities
            1.0, 1.0, // unpolarized-like
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0 at full polarization
            0.0, 0.0, 0.0, //
            1e-20, 0.0, 1e-22, // tiny σ, tiny densities
            1e6, 1e6, 1e6, // very large σ (B88 F_x diverges; must stay finite)
            1.0, 0.5, 0.8, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
