// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Tao–Perdew–Staroverov–Scuseria exchange — `mgga_x_tpss` (libxc 202).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_x_tpss.mpl` +
//! `maple/tpss_x.mpl` + `maple/util.mpl` (`mgga_exchange`, `lda_x_spin`, `X2S`,
//! `K_FACTOR_C`).
//!
//! Like PBE/B88 exchange, TPSS is the LDA exchange of each spin channel times an
//! enhancement factor `F_x` — so it reuses the shared `mgga_exchange` skeleton
//! (screen + `lda_x_spin` + per-channel sum). The enhancement depends on the
//! channel's **squared** reduced gradient `x_σ²` (sqrt-free, divergence #4) and
//! its reduced kinetic-energy density `t_σ = τ_σ/n_σ^(5/3)` (τ direct, no sqrt);
//! the Laplacian is unused (`needs_lapl = false`).

use num_dual::DualNum;

use crate::families::mgga::{mgga_exchange, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::X2S;
use crate::reduced::vars::mgga_alpha;

// TPSS exchange parameters (libxc `tpss_values`): the asymptotic enhancement bound
// `kappa`, the 2nd-order-expansion coefficient `mu`, and the TPSS shape constants
// b, c, e. BLOC_a is the (constant, BLOC_b = 0) exponent `ff` of `z` in eq. 10.
const B: f64 = 0.40;
const C: f64 = 1.59096;
const E: f64 = 1.537;
const KAPPA: f64 = 0.8040;
const MU: f64 = 0.21951;
// BLOC_a = 2.0, BLOC_b = 0.0 for TPSS ⇒ the exponent `ff(z) = BLOC_a` is the
// constant 2, so `z^ff` below is just `z²` (computed as `z*z`, no `powf`).

/// `MU_GE = 10/81`: the second-order gradient-expansion coefficient of exchange.
const MU_GE: f64 = 10.0 / 81.0;
/// `X2S²`, the coefficient of the squared reduced gradient `x²` in `p = X2S²·x²`.
const X2S2: f64 = X2S * X2S;
/// `X2S⁴`, used in the AD-safe factoring of the `√(½(9/25 z² + p²))` term.
const X2S4: f64 = X2S2 * X2S2;

/// TPSS exchange enhancement `F_x` as a function of the **squared** reduced
/// gradient `w = x²` and reduced kinetic-energy density `t = τ_σ/n_σ^(5/3)`
/// (libxc `tpss_f`/`tpss_fx`, `maple/tpss_x.mpl` eqs. 7 & 10). Definitions:
/// `p = X2S²·w`, `z = w/(8t)`, `α = (t − w/8)/K_FACTOR_C` ([`mgga_alpha`]),
/// `qb = 9/20·(α−1)/√(1 + b·α(α−1)) + 2p/3`. The exponent `z^ff` is `z²`
/// (`ff = BLOC_a = 2` for TPSS, `BLOC_b = 0`).
///
/// **AD-safe factoring (the τ-ratio hazard class; docs/api-convention.md §8).** The maple's
/// `√(½(9/25 z² + p²))` is a √ whose argument → 0 as `w → 0` (`z, p ∝ w`), so its
/// forward-AD second derivative would diverge `~ w^(−3/2)` — the same trap as the
/// √σ form (divergence #4). Both `z²` and `p²` carry a factor `w²`, so the term is
/// rewritten cancellation-free as `w·√(½(9/(1600 t²) + X2S⁴))`: the remaining √ is
/// over a strictly-positive, `w`-independent quantity (τ is floored `> 0`), so it
/// never approaches 0 and `fxc` stays accurate as σ → 0. Algebraic identity for
/// `w ≥ 0`; energy/vxc match the maple to f64 reassociation, only `fxc` is repaired.
fn tpss_x_enhancement<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    let sqrt_e = E.sqrt();
    let p = N::from(X2S2) * w;
    let z = w / (N::from(8.0) * t);
    let z2 = z * z; // z^ff with ff = BLOC_a = 2
    let alpha = mgga_alpha(t, w);
    let qb_denom = (N::from(1.0) + N::from(B) * alpha * (alpha - N::from(1.0))).sqrt();
    let qb = N::from(9.0 / 20.0) * (alpha - N::from(1.0)) / qb_denom + N::from(2.0 / 3.0) * p;
    // √(½(9/25 z² + p²)) factored as w·√(½(9/(1600 t²) + X2S⁴)) — AD-safe at w → 0.
    let root = w * (N::from(0.5) * (N::from(9.0 / 1600.0) / (t * t) + N::from(X2S4))).sqrt();
    let onepz2 = N::from(1.0) + z2;
    let fxnum = (N::from(MU_GE) + N::from(C) * z2 / (onepz2 * onepz2)) * p
        + N::from(146.0 / 2025.0) * qb * qb
        - N::from(73.0 / 405.0) * qb * root
        + N::from(MU_GE * MU_GE / KAPPA) * p * p
        + N::from(2.0 * sqrt_e * MU_GE * 9.0 / 25.0) * z2
        + N::from(E * MU) * p * p * p;
    let fxden = N::from(1.0) + N::from(sqrt_e) * p;
    let fx = fxnum / (fxden * fxden);
    // tpss_f = 1 + κ·fx/(κ + fx) (the maple's inlined, cancellation-free form).
    N::from(1.0) + N::from(KAPPA) * fx / (N::from(KAPPA) + fx)
}

pub(crate) struct MggaXTpss {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaXTpss {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaXTpss),
                name: "mgga_x_tpss",
                family: Family::Mgga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self::new()))
    }
}

impl MggaEnergy for MggaXTpss {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        // meta-GGA exchange = per-channel LDA exchange × TPSS enhancement, screened
        // on the floored spin density (shared `mgga_exchange` skeleton).
        mgga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            tpss_x_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn tpss(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaXTpss, spin).unwrap()
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = tpss(Spin::Unpolarized);
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
        let f = tpss(Spin::Polarized);
        let (na, nb, saa, sab, sbb, ta, tb) = (0.6, 0.3, 0.1, 0.05, 0.08, 0.4, 0.25);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let t = [ta, tb];
        let edens = |r: [f64; 2], s: [f64; 3], t: [f64; 2]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s).with_tau(&t)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s).with_tau(&t)).unwrap();
        // vrho
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
        // vsigma_aa, vsigma_bb (pure exchange ⇒ vsigma_ab = 0)
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
        // vtau_a, vtau_b
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
        let up = tpss(Spin::Unpolarized);
        let po = tpss(Spin::Polarized);
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
        // unpol vtau = ∂e/∂τ_total = ½(vtau_a + vtau_b) (chain rule τ = τ_a + τ_b);
        // at z=0 vtau_a = vtau_b, so unpol vtau equals pol vtau_a (as for vrho).
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-11 * ou.vtau[0].abs().max(1.0));
        assert!((op.vtau[0] - op.vtau[1]).abs() <= 1e-11 * op.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = tpss(Spin::Polarized);
        let rho = [
            1.0, 0.0, // full polarization
            0.0, 1.0, //
            1e-12, 1e-13, // small densities
            1.0, 1.0, //
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0
            0.0, 0.0, 0.0, //
            1e-20, 0.0, 1e-22, //
            1e6, 1e6, 1e6, // large σ (τ < τ_W ⇒ α < 0)
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
