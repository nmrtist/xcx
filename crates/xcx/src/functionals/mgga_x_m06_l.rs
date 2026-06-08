// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Minnesota M06-L exchange — `mgga_x_m06_l` (libxc 203).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_x_m06l.mpl` +
//! `maple/gga_exc/gga_x_pbe.mpl` (`pbe_f`) + `maple/gvt4.mpl` (`gtv4`) +
//! `maple/util.mpl` (`mgga_exchange`, `mgga_w`/`mgga_series_w`, `lda_x_spin`,
//! `K_FACTOR_C`). Coefficient tables (`a[12]`, `d[6]`) from `src/mgga_x_m06l.c`
//! `par_m06l`.
//!
//! Like TPSS/r2SCAN exchange, M06-L is per-spin LDA exchange × an enhancement
//! factor `F_x(x_σ², t_σ)` on the shared [`mgga_exchange`] skeleton. The
//! enhancement (`m06_f`) is a *smooth* combination — no SCAN-style `α` switch
//! seam — of two pieces:
//! ```text
//! F_x(x, t) = pbe_f(x)·mgga_series_w(a, 12, t)  +  gtv4(α, d, x, 2(t − K))
//! ```
//! 1. **PBE-exchange enhancement × a kinetic series.** `pbe_f(x)` is *exactly*
//!    PBE-x's enhancement (same κ = 0.804, μ = 0.2195149727645171), so M06-L
//!    reuses the shared, sqrt-free [`pbe_enhancement`] rather than forking it
//!    (CLAUDE.md §2; recovery test [`tests::pbe_part_is_shared_pbe_x`]). It is
//!    multiplied by `mgga_series_w(a, 12, t)`, a degree-11 polynomial in the
//!    bounded kinetic variable `w(t) = (K − t)/(K + t)` (`t = τ_σ/n_σ^(5/3)` the
//!    reduced kinetic-energy density; `K = K_FACTOR_C`).
//! 2. **VS98-type local term.** `gtv4(m06_alpha, d, x, 2(t − K))` — the shared
//!    [`gtv4`] working term, a rational in the **squared** reduced gradient and
//!    `2(t − K)`.
//!
//! **AD-safety.** Every reduced gradient enters squared (`w = x_σ²`,
//! [`reduced_grad_sq`](crate::reduced::vars::reduced_grad_sq)); `pbe_enhancement`
//! and `gtv4` are sqrt-free rationals, and `mgga_w(t)` has a strictly positive
//! denominator `K + t` (`t ≥ 0` floored), so `w(t) → 1` smoothly as `t → 0` (the
//! τ-floor edge) with finite derivatives. No `√σ` path is reintroduced. The
//! Laplacian is unused (`needs_lapl = false`).

use num_dual::DualNum;

use super::gga_x_pbe::{pbe_enhancement, KAPPA, MU_X2S2};
use crate::families::mgga::{gtv4, mgga_exchange, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::K_FACTOR_C;

// M06-L exchange coefficients (libxc `par_m06l`, `src/mgga_x_m06l.c`): the 12
// kinetic-series coefficients `a` and the 6 VS98 `gtv4` coefficients `d`.
const A: [f64; 12] = [
    0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615, 30.97273, -23.18489,
    -56.73480, 21.60364, 34.21814, -9.049762,
];
const D: [f64; 6] = [
    0.6012244,
    0.004748822,
    -0.008635108,
    -0.000009308062,
    0.00004482811,
    0.0,
];
/// `gtv4`'s `α` for M06-L exchange (`mgga_x_m06l.mpl` `m06_alpha`).
const M06_ALPHA: f64 = 0.00186726;

/// `mgga_w(t) = (K_FACTOR_C − t)/(K_FACTOR_C + t)` — libxc's bounded kinetic
/// variable (`util.mpl` `mgga_w`), `t` the reduced kinetic-energy density. Maps
/// `t ∈ [0, ∞)` to `w ∈ (−1, 1]`: `w → 1` as `t → 0` (τ-floor edge) and `w → −1`
/// as `t → ∞`. The denominator `K + t > 0` (`t ≥ 0`), so `w` and its derivatives
/// are finite everywhere — no pole, no `√`.
fn mgga_w<N: DualNum<f64> + Copy>(t: N) -> N {
    (N::from(K_FACTOR_C) - t) / (N::from(K_FACTOR_C) + t)
}

/// `mgga_series_w(a, 12, t) = Σ_{i=0}^{11} a[i]·mgga_w(t)^i` (`util.mpl`
/// `mgga_series_w` with `n = 12`), via Horner. A degree-11 polynomial in the
/// bounded `w(t) ∈ (−1, 1]`, hence smooth and bounded.
fn mgga_series_w<N: DualNum<f64> + Copy>(a: &[f64; 12], t: N) -> N {
    let w = mgga_w(t);
    let mut p = N::from(a[11]);
    for &c in a[..11].iter().rev() {
        p = p * w + N::from(c);
    }
    p
}

/// M06-L exchange enhancement `F_x` as a function of the **squared** reduced
/// gradient `w = x_σ²` and reduced kinetic-energy density `t = τ_σ/n_σ^(5/3)`
/// (`mgga_x_m06l.mpl` `m06_f`): `pbe_f(x)·series(t) + gtv4(α, d, x², 2(t − K))`.
/// `pbe_f(x)` is the shared [`pbe_enhancement`] (PBE-x's `pbe_f0`, taking `x²`),
/// and `gtv4` takes `x²` directly. At the uniform-gas point (`x = 0`, `t = K`)
/// `F_x = a[0] + d[0] = 1` exactly (the correct LDA-exchange normalization).
fn m06_x_enhancement<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    let z = N::from(2.0) * (t - N::from(K_FACTOR_C));
    pbe_enhancement(w, KAPPA, MU_X2S2) * mgga_series_w(&A, t) + gtv4(M06_ALPHA, &D, w, z)
}

pub(crate) struct MggaXM06L {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaXM06L {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaXM06L),
                name: "mgga_x_m06_l",
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

impl MggaEnergy for MggaXM06L {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        // meta-GGA exchange = per-channel LDA exchange × M06-L enhancement,
        // screened on the floored spin density (shared `mgga_exchange` skeleton).
        mgga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            m06_x_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduced::consts::X2S;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn m06l(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaXM06L, spin).unwrap()
    }

    /// Reuse recovery (CLAUDE.md §2): the `pbe_f(x)` factor of M06-L exchange is
    /// the *shared* PBE-x enhancement (`pbe_enhancement`), not a fork — so it must
    /// equal PBE-x's `pbe_f0 = 1 + κμs²/(κ + μs²)` with M06-L's κ = 0.804,
    /// μ = 0.2195149727645171 (`s = X2S·x`, `w = x²`). If gga_x_pbe ever changed
    /// its constants, M06-L's reuse assumption would break and this catches it.
    #[test]
    fn pbe_part_is_shared_pbe_x() {
        let kappa = 0.8040_f64;
        let mu = 0.2195149727645171_f64;
        for &w in &[0.0_f64, 1e-8, 0.01, 0.5, 3.0, 100.0, 1e4] {
            let s2 = X2S * X2S * w;
            let want = 1.0 + kappa * mu * s2 / (kappa + mu * s2);
            let got = pbe_enhancement(w, KAPPA, MU_X2S2);
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1.0),
                "pbe_enhancement({w}) = {got} vs PBE-x pbe_f0 {want}"
            );
        }
    }

    /// At the uniform-electron-gas point (`σ = 0` ⇒ `x = 0`, and `τ = τ_unif` ⇒
    /// `t = K_FACTOR_C` ⇒ `w(t) = 0`, `series = a[0]`, `gtv4 = d[0]`), the
    /// enhancement is `a[0] + d[0] = 1`, so the M06-L exchange **energy** recovers
    /// Slater (`lda_x`) — the GGA→LDA normalization. (`τ_unif = K_FACTOR_C·n^(5/3)`
    /// per spin; unpolarized total `τ = 2·K_FACTOR_C·(n/2)^(5/3)`.) Only the energy
    /// matches: M06-L is a meta-GGA, so `vrho` carries extra terms from `∂F_x/∂t`
    /// (`t = τ/n^(5/3)` varies with `n` at fixed τ) and need not equal LDA's.
    #[test]
    fn uniform_gas_recovers_lda_x() {
        assert!((A[0] + D[0] - 1.0).abs() < 1e-12, "a0 + d0 must be 1");
        let m = m06l(Spin::Unpolarized);
        let lda = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        for &n in &[0.1_f64, 1.0, 7.3, 100.0] {
            let tau_unif = 2.0 * K_FACTOR_C * (n / 2.0).powf(5.0 / 3.0);
            let mm = m
                .eval(1, &XcInput::gga(&[n], &[0.0]).with_tau(&[tau_unif]))
                .unwrap();
            let ll = lda.eval(1, &XcInput::lda(&[n])).unwrap();
            assert!(
                (mm.exc[0] - ll.exc[0]).abs() <= 1e-10 * ll.exc[0].abs(),
                "exc n={n}: {} vs {}",
                mm.exc[0],
                ll.exc[0]
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
            (0.1, 0.02, 0.05),
            (10.0, 5.0, 20.0),
            (1.0, 0.4, 0.06), // τ ≈ τ_W (w(t) large)
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
                (out.vrho[k] - fd).abs() <= 1e-6 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        // pure exchange ⇒ vsigma_ab = 0
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
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-12 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-11 * ou.vtau[0].abs().max(1.0));
        assert!((op.vtau[0] - op.vtau[1]).abs() <= 1e-11 * op.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = m06l(Spin::Polarized);
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
            1e6, 1e6, 1e6, // large σ (τ < τ_W ⇒ FHC clamp active)
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
