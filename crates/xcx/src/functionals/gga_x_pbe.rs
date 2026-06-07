// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Perdew–Burke–Ernzerhof exchange — `gga_x_pbe` (libxc 101).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_x_pbe.mpl` +
//! `maple/util.mpl` (`gga_exchange`, `lda_x_spin`, `X2S`).
//!
//! GGA exchange is the LDA exchange of each spin channel times an enhancement
//! factor `F_x(s_σ)`: `f = Σ_σ [screened? 0 : lda_x_spin(rs, 1±z)·F_x(x_σ)]`.
//! The spin factor uses the cancellation-free `opz`/`omz` (matching libxc's
//! exchange) and the per-channel density screen, exactly as `lda_x`.

use std::f64::consts::PI;

use num_dual::DualNum;

use crate::families::gga::{gga_exchange, Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::X2S;

// PBE exchange parameters. `kappa` is the asymptotic enhancement bound; `mu` is
// `MU_PBE = 0.06672455060314922·π²/3`, computed from π exactly as libxc does (its
// `pbe_values` is `{0.8040, MU_PBE}` with `MU_PBE = 0.066…*M_PI*M_PI/3`).
const KAPPA: f64 = 0.8040;
const MU: f64 = 0.066_724_550_603_149_22 * PI * PI / 3.0;

/// PBE exchange enhancement `F_x(s)`, `s = X2S·x`. Mathematically libxc's
/// `pbe_f0 = 1 + κμs²/(κ + μs²)`, but written here in the algebraically
/// identical `(1 + κ) − κ²/(κ + μs²)` form. The reason is **forward-AD**: the
/// maple form's quotient-rule derivative produces `2κμ²s³ − 2κμ²s³`, which
/// cancels catastrophically at large `s` (≈1e-9 error by `s ~ 10³`, where it
/// would miss the golden tol). This form's AD derivative is `κ²/(κ+μs²)²·2μs`,
/// computed with no large-term subtraction — bit-for-bit close to libxc's
/// pre-simplified analytic `dF` (`2κ²μs/(κ+μs²)²`). The two forms agree on the
/// energy to ~1e-16 (the small-`s` `1 − close-to-1` only perturbs the tiny GGA
/// correction, never the O(1) energy), so `F(0) = 1` still holds.
fn pbe_enhancement<N: DualNum<f64> + Copy>(xs: N) -> N {
    let s = N::from(X2S) * xs;
    let denom = N::from(KAPPA) + N::from(MU) * (s * s); // κ + μs²
    N::from(1.0 + KAPPA) - N::from(KAPPA * KAPPA) / denom
}

pub(crate) struct GgaXPbe {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaXPbe {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaXPbe),
                name: "gga_x_pbe",
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

impl GgaEnergy for GgaXPbe {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // GGA exchange = per-channel LDA exchange × PBE enhancement, screened on
        // the floored spin density (shared `gga_exchange` skeleton; the
        // enhancement is this functional's only contribution).
        gga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            pbe_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pbe(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaXPbe, spin).unwrap()
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = pbe(Spin::Unpolarized);
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
        let f = pbe(Spin::Polarized);
        let (na, nb, saa, sab, sbb) = (0.6, 0.3, 0.1, 0.05, 0.08);
        let r = [na, nb];
        let s = [saa, sab, sbb];
        let edens = |r: [f64; 2], s: [f64; 3]| {
            (r[0] + r[1]) * f.eval(1, &XcInput::gga(&r, &s)).unwrap().exc[0]
        };
        let out = f.eval(1, &XcInput::gga(&r, &s)).unwrap();
        // vrho via na, nb
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
        // vsigma[0], vsigma[2] via σ_aa, σ_bb
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

    /// At σ = 0 the enhancement F_x → 1, so PBE exchange must recover Slater
    /// (lda_x) — the GGA→LDA limit — for both energy and potential.
    #[test]
    fn sigma_zero_recovers_lda_x() {
        let pu = pbe(Spin::Unpolarized);
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
        // polarized
        let pp = pbe(Spin::Polarized);
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
        let up = pbe(Spin::Unpolarized);
        let po = pbe(Spin::Polarized);
        let (n, s) = (0.8, 0.3);
        // total σ = σ_aa + 2σ_ab + σ_bb; equal spins with σ_aa=σ_bb=σ/4, σ_ab=σ/4
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
        let f = pbe(Spin::Polarized);
        let rho = [
            1.0, 0.0, // ζ = +1, full polarization
            0.0, 1.0, // ζ = −1
            1e-12, 1e-13, // small densities
            1.0, 1.0, // unpolarized-like, large σ below
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0 at full polarization
            0.0, 0.0, 0.0, //
            1e-20, 0.0, 1e-22, // tiny σ, tiny densities
            1e6, 1e6, 1e6, // very large σ
            1.0, 0.5, 0.8, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
