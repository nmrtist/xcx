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
//! differs: B88's `F_x` depends on the reduced gradient `x` directly (no `X2S`
//! prefactor, unlike PBE's `s = X2S·x`).

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

/// B88 exchange enhancement `F_x(x) = 1 + (β/X_FACTOR_C)·x² / (1 + γβ·x·asinh(x))`,
/// with `x` the per-channel reduced gradient (libxc's `b88_f`; the maple feeds
/// `xs0`/`xs1` straight in, no `X2S`).
///
/// Written as `1 + m1` directly (matching libxc's `b88_f := 1 + b88_f_m1`): we
/// need `F`, not `F − 1`, so the energy never forms a `(1 + tiny) − 1`
/// cancellation. Under forward-AD this is clean as-is — unlike PBE's rational
/// factor there is **no** large-term cancellation in the derivative (at large
/// `x` the numerator of `dF` scales as `x²·(asinh x − 1)`, a difference of
/// *differently*-scaled terms, not equal ones). `asinh`'s AD derivative is
/// libxc's exact `diff/xc_asinh = 1/√(1+x²)`, and `x > 0` always (the harness
/// floors each spin σ), so `√σ` inside `x` and `asinh(x)` stay finite.
fn b88_enhancement<N: DualNum<f64> + Copy>(xs: N) -> N {
    let x2 = xs * xs;
    let denom = N::from(1.0) + N::from(GAMMA * BETA) * xs * xs.asinh(); // 1 + γβ·x·asinh(x)
    N::from(1.0) + N::from(BETA / X_FACTOR_C) * x2 / denom
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
    use crate::{Functional, FunctionalId, Spin, XcInput};

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
