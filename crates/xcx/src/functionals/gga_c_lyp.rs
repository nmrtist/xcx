// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Lee–Yang–Parr correlation — `gga_c_lyp` (libxc 131).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_c_lyp.mpl` +
//! `maple/util.mpl` (`opz_pow_n`, `one_minus_z_pow_n`).
//!
//! LYP is **not** an enhancement-factor (exchange) form and **not** an
//! `ε_c^unif + H` (PBE-C) form: it is the closed-form Colle–Salvetti correlation
//! expression, with **no uniform-gas limit** to reuse (it never calls PW92). So
//! it does *not* sit on the `gga_exchange` skeleton, and at σ → 0 it recovers its
//! own gradient-free limit `a·(t1 + ω·t3)`, not an LDA correlation.
//!
//! The energy is `f = a·[t1 + ω(rr)·(t2 + t3 + t4 + t5 + t6)]`, with
//! `rr = rs/RS_FACTOR = n^(-1/3)`, `ω(rr) = b·e^(−c·rr)/(1 + d·rr)`,
//! `δ(rr) = (c + d/(1 + d·rr))·rr`. The spin gradients enter asymmetrically as
//! `xs0²·(1+z)^p` (α) vs `xs1²·(1−z)^p` (β), so all three `vsigma` (aa/ab/bb) are
//! distinct and nonzero (σ_ab enters via the total `x_t²`). The spin powers are
//! libxc's **z-based** correlation forms (`opz_pow_n`, `one_minus_z_pow_n`; see
//! docs/api-convention.md §8, divergence A) — *not* the cancellation-free
//! `opz`/`omz` used by exchange.

use std::f64::consts::PI;

use num_dual::DualNum;

use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::RS_FACTOR;
use crate::reduced::vars::{one_minus_z_pow2, opz_pow};

// LYP parameters (gga_c_lyp.c `lyp_values = {0.04918, 0.132, 0.2533, 0.349}`):
// libxc's exact literals (the Colle–Salvetti/Miehlich values).
const A: f64 = 0.04918;
const B: f64 = 0.132;
const C: f64 = 0.2533;
const D: f64 = 0.349;

/// LYP correlation energy per particle `ε_c(rr, z, x_t², x_{s0}², x_{s1}²)`
/// (gga_c_lyp.mpl). `rr = n^(-1/3)`; `xt2`/`xs0_2`/`xs1_2` are the **squared**
/// reduced gradients (LYP uses only the squares, so no `√` is reintroduced).
///
/// Low-density edge: `ω` carries `e^(−c·rr)` and `rr = n^(-1/3) → ∞` as `n → 0`,
/// so `ω` (and its AD derivative) underflow to exactly `0` while every power of
/// `1/n` stays finite (n is floored to `dens_threshold` = 1e-14). `0·finite = 0`,
/// not `0·∞ = NaN` — so the maple's organization is AD-finite as written, no
/// cancellation-free rewrite needed (verified by the finite-derivative edge test).
#[allow(clippy::too_many_arguments)]
fn lyp_energy<N: DualNum<f64> + Copy>(rr: N, z: N, xt2: N, xs0_2: N, xs1_2: N, zt: f64) -> N {
    // Closed-form constants (LYP-specific; computed from the exact closed forms,
    // matching libxc's generated C to f64 precision).
    let cf = 0.3 * (3.0 * PI * PI).powf(2.0 / 3.0); // 3/10·(3π²)^(2/3)
    let aux6 = 2.0_f64.powf(-8.0 / 3.0); // 1/2^(8/3)
    let aux4 = aux6 / 4.0;
    let aux5 = aux4 / 18.0;

    // z-based spin powers (correlation; divergence #1). The (1−z²) prefactor is
    // the *unclamped* product (1−z)(1+z); the (1±z)^p use the zeta-clamped
    // opz_pow_n — distinct, exactly as the maple has them.
    let opz = N::from(1.0) + z; // 1 + z
    let omz = N::from(1.0) - z; // 1 − z  (z-based, not 2·n_b/n)
    let opz83 = opz_pow(opz, 8.0 / 3.0, zt);
    let omz83 = opz_pow(omz, 8.0 / 3.0, zt);
    let opz113 = opz_pow(opz, 11.0 / 3.0, zt);
    let omz113 = opz_pow(omz, 11.0 / 3.0, zt);
    let opz2 = opz_pow(opz, 2.0, zt);
    let omz2 = opz_pow(omz, 2.0, zt);
    let omz2f = one_minus_z_pow2(z); // (1−z)(1+z), unclamped

    // ω(rr) and δ(rr)
    let one_p_drr = N::from(1.0) + N::from(D) * rr; // 1 + d·rr
    let omega = N::from(B) * (-N::from(C) * rr).exp() / one_p_drr;
    let delta = (N::from(C) + N::from(D) / one_p_drr) * rr;

    // t1 = −(1−z²)/(1 + d·rr)
    let t1 = -omz2f / one_p_drr;
    // t2 = −x_t²·[(1−z²)·(47 − 7δ)/72 − 2/3]
    let t2 = -xt2
        * (omz2f * (N::from(47.0) - N::from(7.0) * delta) / N::from(72.0) - N::from(2.0 / 3.0));
    // t3 = −(Cf/2)·(1−z²)·(opz^(8/3) + omz^(8/3))
    let t3 = N::from(-0.5 * cf) * omz2f * (opz83 + omz83);
    // t4 = aux4·(1−z²)·(5/2 − δ/18)·(xs0²·opz^(8/3) + xs1²·omz^(8/3))
    let t4 = N::from(aux4)
        * omz2f
        * (N::from(2.5) - delta / N::from(18.0))
        * (xs0_2 * opz83 + xs1_2 * omz83);
    // t5 = aux5·(1−z²)·(δ − 11)·(xs0²·opz^(11/3) + xs1²·omz^(11/3))
    let t5 = N::from(aux5) * omz2f * (delta - N::from(11.0)) * (xs0_2 * opz113 + xs1_2 * omz113);
    // t6 = −aux6·[ (2/3)(xs0²·opz^(8/3) + xs1²·omz^(8/3))
    //              − (1+z)²·xs1²·omz^(8/3)/4 − (1−z)²·xs0²·opz^(8/3)/4 ]
    let t6 = N::from(-aux6)
        * (N::from(2.0 / 3.0) * (xs0_2 * opz83 + xs1_2 * omz83)
            - opz2 * xs1_2 * omz83 / N::from(4.0)
            - omz2 * xs0_2 * opz83 / N::from(4.0));

    N::from(A) * (t1 + omega * (t2 + t3 + t4 + t5 + t6))
}

pub(crate) struct GgaCLyp {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaCLyp {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaCLyp),
                name: "gga_c_lyp",
                family: Family::Gga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-14, // libxc gga_c_lyp uses 1e-14
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaCLyp {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        // rr = rs/RS_FACTOR = n^(-1/3); LYP uses only the squared reduced
        // gradients — the total `xt2` and the per-spin `xs0_sq`/`xs1_sq`, all
        // carried sqrt-free by the harness. LYP is *linear* in σ, so its true
        // `v2sigma2` is exactly 0; consuming the squared gradients directly (no
        // `√σ`) makes the AD return exactly 0 too, instead of cancellation
        // garbage that blows up as σ → 0 (divergence #4).
        let rr = v.rs / N::from(RS_FACTOR);
        lyp_energy(rr, v.z, v.xt2, v.xs0_sq, v.xs1_sq, self.zeta_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::{A, B, C, D};
    use crate::{Functional, FunctionalId, Spin, XcInput};
    use std::f64::consts::PI;

    fn lyp(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaCLyp, spin).unwrap()
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = lyp(Spin::Unpolarized);
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

    /// LYP's σ enters asymmetrically: σ_aa via x_t² and xs0², σ_bb via x_t² and
    /// xs1², σ_ab only via x_t² — so all three vsigma are distinct and nonzero
    /// (no vsigma_ab ≡ 0, unlike exchange). Verify each independently against FD.
    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = lyp(Spin::Polarized);
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
        // all three vsigma: σ_aa, σ_ab, σ_bb
        for (k, h) in [(0usize, 1e-6 * saa), (1, 1e-6 * sab), (2, 1e-6 * sbb)] {
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
            // and genuinely nonzero (LYP correlation has no vanishing σ channel)
            assert!(out.vsigma[k].abs() > 0.0, "vsigma[{k}] unexpectedly zero");
        }
    }

    /// At σ = 0 LYP recovers its **own** gradient-free limit `a·(t1 + ω·t3)` —
    /// NOT an LDA correlation. Check unpolarized (z = 0) against the closed form
    /// computed independently here:
    ///   ε_c(σ=0, z=0) = a·[ −1/(1+d·rr) − ω·Cf ],  ω = b·e^(−c·rr)/(1+d·rr),
    ///   Cf = (3/10)(3π²)^(2/3),  rr = n^(-1/3).
    #[test]
    fn sigma_zero_recovers_lyp_gradient_free_limit() {
        let f = lyp(Spin::Unpolarized);
        let cf = 0.3 * (3.0 * PI * PI).powf(2.0 / 3.0);
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let got = f.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap().exc[0];
            let rr = n.powf(-1.0 / 3.0);
            let one_p_drr = 1.0 + D * rr;
            let omega = B * (-C * rr).exp() / one_p_drr;
            let want = A * (-1.0 / one_p_drr - omega * cf);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1e-300),
                "n={n}: LYP(σ=0) {got} vs gradient-free limit {want}"
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = lyp(Spin::Unpolarized);
        let po = lyp(Spin::Polarized);
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

    /// Low-density / large-σ edges: derivatives (not just energy) must be finite —
    /// the ε_c(rr)·e^(−c·rr) low-density regime is the AD-NaN risk site, so assert
    /// vrho/vsigma finite, not just exc.
    #[test]
    fn edge_derivatives_finite() {
        let f = lyp(Spin::Polarized);
        let rho = [
            1.0, 0.0, // ζ = +1, full polarization
            0.0, 1.0, // ζ = −1
            1e-13, 1e-14, // very low density (above the 1e-14 threshold)
            1.0, 1.0, //
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0 at full polarization
            0.0, 0.0, 0.0, //
            1e6, 0.0, 1e6, // large σ at low density (stresses ω underflow + AD)
            1e6, 1e6, 1e6, // very large σ
            1.0, 0.5, 0.8, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
