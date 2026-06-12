// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Perdew 86 correlation — `gga_c_p86` (libxc 132). J. P. Perdew,
//! *Phys. Rev. B* **33**, 8822 (1986).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_c_p86.mpl` +
//! `maple/lda_exc/lda_c_pz.mpl` (PZ81 parameter set) + `maple/util.mpl`
//! (`opz_pow_n`, `f_zeta`, `my_piecewise3`).
//!
//! P86 is the PZ81 LDA correlation plus a gradient term:
//! ```text
//! ε_c = ε_c^PZ81(rs, z) + H,   H = x₁²·e^(−Φ)·C(rs)/D(z)
//! x₁ = x_t/√rr   (rr = rs/RS_FACTOR = n^(−1/3))
//! Φ  = f̃·(C∞/C(rs))·x₁,   C∞ = aa + bb
//! C(rs) = aa + (bb + α·rs + β·rs²)/(1 + γ·rs + δ·rs² + 10⁴·β·rs³)
//! D(z) = √[((1+z)^(5/3) + (1−z)^(5/3))/2]
//! ```
//!
//! ## AD-safety (the odd-`x_t` functional)
//!
//! Unlike PBE-C/LYP, P86's `H` is **odd** in the total reduced gradient (`Φ ∝
//! x₁ = √(x_t²/rr)`), so a `√` of the σ-derived quantity is unavoidable. The
//! harness supplies the squared `x_t²`; here `x₁² = x_t²/rr` is formed sqrt-free
//! and the `√` is taken only when `x₁² > 0`. At **exactly** `x_t² = 0` (reachable
//! only through the σ_ab clamp: `σ_aa = σ_bb`, `σ_ab = −σ_aa`, since σ_aa/σ_bb
//! are floored > 0) the branch evaluates `H = x₁²·C/D` — exact value (0) *and*
//! exact first derivative (`d[q·e^(−k√q)]/dσ → q′·C/D` as `q → 0`), where the
//! naive AD chain would produce `0·∞ = NaN` from the `√` node. The true second
//! σ-derivative diverges `∝ σ_tot^(−1/2)` there (an odd-power property of P86
//! itself, present in libxc too); the branch returns the finite `q`-linear part,
//! keeping the fuzz finiteness contract (divergence-C class, non-physical
//! corner).

use num_dual::DualNum;

use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::RS_FACTOR;
use crate::reduced::vars::{f_zeta, opz_pow};

// P86 parameters (gga_c_p86.c `p86_val`, libxc 6.1.0):
// {malpha, mbeta, mgamma, mdelta, aa, bb, ftilde}.
const MALPHA: f64 = 0.023266;
const MBETA: f64 = 7.389e-6;
const MGAMMA: f64 = 8.723;
const MDELTA: f64 = 0.472;
const AA: f64 = 0.001667;
const BB: f64 = 0.002568;
const FTILDE: f64 = 1.745 * 0.11; // libxc's exact expression (≈ (9π)^(1/6)·0.11)

// PZ81 parameters (lda_c_pz.mpl `lda_c_pz_params`): [unpolarized, polarized].
const PZ_GAMMA: [f64; 2] = [-0.1423, -0.0843];
const PZ_BETA1: [f64; 2] = [1.0529, 1.3981];
const PZ_BETA2: [f64; 2] = [0.3334, 0.2611];
const PZ_A: [f64; 2] = [0.0311, 0.01555];
const PZ_B: [f64; 2] = [-0.048, -0.0269];
const PZ_C: [f64; 2] = [0.0020, 0.0007];
const PZ_D: [f64; 2] = [-0.0116, -0.0048];

/// PZ81 per-row ε_c(rs): the low-density Padé (rs ≥ 1) / high-density
/// logarithmic (rs < 1) piecewise form (`lda_c_pz.mpl` `ec`). The branch
/// predicate is on the real part, so forward-AD follows libxc's
/// `my_piecewise3` exactly (including the derivative discontinuity at rs = 1,
/// which PZ81 has by construction).
fn pz_ec<N: DualNum<f64> + Copy>(i: usize, rs: N) -> N {
    if rs.re() >= 1.0 {
        N::from(PZ_GAMMA[i])
            / (N::from(1.0) + rs.sqrt() * N::from(PZ_BETA1[i]) + rs * N::from(PZ_BETA2[i]))
    } else {
        rs.ln() * N::from(PZ_A[i])
            + N::from(PZ_B[i])
            + rs * rs.ln() * N::from(PZ_C[i])
            + rs * N::from(PZ_D[i])
    }
}

/// PZ81 LDA correlation `f_pz(rs, z) = ec₁ + (ec₂ − ec₁)·f(ζ)` (z-based
/// correlation spin interpolation — divergence-A convention).
pub(crate) fn f_pz<N: DualNum<f64> + Copy>(rs: N, z: N, zeta_threshold: f64) -> N {
    let ec1 = pz_ec(0, rs);
    let ec2 = pz_ec(1, rs);
    ec1 + (ec2 - ec1) * f_zeta(z, zeta_threshold)
}

/// P86 rational `C(rs)` (Eq. 6).
fn p86_cc<N: DualNum<f64> + Copy>(rs: N) -> N {
    N::from(AA)
        + (N::from(BB) + rs * N::from(MALPHA) + rs * rs * N::from(MBETA))
            / (N::from(1.0)
                + rs * N::from(MGAMMA)
                + rs * rs * N::from(MDELTA)
                + rs * rs * rs * N::from(1.0e4 * MBETA))
}

pub(crate) struct GgaCP86 {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaCP86 {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaCP86),
                name: "gga_c_p86",
                family: Family::Gga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15, // libxc gga_c_p86 threshold
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaCP86 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        let zt = self.zeta_threshold;
        // D(z) (Eq. 4): the radicand is ≥ 2^(2/3)/2 > 0 at every ζ (clamped
        // (1±z)^(5/3)), so the √ is AD-safe.
        let opz53 = opz_pow(N::from(1.0) + v.z, 5.0 / 3.0, zt);
        let omz53 = opz_pow(N::from(1.0) - v.z, 5.0 / 3.0, zt);
        let dd = ((opz53 + omz53) / N::from(2.0)).sqrt();

        let cc = p86_cc(v.rs);
        let rr = v.rs / N::from(RS_FACTOR); // n^(-1/3)
        let x1_sq = v.xt2 / rr; // x₁² = x_t²/rr, sqrt-free

        // H = x₁²·e^(−Φ)·C/D, with the exact-zero branch documented above.
        let h = if x1_sq.re() > 0.0 {
            let x1 = x1_sq.sqrt();
            let mphi = x1 * N::from(FTILDE * (AA + BB)) / cc;
            x1_sq * (-mphi).exp() * cc / dd
        } else {
            x1_sq * cc / dd
        };

        f_pz(v.rs, v.z, zt) + h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn p86(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaCP86, spin).unwrap()
    }

    #[test]
    fn unpol_vrho_vsigma_match_finite_difference() {
        let f = p86(Spin::Unpolarized);
        let edens = |n: f64, s: f64| n * f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
        // points on both sides of the PZ81 rs = 1 seam (rs(n) = RS_FACTOR/n^(1/3))
        for &(n, s) in &[
            (0.5, 0.1),
            (2.0, 0.7),
            (0.1, 0.02),
            (10.0, 5.0),
            (0.01, 1e-4),
        ] {
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

    /// P86's σ dependence enters only through the *total* x_t², so all three
    /// vsigma components are nonzero and σ_ab enters with weight 2 relative to
    /// σ_aa/σ_bb (∂x_t²/∂σ_ab = 2·∂x_t²/∂σ_aa).
    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = p86(Spin::Polarized);
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
                (out.vrho[k] - fd).abs() <= 1e-6 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        for (k, h) in [(0usize, 1e-6 * saa), (1, 1e-6 * sab), (2, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp) - edens(r, sm)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-6 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
            assert!(out.vsigma[k].abs() > 0.0, "vsigma[{k}] unexpectedly zero");
        }
        // σ_ab weight 2 vs σ_aa (total-gradient-only functional)
        assert!(
            (out.vsigma[1] - 2.0 * out.vsigma[0]).abs() <= 1e-12 * out.vsigma[1].abs(),
            "vsigma_ab must be 2·vsigma_aa: {} vs {}",
            out.vsigma[1],
            out.vsigma[0]
        );
    }

    /// σ = 0 recovers the PZ81 LDA limit exactly (H → 0), checked against the
    /// closed form on both sides of the rs = 1 seam.
    #[test]
    fn sigma_zero_recovers_pz81() {
        let f = p86(Spin::Unpolarized);
        for &n in &[0.01, 0.1, 1.0, 7.3, 100.0] {
            let got = f.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap().exc[0];
            let rs = crate::reduced::consts::RS_FACTOR / n.powf(1.0 / 3.0);
            let want = if rs >= 1.0 {
                PZ_GAMMA[0] / (1.0 + PZ_BETA1[0] * rs.sqrt() + PZ_BETA2[0] * rs)
            } else {
                PZ_A[0] * rs.ln() + PZ_B[0] + PZ_C[0] * rs * rs.ln() + PZ_D[0] * rs
            };
            // σ is floored to sigma_threshold² so H is ~1e-80, not exactly 0
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "n={n}: P86(σ=0) {got} vs PZ81 {want}"
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = p86(Spin::Unpolarized);
        let po = p86(Spin::Polarized);
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

    /// Edge finiteness, including the σ_ab clamp corner that drives the total
    /// gradient to exactly 0 (the x₁ = √(x_t²) AD hazard this file branches on).
    #[test]
    fn edge_derivatives_finite() {
        let f = p86(Spin::Polarized);
        let rho = [
            1.0, 0.0, // ζ = +1
            0.0, 1.0, // ζ = −1
            1e-13, 1e-14, // very low density
            0.5, 0.5, // σ_ab clamp corner: σ_tot = 0 exactly
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e6, 0.0, 1e6, //
            0.1, -10.0, 0.1, // clamps to σ_ab = −0.1 ⇒ σ_tot = 0
            1.0, 0.5, 0.8, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
