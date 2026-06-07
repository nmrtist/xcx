// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Perdew–Wang 1992 (PW92) correlation — `lda_c_pw` (libxc 12).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/lda_exc/lda_c_pw.mpl` + `util.mpl`.
//!
//! [`pw92_ec`] is the uniform-gas correlation energy per particle, **generic over
//! the parametrization** (`a` coefficients + `f''(0)`). It is the reusable
//! building block for PBE correlation's uniform limit — which passes the
//! *modified* set ([`A_MOD`] + the exact `f''(0)` = `reduced::consts::FPP_VWN`),
//! not `lda_c_pw`'s standard set. Same function, different params; do not fork.

use num_dual::DualNum;

use crate::families::lda::{Lda, LdaEnergy, LdaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::vars::f_zeta;

// PW92 parameters (lda_c_pw.mpl). `ALPHA1`/`BETA1..4` are shared by both the
// standard set (`lda_c_pw`, id 12) and the "modified" set used by PBE-C; only
// the `a` coefficients and `f''(0)` differ between them. Rows: [paramagnetic
// ζ=0, ferromagnetic ζ=1, −spin-stiffness]. (`g` parametrizes −α_c; see eq. 8.)
const A_STD: [f64; 3] = [0.031091, 0.015545, 0.016887];
/// Modified `a` set (`lda_c_pw_modified_params`), used by PBE correlation — more
/// decimal places than [`A_STD`]; paired with the exact `f''(0)` (`FPP_VWN`).
pub(crate) const A_MOD: [f64; 3] = [0.0310907, 0.01554535, 0.0168869];
const ALPHA1: [f64; 3] = [0.21370, 0.20548, 0.11125];
const BETA1: [f64; 3] = [7.5957, 14.1189, 10.357];
const BETA2: [f64; 3] = [3.5876, 6.1977, 3.6231];
const BETA3: [f64; 3] = [1.6382, 3.3662, 0.88026];
const BETA4: [f64; 3] = [0.49294, 0.62517, 0.49671];
/// `f''(0)` for the standard set: the rounded literal libxc's `lda_c_pw` (id 12)
/// uses. The modified set (PBE-C) passes the exact value `FPP_VWN` instead.
const FZ20_STD: f64 = 1.709921;

/// PW92 `G(rs)` for parameter row `k` (libxc eq. 10), given the `a` set:
/// `−2a(1 + α₁ rs)·log1p(1/(2a·(β₁√rs + β₂ rs + β₃ rs^1.5 + β₄ rs²)))`.
/// `log1p` keeps both the large-rs (argument → 0) and small-rs limits accurate.
fn g_pw<N: DualNum<f64> + Copy>(rs: N, k: usize, a: &[f64; 3]) -> N {
    let sqrt_rs = rs.sqrt();
    let aux = N::from(BETA1[k]) * sqrt_rs
        + N::from(BETA2[k]) * rs
        + N::from(BETA3[k]) * (rs * sqrt_rs)
        + N::from(BETA4[k]) * (rs * rs);
    let two_a = N::from(2.0 * a[k]);
    let q = (two_a * aux).recip();
    -two_a * (N::from(1.0) + N::from(ALPHA1[k]) * rs) * q.ln_1p()
}

/// PW92 uniform-gas correlation energy per particle `ε_c(rs, ζ)` (libxc eq. 8):
/// `g0 + ζ⁴·f(ζ)·(g1 − g0 + g2/f''(0)) − f(ζ)·g2/f''(0)`,
/// with `g0 = ε_c(rs,0)`, `g1 = ε_c(rs,1)`, `g2 = −α_c(rs)`. The parametrization
/// (`a`, `fz20 = f''(0)`) is passed in so the **one** implementation serves both
/// `lda_c_pw` (standard) and PBE correlation (modified — see [`A_MOD`]).
pub(crate) fn pw92_ec<N: DualNum<f64> + Copy>(
    rs: N,
    z: N,
    zeta_threshold: f64,
    a: &[f64; 3],
    fz20: f64,
) -> N {
    let g0 = g_pw(rs, 0, a);
    let g1 = g_pw(rs, 1, a);
    let g2 = g_pw(rs, 2, a);
    let fz = f_zeta(z, zeta_threshold);
    let z4 = z.powi(4);
    let fz20 = N::from(fz20);
    g0 + z4 * fz * (g1 - g0 + g2 / fz20) - fz * g2 / fz20
}

pub(crate) struct LdaCPw {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl LdaCPw {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::LdaCPw),
                name: "lda_c_pw",
                family: Family::Lda,
                kind: Kind::Correlation,
                needs_sigma: false,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON,
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Lda(Self::new()))
    }
}

impl LdaEnergy for LdaCPw {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
        pw92_ec(v.rs, v.z, self.zeta_threshold, &A_STD, FZ20_STD)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    #[test]
    fn unpol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaCPw, Spin::Unpolarized).unwrap();
        let edens = |x: f64| x * f.eval(1, &XcInput::lda(&[x])).unwrap().exc[0];
        for &n in &[0.02, 0.2, 2.0, 50.0] {
            let h = 1e-6 * n;
            let fd = (edens(n + h) - edens(n - h)) / (2.0 * h);
            let v = f.eval(1, &XcInput::lda(&[n])).unwrap().vrho[0];
            assert!(
                (v - fd).abs() <= 1e-6 * v.abs().max(1.0),
                "n={n}: {v} vs fd {fd}"
            );
        }
    }

    #[test]
    fn pol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaCPw, Spin::Polarized).unwrap();
        // intermediate polarization, central FD on each channel
        let (na, nb) = (0.6, 0.25);
        let e = |a: f64, b: f64| (a + b) * f.eval(1, &XcInput::lda(&[a, b])).unwrap().exc[0];
        let out = f.eval(1, &XcInput::lda(&[na, nb])).unwrap();
        let ha = 1e-6 * na;
        let hb = 1e-6 * nb;
        let fda = (e(na + ha, nb) - e(na - ha, nb)) / (2.0 * ha);
        let fdb = (e(na, nb + hb) - e(na, nb - hb)) / (2.0 * hb);
        assert!((out.vrho[0] - fda).abs() <= 1e-6 * out.vrho[0].abs().max(1.0));
        assert!((out.vrho[1] - fdb).abs() <= 1e-6 * out.vrho[1].abs().max(1.0));
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = Functional::new(FunctionalId::LdaCPw, Spin::Unpolarized).unwrap();
        let po = Functional::new(FunctionalId::LdaCPw, Spin::Polarized).unwrap();
        let n = 0.9;
        let ou = up.eval(1, &XcInput::lda(&[n])).unwrap();
        let op = po.eval(1, &XcInput::lda(&[n / 2.0, n / 2.0])).unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-13 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-12 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-12 * ou.vrho[0].abs());
    }

    #[test]
    fn edge_energy_and_derivatives_finite() {
        let f = Functional::new(FunctionalId::LdaCPw, Spin::Polarized).unwrap();
        let rho = [
            1.0, 0.0, // ζ = +1 exact
            0.0, 1.0, // ζ = −1 exact
            1e-3, 0.0, // ζ = 1, high rs
            1e-12, 1e-13, // small densities
            100.0, 50.0, // low rs
        ];
        let out = f.eval(5, &XcInput::lda(&rho)).unwrap();
        for v in out.exc.iter().chain(&out.vrho) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
