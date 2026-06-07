// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Vosko–Wilk–Nusair correlation, parametrization III ("VWN3", RPA spin
//! interpolation) — `lda_c_vwn_3` (libxc 30).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/lda_exc/lda_c_vwn_3.mpl` +
//! `maple/vwn.mpl` + `util.mpl`.
//!
//! VWN3 differs from VWN5 (`lda_c_vwn`) only in the spin-stiffness term: instead
//! of the VWN stiffness row it uses the **RPA** stiffness `α_c^RPA(rs)` rescaled
//! by `ΔMC/ΔRPA` (the ratio of the ferro−para correlation differences of the
//! "Monte-Carlo"/VWN and RPA parameter sets). The paramagnetic limit and the
//! `ζ⁴` term are identical to VWN5, so the two agree exactly when unpolarized.
//! The shared [`vwn_f_aux`] and the VWN parameter rows are reused from the VWN5
//! module; only the RPA rows are added here.

use std::f64::consts::PI;

use num_dual::DualNum;

use super::lda_c_vwn::{vwn_f_aux, A_VWN, B_VWN, C_VWN, X0_VWN};
use crate::families::lda::{Lda, LdaEnergy, LdaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::FPP_VWN;
use crate::reduced::vars::{f_zeta, one_minus_z_pow4};

// RPA parameters (vwn.mpl), already converted from Rydberg to Hartree. Rows,
// 0-based: [paramagnetic ζ=0, ferromagnetic ζ=1, spin stiffness]. The A column
// equals the VWN one (stiffness amplitude −1/(6π²)); it is spelled out here to
// match the maple `A_rpa` definition rather than aliasing `A_VWN`.
// `pub(crate)` because `lda_c_vwn_rpa` (libxc 8) reuses the para/ferro rows — one
// shared source, no fork (it uses only rows 0/1; the stiffness row is VWN3-only).
pub(crate) const A_RPA: [f64; 3] = [0.0310907, 0.01554535, -1.0 / (6.0 * PI * PI)];
pub(crate) const B_RPA: [f64; 3] = [13.0720, 20.1231, 1.06835];
pub(crate) const C_RPA: [f64; 3] = [42.7198, 101.578, 11.4813];
pub(crate) const X0_RPA: [f64; 3] = [-0.409286, -0.743294, -0.228344];

/// VWN3 correlation energy per particle `ε_c(rs, ζ)` (lda_c_vwn_3.mpl):
/// `ε_c(rs,0) + (ΔMC/ΔRPA)·α_c^RPA(rs)·f(ζ)·(1−ζ⁴)/f''(0) + ΔMC·f(ζ)·ζ⁴`,
/// where `ΔMC = ε_c^VWN(rs,1) − ε_c^VWN(rs,0)` and `ΔRPA` is its RPA analogue.
fn vwn3_ec<N: DualNum<f64> + Copy>(rs: N, z: N, zeta_threshold: f64) -> N {
    let eps0 = vwn_f_aux(rs, A_VWN[0], B_VWN[0], C_VWN[0], X0_VWN[0]); // ε_c^VWN(rs,0)
    let dmc = vwn_f_aux(rs, A_VWN[1], B_VWN[1], C_VWN[1], X0_VWN[1]) - eps0; // ΔMC
    let drpa = vwn_f_aux(rs, A_RPA[1], B_RPA[1], C_RPA[1], X0_RPA[1])
        - vwn_f_aux(rs, A_RPA[0], B_RPA[0], C_RPA[0], X0_RPA[0]); // ΔRPA
    let alpha = vwn_f_aux(rs, A_RPA[2], B_RPA[2], C_RPA[2], X0_RPA[2]); // α_c^RPA(rs)
    let fz = f_zeta(z, zeta_threshold);
    let z4 = z.powi(4);
    eps0 + (dmc / drpa) * alpha * fz * one_minus_z_pow4(z) / N::from(FPP_VWN) + dmc * fz * z4
}

pub(crate) struct LdaCVwn3 {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl LdaCVwn3 {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::LdaCVwn3),
                name: "lda_c_vwn_3",
                family: Family::Lda,
                kind: Kind::Correlation,
                needs_sigma: false,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Lda(Self::new()))
    }
}

impl LdaEnergy for LdaCVwn3 {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
        vwn3_ec(v.rs, v.z, self.zeta_threshold)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    #[test]
    fn unpol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaCVwn3, Spin::Unpolarized).unwrap();
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
        let f = Functional::new(FunctionalId::LdaCVwn3, Spin::Polarized).unwrap();
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

    /// At ζ = 0 VWN3 and VWN5 collapse to the same paramagnetic limit ε_c(rs,0),
    /// so the unpolarized and equal-spin polarized evaluations must agree.
    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = Functional::new(FunctionalId::LdaCVwn3, Spin::Unpolarized).unwrap();
        let po = Functional::new(FunctionalId::LdaCVwn3, Spin::Polarized).unwrap();
        let n = 0.9;
        let ou = up.eval(1, &XcInput::lda(&[n])).unwrap();
        let op = po.eval(1, &XcInput::lda(&[n / 2.0, n / 2.0])).unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-13 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-12 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-12 * ou.vrho[0].abs());
    }

    /// VWN3 must differ from VWN5 at intermediate polarization (different
    /// spin-stiffness interpolation) but coincide when unpolarized.
    #[test]
    fn differs_from_vwn5_only_when_polarized() {
        let v3 = Functional::new(FunctionalId::LdaCVwn3, Spin::Polarized).unwrap();
        let v5 = Functional::new(FunctionalId::LdaCVwn, Spin::Polarized).unwrap();
        let pol = v3.eval(1, &XcInput::lda(&[0.6, 0.25])).unwrap();
        let pol5 = v5.eval(1, &XcInput::lda(&[0.6, 0.25])).unwrap();
        assert!(
            (pol.exc[0] - pol5.exc[0]).abs() > 1e-9,
            "VWN3 should differ from VWN5 when polarized"
        );

        let unp3 = v3.eval(1, &XcInput::lda(&[0.3, 0.3])).unwrap();
        let unp5 = v5.eval(1, &XcInput::lda(&[0.3, 0.3])).unwrap();
        assert!((unp3.exc[0] - unp5.exc[0]).abs() <= 1e-13 * unp3.exc[0].abs());
    }

    #[test]
    fn edge_energy_and_derivatives_finite() {
        let f = Functional::new(FunctionalId::LdaCVwn3, Spin::Polarized).unwrap();
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
