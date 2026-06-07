// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Vosko–Wilk–Nusair correlation, parametrization V ("VWN5") — `lda_c_vwn`
//! (libxc 7).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/lda_exc/lda_c_vwn.mpl` +
//! `maple/vwn.mpl` + `util.mpl`.
//!
//! VWN interpolates between the paramagnetic (ζ=0) and ferromagnetic (ζ=1)
//! correlation energies via the spin-stiffness α_c(rs). [`vwn_f_aux`] is the
//! shared per-parameter-row building block (the `f_aux` of `vwn.mpl`); VWN3
//! (`lda_c_vwn_3`) reuses it — and the `*_VWN` parameter rows — with a different
//! mixing, so they are kept `pub(crate)` here.

use std::f64::consts::PI;

use num_dual::DualNum;

use crate::families::lda::{Lda, LdaEnergy, LdaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::FPP_VWN;
use crate::reduced::vars::{f_zeta, one_minus_z_pow4};

// VWN parameters (vwn.mpl), already converted from Rydberg to Hartree (reference
// values divided by two). Rows, 0-based: [paramagnetic ζ=0, ferromagnetic ζ=1,
// spin stiffness]. The stiffness amplitude is A = −1/(6π²).
pub(crate) const A_VWN: [f64; 3] = [0.0310907, 0.01554535, -1.0 / (6.0 * PI * PI)];
pub(crate) const B_VWN: [f64; 3] = [3.72744, 7.06042, 1.13107];
pub(crate) const C_VWN: [f64; 3] = [12.9352, 18.0578, 13.0045];
pub(crate) const X0_VWN: [f64; 3] = [-0.10498, -0.32500, -0.0047584];

/// VWN `f_aux(A, b, c, x0; rs)` (vwn.mpl): one parameter row's correlation
/// energy contribution. With `y = √rs`, `Q = √(4c − b²)`, `X(y) = y² + by + c`:
///
/// `A·[ log((rs)/X) + (f1 − f2·f3)·arctan(Q/(2y+b)) − f2·log((y−x0)²/X) ]`,
///
/// `f1 = 2b/Q`, `f2 = b·x0/X(x0)`, `f3 = 2(2x0+b)/Q`. Both logs are written via
/// `log1p` in libxc's form: the first numerator `rs − X = −(b√rs + c)` is taken
/// cancellation-free, the second `(y−x0)² − X` exactly as the generated C does.
pub(crate) fn vwn_f_aux<N: DualNum<f64> + Copy>(rs: N, a: f64, b: f64, c: f64, x0: f64) -> N {
    let y = rs.sqrt(); // √rs
    let q = (4.0 * c - b * b).sqrt(); // Q = √(4c − b²)
    let f1 = 2.0 * b / q; // 2b/Q
    let f2 = b * x0 / (x0 * x0 + b * x0 + c); // b·x0/X(x0)
    let f3 = 2.0 * (2.0 * x0 + b) / q; // 2(2x0+b)/Q
    let xx = rs + N::from(b) * y + N::from(c); // X(√rs) = rs + b√rs + c
    let inv = xx.recip();
    // log1p((rs − X)/X): rs − X = −(b√rs + c), cancellation-free.
    let l1 = (-(N::from(b) * y + N::from(c)) * inv).ln_1p();
    // log1p(((√rs − x0)² − X)/X): libxc's exact (mildly cancelling) numerator.
    let ymx0 = y - N::from(x0);
    let l2 = ((ymx0 * ymx0 - xx) * inv).ln_1p();
    let at = (N::from(q) / (y + y + N::from(b))).atan(); // arctan(Q/(2√rs + b))
    N::from(a) * (l1 + N::from(f1 - f2 * f3) * at - N::from(f2) * l2)
}

/// VWN5 correlation energy per particle `ε_c(rs, ζ)` (lda_c_vwn.mpl):
/// `ε_c(rs,0) + α_c(rs)·f(ζ)·(1−ζ⁴)/f''(0) + [ε_c(rs,1) − ε_c(rs,0)]·f(ζ)·ζ⁴`,
/// with the spin stiffness `α_c` from the third row and the exact `f''(0)`.
fn vwn5_ec<N: DualNum<f64> + Copy>(rs: N, z: N, zeta_threshold: f64) -> N {
    let eps0 = vwn_f_aux(rs, A_VWN[0], B_VWN[0], C_VWN[0], X0_VWN[0]); // ε_c(rs,0)
    let eps1 = vwn_f_aux(rs, A_VWN[1], B_VWN[1], C_VWN[1], X0_VWN[1]); // ε_c(rs,1)
    let alpha = vwn_f_aux(rs, A_VWN[2], B_VWN[2], C_VWN[2], X0_VWN[2]); // α_c(rs)
    let fz = f_zeta(z, zeta_threshold);
    let z4 = z.powi(4);
    eps0 + alpha * fz * one_minus_z_pow4(z) / N::from(FPP_VWN) + (eps1 - eps0) * fz * z4
}

pub(crate) struct LdaCVwn {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl LdaCVwn {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::LdaCVwn),
                name: "lda_c_vwn",
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

impl LdaEnergy for LdaCVwn {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
        vwn5_ec(v.rs, v.z, self.zeta_threshold)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    #[test]
    fn unpol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaCVwn, Spin::Unpolarized).unwrap();
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
        let f = Functional::new(FunctionalId::LdaCVwn, Spin::Polarized).unwrap();
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
        let up = Functional::new(FunctionalId::LdaCVwn, Spin::Unpolarized).unwrap();
        let po = Functional::new(FunctionalId::LdaCVwn, Spin::Polarized).unwrap();
        let n = 0.9;
        let ou = up.eval(1, &XcInput::lda(&[n])).unwrap();
        let op = po.eval(1, &XcInput::lda(&[n / 2.0, n / 2.0])).unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-13 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-12 * ou.vrho[0].abs());
        assert!((ou.vrho[0] - op.vrho[1]).abs() <= 1e-12 * ou.vrho[0].abs());
    }

    #[test]
    fn edge_energy_and_derivatives_finite() {
        let f = Functional::new(FunctionalId::LdaCVwn, Spin::Polarized).unwrap();
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
