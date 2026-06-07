// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Vosko–Wilk–Nusair correlation, RPA parametrization ("VWN5_RPA") —
//! `lda_c_vwn_rpa` (libxc 8).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/lda_exc/lda_c_vwn_rpa.mpl` +
//! `maple/vwn.mpl` + `util.mpl`.
//!
//! This is the parametrization libxc's **B3LYP** (`hyb_gga_xc_b3lyp`, 402) mixes
//! in — distinct from both VWN5 (`lda_c_vwn`, 7) and VWN3 (`lda_c_vwn_3`, 30).
//! Its form is the simplest of the three: a plain interpolation between the RPA
//! paramagnetic and ferromagnetic energies via `f(ζ)`, with **no** spin-stiffness
//! or `ζ⁴` term (lda_c_vwn_rpa.mpl):
//!
//! `ε_c(rs, ζ) = ε_c^RPA(rs,0)·(1 − f(ζ)) + ε_c^RPA(rs,1)·f(ζ)`.
//!
//! It reuses the shared [`vwn_f_aux`] and the RPA parameter rows already defined
//! for VWN3 ([`A_RPA`] etc.) — one source, no fork — and the cancellation-free
//! [`one_minus_f_zeta`] (z-based correlation convention; see
//! docs/api-convention.md §8, divergence A).

use num_dual::DualNum;

use super::lda_c_vwn::vwn_f_aux;
use super::lda_c_vwn_3::{A_RPA, B_RPA, C_RPA, X0_RPA};
use crate::families::lda::{Lda, LdaEnergy, LdaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::vars::f_zeta;

/// VWN_RPA correlation energy per particle `ε_c(rs, ζ)`. libxc's lda_c_vwn_rpa.mpl
/// writes it `f_aux(RPA_para)·(1 − f(ζ)) + f_aux(RPA_ferro)·f(ζ)` (with the
/// cancellation-free `one_minus_f_zeta` for the para weight). We use the
/// algebraically identical `para + (ferro − para)·f(ζ)`, which depends on the
/// **single shared z-based `f_zeta`** rather than a separate `one_minus_f_zeta`.
///
/// Why this form (it matters for the *derivative* at full polarization): libxc's
/// `one_minus_f_zeta` keeps the **energy** precise at `ζ → ±1`, but its
/// maple-`diff` and our forward-AD of it lose precision *differently* when the
/// minority spin is floored (`z` within ~1 ulp of 1), giving a ~1e-7 split in the
/// minority `vrho`. Routing the para weight through the same `f_zeta` the other
/// VWN functionals use makes the AD derivative reproduce libxc's `f_zeta`-based
/// cancellation exactly (vwn5/vwn3 are golden-green at those points). The energy
/// cost of `(1 − f_zeta)` vs `one_minus_f_zeta` is ~5e-18 (para · the `1 −
/// close-to-1` residue), far under the 1e-10 golden tolerance.
fn vwn_rpa_ec<N: DualNum<f64> + Copy>(rs: N, z: N, zeta_threshold: f64) -> N {
    let para = vwn_f_aux(rs, A_RPA[0], B_RPA[0], C_RPA[0], X0_RPA[0]); // ε_c^RPA(rs,0)
    let ferro = vwn_f_aux(rs, A_RPA[1], B_RPA[1], C_RPA[1], X0_RPA[1]); // ε_c^RPA(rs,1)
    para + (ferro - para) * f_zeta(z, zeta_threshold)
}

pub(crate) struct LdaCVwnRpa {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl LdaCVwnRpa {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::LdaCVwnRpa),
                name: "lda_c_vwn_rpa",
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

impl LdaEnergy for LdaCVwnRpa {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: LdaVars<N>) -> N {
        vwn_rpa_ec(v.rs, v.z, self.zeta_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::vwn_rpa_ec;
    use super::{A_RPA, B_RPA, C_RPA, X0_RPA};
    use crate::functionals::lda_c_vwn::vwn_f_aux;
    use crate::reduced::vars::{f_zeta, rs_from_n};
    use crate::{Functional, FunctionalId, Spin, XcInput};

    #[test]
    fn unpol_vrho_matches_finite_difference() {
        let f = Functional::new(FunctionalId::LdaCVwnRpa, Spin::Unpolarized).unwrap();
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
        let f = Functional::new(FunctionalId::LdaCVwnRpa, Spin::Polarized).unwrap();
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

    /// Reuse guard: the functional must equal the shared `vwn_f_aux` + RPA rows
    /// interpolation it is built on (so a future fork/param swap is caught), and
    /// must DIFFER from VWN5 and VWN3 when polarized (it is a distinct
    /// parametrization). At ζ = 0 all three share ε_c(rs,0)·1 only if their para
    /// rows match — they do NOT (RPA para row ≠ VWN para row), so VWN_RPA differs
    /// from VWN5/VWN3 even unpolarized.
    #[test]
    fn matches_shared_form_and_differs_from_vwn5_vwn3() {
        let zt = f64::EPSILON;
        let (na, nb) = (0.6, 0.25);
        let rs = rs_from_n(na + nb);
        let z = (na - nb) / (na + nb);
        // shared-form reference
        let para = vwn_f_aux(rs, A_RPA[0], B_RPA[0], C_RPA[0], X0_RPA[0]);
        let ferro = vwn_f_aux(rs, A_RPA[1], B_RPA[1], C_RPA[1], X0_RPA[1]);
        let want = para + (ferro - para) * f_zeta(z, zt);
        assert!((vwn_rpa_ec(rs, z, zt) - want).abs() <= 1e-15 * want.abs());

        let rpa = Functional::new(FunctionalId::LdaCVwnRpa, Spin::Polarized).unwrap();
        let got = rpa.eval(1, &XcInput::lda(&[na, nb])).unwrap().exc[0];
        for other in [FunctionalId::LdaCVwn, FunctionalId::LdaCVwn3] {
            let f = Functional::new(other, Spin::Polarized).unwrap();
            let e = f.eval(1, &XcInput::lda(&[na, nb])).unwrap().exc[0];
            assert!(
                (got - e).abs() > 1e-6 * e.abs(),
                "VWN_RPA should differ from {other:?}: {got} vs {e}"
            );
        }
    }

    #[test]
    fn edge_energy_and_derivatives_finite() {
        let f = Functional::new(FunctionalId::LdaCVwnRpa, Spin::Polarized).unwrap();
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
