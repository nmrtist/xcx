// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Perdew–Burke–Ernzerhof correlation — `gga_c_pbe` (libxc 130).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/gga_exc/gga_c_pbe.mpl` +
//! `maple/lda_exc/lda_c_pw.mpl` + `maple/util.mpl`.
//!
//! PBE correlation is the uniform-gas limit plus a gradient correction `H`:
//! `f_pbe = ε_c^unif(rs, ζ) + H(rs, ζ, t)`, `t = tt(rs, x_t, φ(ζ))`.
//! The uniform limit is the **shared** [`pw92_ec`] called with the **modified**
//! PW92 parametrization ([`A_MOD`] + exact `f''(0)` = `FPP_VWN`) — *not*
//! `lda_c_pw`'s standard set. libxc's PBE-C uses the modified parametrization, so
//! at `t → 0` it recovers modified PW92, which differs from `lda_c_pw` by ~1e-5.

use num_dual::DualNum;

use super::lda_c_pw::{pw92_ec, A_MOD};
use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::consts::FPP_VWN;
use crate::reduced::vars::{mphi, tt_sq};

// PBE correlation parameters (gga_c_pbe.c `pbe_values`): libxc's exact literals.
// gamma = (1 − ln2)/π² (the literal libxc stores), beta the PBE constant.
// BB = tscale = 1 are folded in where used.
const BETA: f64 = 0.066_724_550_603_149_22;
const GAMMA: f64 = 0.031_090_690_869_654_895;

/// PBE correlation gradient correction `H(rs, ζ, t²)` (gga_c_pbe.mpl eq. 7), given
/// the uniform-gas `ε_c`, `φ(ζ)`, and the **squared** reduced gradient `t²`:
/// `A = β / (γ·expm1(−ε_c/(γφ³)))`, `f1 = t² + A (t²)²`,
/// `f2 = β f1 / (γ(1 + A f1))`, `H = γ φ³ · log1p(f2)`.
/// `H` depends on `t` only through `t²`/`t⁴`, so it takes `t²` directly (no `√`).
/// `expm1`/`log1p` keep the low-density (`ε_c → 0`) and small-`t` limits
/// cancellation-free, matching libxc's `xc_expm1`/`xc_log1p`.
fn pbe_h<N: DualNum<f64> + Copy>(ec_unif: N, phi: N, t2: N) -> N {
    let phi3 = phi * phi * phi;
    let a = N::from(BETA) / (N::from(GAMMA) * (-ec_unif / (N::from(GAMMA) * phi3)).exp_m1());
    let f1 = t2 + a * t2 * t2; // t² + A·t⁴  (BB = 1)
    let f2 = N::from(BETA) * f1 / (N::from(GAMMA) * (N::from(1.0) + a * f1));
    N::from(GAMMA) * phi3 * f2.ln_1p()
}

pub(crate) struct GgaCPbe {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl GgaCPbe {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::GgaCPbe),
                name: "gga_c_pbe",
                family: Family::Gga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-12, // libxc gga_c_pbe uses 1e-12 (not 1e-15)
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self::new()))
    }
}

impl GgaEnergy for GgaCPbe {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        let zt = self.zeta_threshold;
        // Uniform-gas limit: the SHARED pw92_ec, MODIFIED parametrization.
        let ec_unif = pw92_ec(v.rs, v.z, zt, &A_MOD, FPP_VWN);
        let phi = mphi(v.z, zt);
        let t2 = tt_sq(v.rs, v.xt2, phi);
        ec_unif + pbe_h(ec_unif, phi, t2)
    }
}

#[cfg(test)]
mod tests {
    use super::{A_MOD, FPP_VWN};
    use crate::functionals::lda_c_pw::pw92_ec;
    use crate::reduced::vars::rs_from_n;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pbe(spin: Spin) -> Functional {
        Functional::new(FunctionalId::GgaCPbe, spin).unwrap()
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
        // correlation depends on σ_ab through the total gradient x_t, so all
        // three vsigma components are nonzero (unlike pure exchange).
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
        }
    }

    /// Structural guard for the shared-`pw92_ec` reuse: at σ→0 the gradient
    /// correction `H → 0`, so PBE-C must recover the **modified** PW92 uniform
    /// limit — the exact same `pw92_ec(&A_MOD, FPP_VWN)` it is built on — and must
    /// *differ* from `lda_c_pw` (standard PW92) by ~1e-5. If someone forks a
    /// private PW92 copy or swaps in the standard params, this catches it.
    #[test]
    fn sigma_zero_recovers_modified_pw92() {
        let pu = pbe(Spin::Unpolarized);
        let lda = Functional::new(FunctionalId::LdaCPw, Spin::Unpolarized).unwrap();
        let zt = f64::EPSILON;
        for &n in &[0.1, 1.0, 7.3, 100.0] {
            let got = pu.eval(1, &XcInput::gga(&[n], &[0.0])).unwrap().exc[0];
            let rs = rs_from_n(n);
            let want = pw92_ec(rs, 0.0_f64, zt, &A_MOD, FPP_VWN); // shared, modified
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "n={n}: PBE-C(σ=0) {got} vs shared modified pw92_ec {want}"
            );
            // and it must NOT be the standard lda_c_pw (modified ≠ standard ~1e-5)
            let std = lda.eval(1, &XcInput::lda(&[n])).unwrap().exc[0];
            assert!(
                (got - std).abs() > 1e-7 * std.abs(),
                "n={n}: PBE-C(σ=0) unexpectedly equals standard lda_c_pw"
            );
        }
        // polarized
        let pp = pbe(Spin::Polarized);
        let (na, nb) = (0.6, 0.3);
        let got = pp
            .eval(1, &XcInput::gga(&[na, nb], &[0.0, 0.0, 0.0]))
            .unwrap()
            .exc[0];
        let rs = rs_from_n(na + nb);
        let z = (na - nb) / (na + nb);
        let want = pw92_ec(rs, z, zt, &A_MOD, FPP_VWN);
        assert!((got - want).abs() <= 1e-10 * want.abs());
    }

    #[test]
    fn edge_outputs_finite() {
        let f = pbe(Spin::Polarized);
        let rho = [
            1.0, 0.0, // ζ = +1, full polarization
            0.0, 1.0, // ζ = −1
            1e-10, 1e-11, // small densities (above the 1e-12 threshold)
            1.0, 1.0, //
            100.0, 50.0, // low rs
        ];
        let sigma = [
            0.0, 0.0, 0.0, // σ → 0 at full polarization
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, // tiny σ
            1e6, 1e6, 1e6, // very large σ
            1.0, 0.5, 0.8, //
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
