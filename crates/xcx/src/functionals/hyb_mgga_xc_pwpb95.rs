// Copyright (c) 2026 Jiekang Tian and the xcx authors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PWPB95 — double-hybrid meta-GGA (xcx-private id 100003; **not in libxc**).
//! L. Goerigk & S. Grimme, *J. Chem. Theory Comput.* **7**, 291 (2011).
//!
//! Provenance: clean-room (from the publication); the component forms reuse
//! xcx's mPW91-exchange / B95-correlation sources
//! ([`super::hyb_mgga_xc_pw6b95`]), which are golden-locked against libxc for
//! PW6B95's parameter set and libxc-verified for *this* parameter set through
//! `xc_func_set_ext_params` in the snapshot generator.
//!
//! `E_xc = (1−a_x)·E_x^mPW(reopt.) + a_x·E_x^HF + (1−c)·E_c^B95(reopt.) +
//! c·E_c^{OS-PT2}` with `a_x = 0.50`, `c = c_os = 0.269`, `c_ss = 0`
//! (spin-opposite-scaled PT2 only). The PW91-form exchange parameters and the
//! two B95 parameters were reoptimized (paper §2.3 / SI; cross-checked
//! against Psi4's `dh_functionals.py` "paper values" tweak of `GGA_X_MPW91`):
//! `bt = 0.004440`, `c = 0.32620` (the exponential coefficient — `α_w` in
//! xcx's w-space, `c/X2S²` in libxc's s-space), `expo = 3.7868`;
//! `c_ss(B95) = 0.03241`, `c_opp(B95) = 0.00250`. xcx emits only the
//! semilocal mix; the host adds 50% EXX and the SOS-PT2 term from metadata
//! (`exx_fraction`, `double_hybrid()`).

use super::hyb_mgga_xc_pw6b95::{bc95_c_component, mpw91_x_component};
use crate::error::XcError;
use crate::families::XcEval;
use crate::func::{mixed_eval, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind};

/// Exact-exchange fraction `a_x` (Goerigk & Grimme 2011).
const EXX_FRACTION: f64 = 0.50;
/// PT2 mixing `c` — the opposite-spin coefficient (SOS-PT2; `c_ss = 0`).
const C_PT2: f64 = 0.269;

// Reoptimized PW91-form exchange parameters (paper "paper values": bt, c —
// the w-space exponential coefficient — and the damping exponent; same
// 3-parameter family as PW6B95's {0.00538, 1.7382, 3.8901}).
const BT: f64 = 0.004440;
const ALPHA_W: f64 = 0.32620;
const EXPO: f64 = 3.7868;

// Reoptimized B95 correlation parameters (paper values).
const C_SS: f64 = 0.03241;
const C_OPP: f64 = 0.00250;

/// Build PWPB95's semilocal part as the mix
/// `0.50·mPW-x(reopt.) + (1 − 0.269)·B95-c(reopt.)`.
pub(crate) fn pwpb95() -> Result<Box<dyn XcEval>, XcError> {
    let info = FunctionalInfo {
        id: Some(FunctionalId::HybMggaXcPwpb95),
        name: "hyb_mgga_xc_pwpb95",
        family: Family::HybMgga,
        kind: Kind::ExchangeCorrelation,
        needs_sigma: true,
        needs_lapl: false,
        needs_tau: true,
        dens_threshold: 1e-14, // component minimum (B95: 1e-14, mPW91: 1e-15)
        hybrid: Some(HybridInfo {
            exx_fraction: EXX_FRACTION,
            cam: None,
            vv10: None,
        }),
    };
    Ok(mixed_eval(
        vec![
            (
                1.0 - EXX_FRACTION,
                mpw91_x_component("gga_x_mpw91 (PWPB95 parameters)", BT, ALPHA_W, EXPO),
            ),
            (
                1.0 - C_PT2,
                bc95_c_component("mgga_c_bc95 (PWPB95 parameters)", C_SS, C_OPP),
            ),
        ],
        info,
    ))
}

#[cfg(test)]
mod tests {
    use crate::func::Rung;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn pwpb95(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybMggaXcPwpb95, spin).unwrap()
    }

    /// Metadata: DoubleHybrid rung, 50% EXX, SOS-PT2 (c_os = 0.269, c_ss = 0).
    #[test]
    fn metadata_double_hybrid() {
        let f = pwpb95(Spin::Unpolarized);
        assert_eq!(f.exx_fraction(), 0.50);
        assert_eq!(f.info().name, "hyb_mgga_xc_pwpb95");
        assert_eq!(f.info().rung(), Rung::DoubleHybrid);
        let p = f.info().double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.269, 0.0));
        assert!(f.info().needs_tau && f.info().needs_sigma);
    }

    /// PWPB95 must differ from PW6B95 (the parameters are reoptimized) while
    /// sharing the component forms.
    #[test]
    fn differs_from_pw6b95() {
        let a = pwpb95(Spin::Unpolarized);
        let b = Functional::new(FunctionalId::HybMggaXcPw6b95, Spin::Unpolarized).unwrap();
        let inp = XcInput::gga(&[0.7], &[0.3]).with_tau(&[0.5]);
        let ea = a.eval(1, &inp).unwrap().exc[0];
        let eb = b.eval(1, &inp).unwrap().exc[0];
        assert!((ea - eb).abs() > 1e-6 * ea.abs(), "{ea} vs {eb}");
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = pwpb95(Spin::Unpolarized);
        let edens = |n: f64, s: f64, tau: f64| {
            n * f
                .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                .unwrap()
                .exc[0]
        };
        for &(n, s, tau) in &[
            (0.5, 0.1, 0.3),
            (2.0, 0.7, 1.5),
            (0.3, 0.02, 0.2),
            (5.0, 3.0, 8.0),
            (1.0, 0.4, 0.06), // τ ≈ τ_W
        ] {
            let out = f
                .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                .unwrap();
            let (hn, hs, ht) = (1e-6 * n, 1e-6 * s, 1e-6 * tau);
            let fdn = (edens(n + hn, s, tau) - edens(n - hn, s, tau)) / (2.0 * hn);
            let fds = (edens(n, s + hs, tau) - edens(n, s - hs, tau)) / (2.0 * hs);
            let fdt = (edens(n, s, tau + ht) - edens(n, s, tau - ht)) / (2.0 * ht);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-5 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s} t={tau}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-5 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s} t={tau}: {} vs {fds}",
                out.vsigma[0]
            );
            assert!(
                (out.vtau[0] - fdt).abs() <= 1e-5 * out.vtau[0].abs().max(1.0),
                "vtau n={n} s={s} t={tau}: {} vs {fdt}",
                out.vtau[0]
            );
        }
    }

    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = pwpb95(Spin::Polarized);
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
                (out.vrho[k] - fd).abs() <= 1e-5 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        assert_eq!(out.vsigma[1], 0.0, "PWPB95 vsigma_ab must be 0");
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp, t) - edens(r, sm, t)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-5 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
        for (k, h) in [(0usize, 1e-6 * ta), (1, 1e-6 * tb)] {
            let (mut tp, mut tm) = (t, t);
            tp[k] += h;
            tm[k] -= h;
            let fd = (edens(r, s, tp) - edens(r, s, tm)) / (2.0 * h);
            assert!(
                (out.vtau[k] - fd).abs() <= 1e-5 * out.vtau[k].abs().max(1.0),
                "vtau[{k}]: {} vs {fd}",
                out.vtau[k]
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = pwpb95(Spin::Unpolarized);
        let po = pwpb95(Spin::Polarized);
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
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-11 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-10 * ou.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = pwpb95(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-10, 1e-11, 1.0, 1.0, 100.0, 50.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1.0, 0.5, 0.8, //
        ];
        let tau = [0.5, 0.0, 0.0, 0.5, 1e-12, 1e-13, 0.5, 0.5, 50.0, 30.0];
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
