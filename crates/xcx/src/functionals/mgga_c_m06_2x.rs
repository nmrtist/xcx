// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Minnesota M06-2X correlation — `mgga_c_m06_2x` (libxc 236).
//!
//! Provenance: ported-from-libxc (MPL-2.0); the **same functional form** as
//! M06-L correlation (`maple/mgga_exc/mgga_c_m06l.mpl` — M05 + VSXC pieces on
//! the Stoll-decomposed modified-PW92 LSDA), with the M06-2X coefficient table
//! `m062x_values` from `src/mgga_c_m06l.c`. Reference: Y. Zhao & D. G. Truhlar,
//! Theor. Chem. Acc. 120, 215 (2008).
//!
//! Per the reuse rule the energy expression lives once, parameterized, in
//! [`super::mgga_c_m06_l`] ([`MggaCM06`]); this file holds only the M06-2X
//! parameters and metadata. All AD-safety notes there (sqrt-free squared
//! gradients, pole-free `b97_g`/`gtv4`/`Fermi_D`) apply unchanged.

use super::mgga_c_m06_l::{M06CParams, MggaCM06};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};

// M06-2X correlation coefficients (libxc `m062x_values`, `src/mgga_c_m06l.c`).
// γ/α are family-wide constants identical to M06-L's.
const M06_2X_PAR: M06CParams = M06CParams {
    gamma_ss: 0.06,
    gamma_ab: 0.0031,
    alpha_ss: 0.00515088,
    alpha_ab: 0.00304966,
    css: [
        3.097855e-01,
        -5.528642e+00,
        1.347420e+01,
        -3.213623e+01,
        2.846742e+01,
    ],
    cab: [
        8.833596e-01,
        3.357972e+01,
        -7.043548e+01,
        4.978271e+01,
        -1.852891e+01,
    ],
    dss: [
        6.902145e-01,
        9.847204e-02,
        2.214797e-01,
        -1.968264e-03,
        -6.775479e-03,
        0.0,
    ],
    dab: [
        1.166404e-01,
        -9.120847e-02,
        -6.726189e-02,
        6.720580e-05,
        8.448011e-04,
        0.0,
    ],
};

pub(crate) struct MggaCM062x;

impl MggaCM062x {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        MggaCM06::boxed(
            FunctionalInfo {
                id: Some(FunctionalId::MggaCM062x),
                name: "mgga_c_m06_2x",
                family: Family::Mgga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-12, // libxc mgga_c_m06_2x threshold
                hybrid: None,
            },
            &M06_2X_PAR,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn m062x(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaCM062x, spin).unwrap()
    }

    /// Same form as M06-L, different parameters: the two must *differ* at a
    /// generic point (guards against wiring the wrong parameter set).
    #[test]
    fn differs_from_m06_l() {
        let f = m062x(Spin::Unpolarized);
        let l = Functional::new(FunctionalId::MggaCM06L, Spin::Unpolarized).unwrap();
        let inp = XcInput::gga(&[0.8], &[0.3]).with_tau(&[0.6]);
        let a = f.eval(1, &inp).unwrap().exc[0];
        let b = l.eval(1, &inp).unwrap().exc[0];
        assert!(
            (a - b).abs() > 1e-6 * a.abs(),
            "M06-2X ≡ M06-L?! {a} vs {b}"
        );
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = m062x(Spin::Unpolarized);
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
        let f = m062x(Spin::Polarized);
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
        assert_eq!(out.vsigma[1], 0.0, "M06-2X correlation vsigma_ab must be 0");
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
        let up = m062x(Spin::Unpolarized);
        let po = m062x(Spin::Polarized);
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
        let f = m062x(Spin::Polarized);
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
