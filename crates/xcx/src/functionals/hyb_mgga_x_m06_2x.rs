// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Minnesota M06-2X hybrid exchange — `hyb_mgga_x_m06_2x` (libxc 450), 54% EXX.
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/hyb_mgga_x_m05.mpl`
//! plus `maple/gga_exc/gga_x_pbe.mpl` (`pbe_f`) and `maple/util.mpl`
//! (`mgga_exchange`, `mgga_series_w`). Coefficient table (`a[12]`, `csi_HF = 1`)
//! from `src/hyb_mgga_x_m05.c` `par_m06_2x`; reference: Y. Zhao & D. G. Truhlar,
//! Theor. Chem. Acc. 120, 215 (2008).
//!
//! M06-2X exchange uses the **M05 functional form** — unlike M06-L it has **no
//! VS98 (`gtv4`) part** (libxc implements it in `hyb_mgga_x_m05.c`, not the
//! M06-L file):
//! ```text
//! F_x(x, t) = csi_HF · pbe_f(x) · mgga_series_w(a, 12, t),   csi_HF = 1
//! ```
//! on the shared [`mgga_exchange`] skeleton — per-spin LDA exchange × `F_x` of
//! the **squared** reduced gradient `w = x_σ²` and reduced kinetic-energy
//! density `t_σ`. `pbe_f` is the shared, sqrt-free [`pbe_enhancement`] (same
//! κ = 0.804, μ = 0.2195149727645171 as PBE-x), and the kinetic series is the
//! shared [`mgga_series_w`] in the bounded `w(t) = (K − t)/(K + t)` — both
//! reused from PBE-x / M06-L rather than forked.
//!
//! **Hybrid:** xcx emits only this semilocal part; the host adds 54% exact
//! exchange ([`HybridInfo::exx_fraction`] = 0.54, libxc `cam_alpha`). Per the
//! libxc comment the 2X mixing is already folded into the `a` coefficients, so
//! `csi_HF = 1` (no extra scaling).
//!
//! **AD-safety.** Identical to M06-L exchange: the reduced gradient enters
//! squared, `pbe_enhancement` is a sqrt-free rational, and `mgga_w`'s
//! denominator `K + t > 0` is pole-free; no `√σ` path. The Laplacian is unused.

use num_dual::DualNum;

use super::gga_x_pbe::{pbe_enhancement, KAPPA, MU_X2S2};
use super::mgga_x_m06_l::mgga_series_w;
use crate::families::mgga::{mgga_exchange, Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, HybridInfo, Kind};

// M06-2X exchange kinetic-series coefficients (libxc `par_m06_2x`,
// `src/hyb_mgga_x_m05.c`); `csi_HF = 1.0` (mixing already in `a`), EXX = 0.54.
const A: [f64; 12] = [
    4.600000e-01,
    -2.206052e-01,
    -9.431788e-02,
    2.164494e+00,
    -2.556466e+00,
    -1.422133e+01,
    1.555044e+01,
    3.598078e+01,
    -2.722754e+01,
    -3.924093e+01,
    1.522808e+01,
    1.522227e+01,
];

/// Fraction of exact exchange the host must add (libxc `par_m06_2x` `_cx`).
const EXX_FRACTION: f64 = 0.54;

/// M06-2X exchange enhancement `F_x(w, t) = pbe_f(x)·mgga_series_w(a, 12, t)`
/// with `w = x_σ²` (the M05 form, `csi_HF = 1`; no VS98 `gtv4` term).
fn m06_2x_enhancement<N: DualNum<f64> + Copy>(w: N, t: N) -> N {
    pbe_enhancement(w, KAPPA, MU_X2S2) * mgga_series_w(&A, t)
}

pub(crate) struct HybMggaXM062x {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl HybMggaXM062x {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::HybMggaXM062x),
                name: "hyb_mgga_x_m06_2x",
                family: Family::HybMgga,
                kind: Kind::Exchange,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-15, // libxc hyb_mgga_x_m06_2x threshold
                hybrid: Some(HybridInfo {
                    exx_fraction: EXX_FRACTION,
                    cam: None,
                    vv10: None,
                }),
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self::new()))
    }
}

impl MggaEnergy for HybMggaXM062x {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        mgga_exchange(
            &v,
            self.info.dens_threshold,
            self.zeta_threshold,
            m06_2x_enhancement,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduced::consts::K_FACTOR_C;
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn m062x(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybMggaXM062x, spin).unwrap()
    }

    /// Metadata: hybrid meta-GGA exchange with 54% EXX (Zhao & Truhlar 2008).
    #[test]
    fn metadata_reports_54_percent_exx() {
        let f = m062x(Spin::Unpolarized);
        assert_eq!(f.exx_fraction(), 0.54);
        assert_eq!(f.info().name, "hyb_mgga_x_m06_2x");
        assert!(f.info().needs_tau);
    }

    /// At the uniform-gas point (`σ = 0`, `τ = τ_unif` ⇒ `w(t) = 0`, series =
    /// `a[0]`), the semilocal enhancement is `a[0] = 0.46` — by design the DFT
    /// part recovers only `1 − exx − …`-scaled Slater; together with `0.54` EXX
    /// the *total* exchange is normalized (a[0] + 0.54 = 1).
    #[test]
    fn uniform_gas_semilocal_part_is_a0_times_lda_x() {
        assert!(
            (A[0] + EXX_FRACTION - 1.0).abs() < 1e-12,
            "a0 + exx must be 1"
        );
        let m = m062x(Spin::Unpolarized);
        let lda = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        for &n in &[0.1_f64, 1.0, 7.3, 100.0] {
            let tau_unif = 2.0 * K_FACTOR_C * (n / 2.0).powf(5.0 / 3.0);
            let mm = m
                .eval(1, &XcInput::gga(&[n], &[0.0]).with_tau(&[tau_unif]))
                .unwrap();
            let ll = lda.eval(1, &XcInput::lda(&[n])).unwrap();
            assert!(
                (mm.exc[0] - A[0] * ll.exc[0]).abs() <= 1e-10 * ll.exc[0].abs(),
                "exc n={n}: {} vs a0·{}",
                mm.exc[0],
                ll.exc[0]
            );
        }
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
            (0.1, 0.02, 0.05),
            (10.0, 5.0, 20.0),
            (1.0, 0.4, 0.06), // τ ≈ τ_W (w(t) large)
        ] {
            let out = f
                .eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                .unwrap();
            let (hn, hs, ht) = (1e-6 * n, 1e-6 * s, 1e-6 * tau);
            let fdn = (edens(n + hn, s, tau) - edens(n - hn, s, tau)) / (2.0 * hn);
            let fds = (edens(n, s + hs, tau) - edens(n, s - hs, tau)) / (2.0 * hs);
            let fdt = (edens(n, s, tau + ht) - edens(n, s, tau - ht)) / (2.0 * ht);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-6 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s} t={tau}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-6 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s} t={tau}: {} vs {fds}",
                out.vsigma[0]
            );
            assert!(
                (out.vtau[0] - fdt).abs() <= 1e-6 * out.vtau[0].abs().max(1.0),
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
                (out.vrho[k] - fd).abs() <= 1e-6 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        // pure exchange ⇒ vsigma_ab = 0
        assert_eq!(out.vsigma[1], 0.0, "exchange vsigma_ab must be 0");
        for (k, h) in [(0usize, 1e-6 * ta), (1, 1e-6 * tb)] {
            let (mut tp, mut tm) = (t, t);
            tp[k] += h;
            tm[k] -= h;
            let fd = (edens(r, s, tp) - edens(r, s, tm)) / (2.0 * h);
            assert!(
                (out.vtau[k] - fd).abs() <= 1e-6 * out.vtau[k].abs().max(1.0),
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
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-12 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-11 * ou.vrho[0].abs());
        assert!((ou.vtau[0] - op.vtau[0]).abs() <= 1e-11 * ou.vtau[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = m062x(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-12, 1e-13, 1.0, 1.0, 100.0, 50.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-20, 0.0, 1e-22, //
            1e6, 1e6, 1e6, // τ < τ_W ⇒ FHC clamp active
            1.0, 0.5, 0.8, //
        ];
        let tau = [0.5, 0.0, 0.0, 0.5, 1e-15, 1e-16, 0.1, 0.1, 50.0, 30.0];
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
