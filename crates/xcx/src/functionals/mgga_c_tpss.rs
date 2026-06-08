// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Tao–Perdew–Staroverov–Scuseria correlation — `mgga_c_tpss` (libxc 231).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `maple/mgga_exc/mgga_c_tpss.mpl` +
//! `maple/tpss_c.mpl` + `maple/gga_exc/gga_c_pbe.mpl` + `maple/util.mpl`.
//!
//! TPSS correlation is built on PBE correlation `f_pbe` (the shared
//! [`pbe_c_energy`]; same modified-PW92 uniform limit + gradient correction H).
//! It forms a self-interaction-corrected "revised PKZB" energy (eq. 24–25 of the
//! paper):
//! `f = f0·(1 + d·f0·z̃³)`, `f0 = perp + par`,
//! `perp = (1 + C0·z̃²)·f_pbe(rs, ζ, x_t)`,
//! `par  = −(1 + C0)·z̃²·Σ_σ [screened? f_pbe·(1±ζ)/2 : max(f_pbe^σ, f_pbe)·(1±ζ)/2]`,
//! where `z̃ = min(x_t²/(8 τ_red), 1)` is the (clamped) total von-Weizsäcker ratio,
//! `C0(ζ, ξ²)` the gradient-dependent coefficient (eq. 33–34), and the per-spin
//! `f_pbe^σ` is PBE-C of that spin treated as a fully-polarized density. All reduced
//! gradients are **squared** (sqrt-free); τ enters via the reduced KE `t_σ`. The
//! Laplacian is unused (`needs_lapl = false`).

use num_dual::DualNum;

use super::gga_c_pbe::{pbe_c_energy, BETA};
use crate::families::mgga::{Mgga, MggaEnergy, MggaVars};
use crate::families::XcEval;
use crate::func::{Family, FunctionalId, FunctionalInfo, Kind};
use crate::reduced::vars::{one_minus_z_pow2, opz_pow, t_total};

// TPSS correlation parameters (libxc `tpss_values`). `beta` is folded into the
// shared PBE-C; here we need `d` (eq. 24) and the four `C0` coefficients (eq. 33).
const D: f64 = 2.8;
const C0_C: [f64; 4] = [0.53, 0.87, 0.50, 2.26];
/// `(2·(3π²)^(1/3))²` — the denominator of ξ² (eq. 28), the squared gradient of ζ.
const XI2_DENOM: f64 = 38.283_120_002_509_214;

/// libxc `z_thr`: clamp ζ to `[zeta_threshold − 1, 1 − zeta_threshold]`. The branch
/// is taken on the real part so forward-AD follows the same piece libxc's
/// generated C does. Provenance: util.mpl `z_thr`.
fn z_thr<N: DualNum<f64> + Copy>(z: N, zeta_threshold: f64) -> N {
    if 1.0 + z.re() <= zeta_threshold {
        N::from(zeta_threshold - 1.0)
    } else if 1.0 - z.re() <= zeta_threshold {
        N::from(1.0 - zeta_threshold)
    } else {
        z
    }
}

/// ξ² (eq. 28 squared): `(1−ζ²)·(t_total(ζ, x_{s0}², x_{s1}²) − x_t²)/(2(3π²)^(1/3))²`,
/// taking the **squared** per-spin and total reduced gradients. ξ² → 0 at ζ = 0
/// (the two gradient measures coincide there). Provenance: tpss_c.mpl `tpss_xi2`.
fn tpss_xi2<N: DualNum<f64> + Copy>(z: N, xt2: N, xs0_sq: N, xs1_sq: N) -> N {
    one_minus_z_pow2(z) * (t_total(z, xs0_sq, xs1_sq) - xt2) / N::from(XI2_DENOM)
}

/// `C0` denominator (eq. 34): `(1 + ξ²·((1+z)^(−4/3) + (1−z)^(−4/3))/2)`, with the
/// `(1±z)^(−4/3)` clamped at `zeta_threshold` (libxc `opz_pow_n`). Called with
/// `z = z_thr(z)`. Provenance: tpss_c.mpl `tpss_C0_den`.
fn tpss_c0_den<N: DualNum<f64> + Copy>(
    z: N,
    xt2: N,
    xs0_sq: N,
    xs1_sq: N,
    zeta_threshold: f64,
) -> N {
    let xi2 = tpss_xi2(z, xt2, xs0_sq, xs1_sq);
    let pow_sum = opz_pow(N::from(1.0) + z, -4.0 / 3.0, zeta_threshold)
        + opz_pow(N::from(1.0) - z, -4.0 / 3.0, zeta_threshold);
    N::from(1.0) + xi2 * pow_sum / N::from(2.0)
}

/// `C0(ζ, ξ²)` (eq. 33–34): `C00(ζ)/C0_den(z_thr(ζ))⁴` for `|ζ| < 1 − 1e-12`, else
/// the constant `Σ c_i` (libxc's guard against the C0 derivative diverging for
/// ferromagnetic densities — `1 − |z| ≤ 1e-12`). `C00(ζ) = Σ c_i ζ^(2(i−1))`.
/// Provenance: tpss_c.mpl `tpss_C0`/`tpss_C00`.
fn tpss_c0<N: DualNum<f64> + Copy>(z: N, xt2: N, xs0_sq: N, xs1_sq: N, zeta_threshold: f64) -> N {
    if 1.0 - z.re().abs() <= 1e-12 {
        N::from(C0_C[0] + C0_C[1] + C0_C[2] + C0_C[3])
    } else {
        let z2 = z * z;
        let z4 = z2 * z2;
        let z6 = z4 * z2;
        let c00 = N::from(C0_C[0])
            + N::from(C0_C[1]) * z2
            + N::from(C0_C[2]) * z4
            + N::from(C0_C[3]) * z6;
        let den = tpss_c0_den(
            z_thr(z, zeta_threshold),
            xt2,
            xs0_sq,
            xs1_sq,
            zeta_threshold,
        );
        let den2 = den * den;
        c00 / (den2 * den2)
    }
}

pub(crate) struct MggaCTpss {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl MggaCTpss {
    fn new() -> Self {
        Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::MggaCTpss),
                name: "mgga_c_tpss",
                family: Family::Mgga,
                kind: Kind::Correlation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: true,
                dens_threshold: 1e-15,
                hybrid: None,
            },
            zeta_threshold: f64::EPSILON, // libxc default (DBL_EPSILON)
        }
    }

    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Mgga(Self::new()))
    }
}

impl MggaEnergy for MggaCTpss {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
        let zt = self.zeta_threshold;
        let thr = self.info.dens_threshold;
        let MggaVars {
            rs,
            z,
            na,
            nb,
            xt2,
            xs0_sq,
            xs1_sq,
            t0,
            t1,
            ..
        } = v;

        // f_pbe(rs, ζ, x_t²): the shared PBE-correlation energy.
        // TPSS-c is built on PBE-c with the standard PBE β (BETA).
        let fpbe = |rs: N, z: N, xt2: N| pbe_c_energy(rs, z, xt2, zt, BETA);

        // z̃ = min(x_t²/(8·τ_red), 1) — total von-Weizsäcker ratio (τ_red = τ/n^(5/3)).
        let tau_red = t_total(z, t0, t1);
        let aux_raw = xt2 / (N::from(8.0) * tau_red);
        let aux = if aux_raw.re() > 1.0 {
            N::from(1.0)
        } else {
            aux_raw
        };
        let aux2 = aux * aux;

        let c0 = tpss_c0(z, xt2, xs0_sq, xs1_sq, zt);
        let zc = z_thr(z, zt);

        // perp = (1 + C0·z̃²)·f_pbe(rs, ζ, x_t²).
        let perp = (N::from(1.0) + c0 * aux2) * fpbe(rs, z, xt2);

        // par: per-spin parallel correlation, screened on each channel's density.
        let fpbe_clamped = fpbe(rs, zc, xt2);
        // spin-up
        let term_up = if na.re() <= thr || 1.0 + z.re() <= zt {
            fpbe_clamped * (N::from(1.0) + z) / N::from(2.0)
        } else {
            let rs_up = rs * (N::from(2.0) / (N::from(1.0) + zc)).powf(1.0 / 3.0);
            let a = fpbe(rs_up, N::from(1.0), xs0_sq);
            let mx = if a.re() > fpbe_clamped.re() {
                a
            } else {
                fpbe_clamped
            };
            mx * (N::from(1.0) + zc) / N::from(2.0)
        };
        // spin-down
        let term_dn = if nb.re() <= thr || 1.0 - z.re() <= zt {
            fpbe_clamped * (N::from(1.0) - z) / N::from(2.0)
        } else {
            let rs_dn = rs * (N::from(2.0) / (N::from(1.0) - zc)).powf(1.0 / 3.0);
            let a = fpbe(rs_dn, N::from(-1.0), xs1_sq);
            let mx = if a.re() > fpbe_clamped.re() {
                a
            } else {
                fpbe_clamped
            };
            mx * (N::from(1.0) - zc) / N::from(2.0)
        };
        let par = -(N::from(1.0) + c0) * aux2 * (term_up + term_dn);

        // f0 then the revised-PKZB self-interaction correction (eq. 24).
        let f0 = perp + par;
        f0 * (N::from(1.0) + N::from(D) * f0 * aux2 * aux)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    fn tpss(spin: Spin) -> Functional {
        Functional::new(FunctionalId::MggaCTpss, spin).unwrap()
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = tpss(Spin::Unpolarized);
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
        let f = tpss(Spin::Polarized);
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
        for (k, h) in [(0usize, 1e-6 * saa), (1, 1e-6 * sab), (2, 1e-6 * sbb)] {
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
        let up = tpss(Spin::Unpolarized);
        let po = tpss(Spin::Polarized);
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
        let f = tpss(Spin::Polarized);
        let rho = [
            1.0, 0.0, // full polarization
            0.0, 1.0, //
            1e-10, 1e-11, // small densities
            1.0, 1.0, //
            100.0, 50.0, //
        ];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1.0, 0.5, 0.8, //
        ];
        let tau = [
            0.5, 0.0, //
            0.0, 0.5, //
            1e-12, 1e-13, //
            0.5, 0.5, //
            50.0, 30.0, //
        ];
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
