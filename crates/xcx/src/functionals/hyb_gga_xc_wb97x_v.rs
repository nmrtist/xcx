// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! ωB97X-V — `hyb_gga_xc_wb97x_v` (libxc 466): range-separated hybrid GGA with
//! VV10 nonlocal correlation. N. Mardirossian & M. Head-Gordon, *Phys. Chem.
//! Chem. Phys.* **16**, 9904 (2014).
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/hyb_gga_xc_wb97.c`
//! (`par_wb97x_v`) + `maple/gga_exc/hyb_gga_xc_wb97.mpl` + `maple/b97.mpl`
//! (`b97_g`) + `maple/lda_exc/lda_x_erf.mpl` + `maple/attenuation.mpl` +
//! `maple/lda_exc/lda_c_pw.mpl` (**standard** PW92 set — the wb97 maple defines
//! only `lda_c_pw_params`, unlike B97M-V's modified set) + `maple/util.mpl`
//! (`lda_stoll_par`/`lda_stoll_perp`).
//!
//! ## Scope fence
//!
//! xcx evaluates only the **semilocal** part: the SR-erf-attenuated B97
//! exchange plus B97 correlation. The host assembles, from metadata,
//! - exact exchange `EXX(r₁₂) = α + β·erf(ω·r₁₂)` with
//!   `ω = 0.30, α = 0.167, β = 0.833` (libxc `cam_alpha = 1.0`,
//!   `cam_beta = −(1 − 0.167)` translated to the xcx frozen convention), and
//! - the VV10 nonlocal term with `b = 6.0, C = 0.01` (libxc `nlc_b`/`nlc_C`).
//!
//! ## Form
//!
//! Per spin channel, exchange is SR-LDA × the B97 inhomogeneity series:
//! `lda_x_spin(rs, 1±z) · F(a_σ) · g(γ_x, c_x, x_σ²)`, with `F` the shared
//! [`attenuation_erf`] and `a_σ = (4/(9π))^(1/3)·(ω/2)·rs/(1±z)^(1/3)`
//! (`lda_x_erf.mpl`, channel-scaled `rs`). Correlation is the B97 Stoll split
//! of standard PW92: same-spin `stoll_par(±z)·g(γ_ss, c_ss, x_σ²)`,
//! opposite-spin `stoll_perp(z)·g(γ_os, c_os, (x₀²+x₁²)/2)`. The series
//! `g(γ, c, w) = Σ c_i·u^i`, `u = γw/(1+γw)` is the shared sqrt-free
//! [`b97_g`] (every gradient enters squared; the maple's opposite-spin
//! `√(x₀²+x₁²)/√2` cancels inside `u`).
//!
//! ωB97X-V truncates the B97 series at 3/2/2 nonzero terms (libxc keeps the
//! 5-term array with zeros; we keep the full array so `b97_g` is reused as-is).

use num_dual::DualNum;

use super::attenuation::{att_a_cnst, attenuation_erf};
use super::lda_c_pw::{pw92_ec, A_STD, FZ20_STD};
use super::mgga_c_m06_l::b97_g;
use crate::families::gga::{Gga, GgaEnergy, GgaVars};
use crate::families::XcEval;
use crate::func::{CamParams, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind, Vv10Params};
use crate::reduced::vars;

/// Range separation ω (libxc `_omega` = 0.3).
pub(crate) const OMEGA: f64 = 0.3;
/// Global (short-range) exact-exchange fraction α in the xcx CAM convention
/// `EXX(r₁₂) = α + β·erf(ω·r₁₂)` (libxc: `cam_alpha + cam_beta = 0.167`).
pub(crate) const CAM_ALPHA: f64 = 0.167;
/// Long-range increment β: `α + β = 1` (100% LR exact exchange).
pub(crate) const CAM_BETA: f64 = 1.0 - CAM_ALPHA;

// B97 power-series coefficients and γ's (libxc `par_wb97x_v` / the shared wb97
// maple): exchange, same-spin, opposite-spin.
const GAMMA_X: f64 = 0.004;
const GAMMA_SS: f64 = 0.2;
const GAMMA_OS: f64 = 0.006;
const C_X: [f64; 5] = [0.833, 0.603, 1.194, 0.0, 0.0];
const C_SS: [f64; 5] = [0.556, -0.257, 0.0, 0.0, 0.0];
const C_OS: [f64; 5] = [1.219, -1.850, 0.0, 0.0, 0.0];

pub(crate) struct HybGgaXcWb97xV {
    info: FunctionalInfo,
    zeta_threshold: f64,
}

impl HybGgaXcWb97xV {
    pub(crate) fn boxed() -> Box<dyn XcEval> {
        Box::new(Gga(Self {
            info: FunctionalInfo {
                id: Some(FunctionalId::HybGgaXcWb97xV),
                name: "hyb_gga_xc_wb97x_v",
                family: Family::HybGga,
                kind: Kind::ExchangeCorrelation,
                needs_sigma: true,
                needs_lapl: false,
                needs_tau: false,
                dens_threshold: 1e-14, // libxc hyb_gga_xc_wb97x_v threshold
                hybrid: Some(HybridInfo {
                    exx_fraction: CAM_ALPHA,
                    cam: Some(CamParams {
                        omega: OMEGA,
                        alpha: CAM_ALPHA,
                        beta: CAM_BETA,
                    }),
                    vv10: Some(Vv10Params { b: 6.0, c: 0.01 }),
                }),
            },
            zeta_threshold: f64::EPSILON,
        }))
    }
}

impl GgaEnergy for HybGgaXcWb97xV {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn f<N: DualNum<f64> + Copy>(&self, v: GgaVars<N>) -> N {
        let zt = self.zeta_threshold;
        let thr = self.info.dens_threshold;
        let GgaVars {
            rs,
            z,
            opz,
            omz,
            na,
            nb,
            xs0_sq,
            xs1_sq,
            ..
        } = v;
        let a_cnst = att_a_cnst(OMEGA);

        // Channel screens: libxc `screen_dens_zeta` (spin density at the floor,
        // or the channel fully unpopulated).
        let up_screened = na.re() <= thr || opz.re() <= zt;
        let dn_screened = nb.re() <= thr || omz.re() <= zt;

        // --- SR-erf B97 exchange, per channel (wb97 maple `wb97_x`) ---
        let x_chan = |opzv: N, xs_sq: N| {
            let a = N::from(a_cnst) * rs / opzv.cbrt();
            vars::lda_x_spin(rs, opzv, zt) * attenuation_erf(a) * b97_g(GAMMA_X, &C_X, xs_sq)
        };
        let x_up = if up_screened {
            N::from(0.0)
        } else {
            x_chan(opz, xs0_sq)
        };
        let x_dn = if dn_screened {
            N::from(0.0)
        } else {
            x_chan(omz, xs1_sq)
        };

        // --- B97 correlation on the Stoll split of *standard* PW92 ---
        let f_pw = |rs: N, zz: N| pw92_ec(rs, zz, zt, &A_STD, FZ20_STD);
        let par_up = if up_screened {
            N::from(0.0)
        } else {
            opz / N::from(2.0) * f_pw(rs * (N::from(2.0) / opz).powf(1.0 / 3.0), N::from(1.0))
        };
        let par_dn = if dn_screened {
            N::from(0.0)
        } else {
            omz / N::from(2.0) * f_pw(rs * (N::from(2.0) / omz).powf(1.0 / 3.0), N::from(-1.0))
        };
        let perp = f_pw(rs, z) - par_up - par_dn;

        let c_up = if up_screened {
            N::from(0.0)
        } else {
            par_up * b97_g(GAMMA_SS, &C_SS, xs0_sq)
        };
        let c_dn = if dn_screened {
            N::from(0.0)
        } else {
            par_dn * b97_g(GAMMA_SS, &C_SS, xs1_sq)
        };
        // opposite-spin gradient: maple's √(x₀²+x₁²)/√2 squared = (x₀²+x₁²)/2
        let cross = perp * b97_g(GAMMA_OS, &C_OS, (xs0_sq + xs1_sq) / N::from(2.0));

        x_up + x_dn + c_up + c_dn + cross
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::Rung;
    use crate::{Functional, Spin, XcInput};

    fn wb97xv(spin: Spin) -> Functional {
        Functional::new(FunctionalId::HybGgaXcWb97xV, spin).unwrap()
    }

    /// Acceptance metadata: RangeSeparatedHybrid rung, verified CAM params
    /// (ω = 0.30, α = 0.167, β = 0.833 in EXX(r₁₂) = α + β·erf(ω·r₁₂)),
    /// exx_fraction == α, VV10 b = 6.0 / C = 0.01, level-4 grid-sensitive grid.
    #[test]
    fn metadata_cam_vv10_grid() {
        let f = wb97xv(Spin::Unpolarized);
        let info = f.info();
        assert_eq!(info.rung(), Rung::RangeSeparatedHybrid);
        let h = info.hybrid.unwrap();
        let cam = h.cam.unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.3, 0.167, 1.0 - 0.167));
        assert_eq!(f.exx_fraction(), 0.167);
        let vv10 = h.vv10.unwrap();
        assert_eq!((vv10.b, vv10.c), (6.0, 0.01));
        let g = info.grid();
        assert_eq!((g.level, g.grid_sensitive), (4, true));
        assert!(!info.needs_tau && info.needs_sigma);
    }

    #[test]
    fn unpol_derivs_match_finite_difference() {
        let f = wb97xv(Spin::Unpolarized);
        let edens = |n: f64, s: f64| n * f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
        for &(n, s) in &[
            (0.5, 0.1),
            (2.0, 0.7),
            (0.3, 0.02),
            (5.0, 3.0),
            (1e-3, 1e-8),
        ] {
            let out = f.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
            let (hn, hs) = (1e-6 * n, 1e-6 * s);
            let fdn = (edens(n + hn, s) - edens(n - hn, s)) / (2.0 * hn);
            let fds = (edens(n, s + hs) - edens(n, s - hs)) / (2.0 * hs);
            assert!(
                (out.vrho[0] - fdn).abs() <= 1e-5 * out.vrho[0].abs().max(1.0),
                "vrho n={n} s={s}: {} vs {fdn}",
                out.vrho[0]
            );
            assert!(
                (out.vsigma[0] - fds).abs() <= 1e-5 * out.vsigma[0].abs().max(1.0),
                "vsigma n={n} s={s}: {} vs {fds}",
                out.vsigma[0]
            );
        }
    }

    #[test]
    fn pol_derivs_match_finite_difference() {
        let f = wb97xv(Spin::Polarized);
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
                (out.vrho[k] - fd).abs() <= 1e-5 * out.vrho[k].abs().max(1.0),
                "vrho[{k}]: {} vs {fd}",
                out.vrho[k]
            );
        }
        // σ_ab never enters (per-spin gradients only) ⇒ vsigma_ab = 0.
        assert_eq!(out.vsigma[1], 0.0);
        for (k, h) in [(0usize, 1e-6 * saa), (2usize, 1e-6 * sbb)] {
            let (mut sp, mut sm) = (s, s);
            sp[k] += h;
            sm[k] -= h;
            let fd = (edens(r, sp) - edens(r, sm)) / (2.0 * h);
            assert!(
                (out.vsigma[k] - fd).abs() <= 1e-5 * out.vsigma[k].abs().max(1.0),
                "vsigma[{k}]: {} vs {fd}",
                out.vsigma[k]
            );
        }
    }

    #[test]
    fn unpol_pol_symmetry_at_zero_polarization() {
        let up = wb97xv(Spin::Unpolarized);
        let po = wb97xv(Spin::Polarized);
        let (n, s) = (0.8, 0.3);
        let ou = up.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
        let op = po
            .eval(
                1,
                &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0]),
            )
            .unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-11 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
    }

    #[test]
    fn edge_outputs_finite() {
        let f = wb97xv(Spin::Polarized);
        let rho = [1.0, 0.0, 0.0, 1.0, 1e-10, 1e-11, 1.0, 1.0, 1000.0, 500.0];
        let sigma = [
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            1e-18, 0.0, 1e-20, //
            1e6, 1e6, 1e6, //
            1e6, 5e5, 8e5, // large density: very small attenuation argument a
        ];
        let out = f.eval(5, &XcInput::gga(&rho, &sigma)).unwrap();
        for v in out.exc.iter().chain(&out.vrho).chain(&out.vsigma) {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
