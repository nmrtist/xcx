// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Meta-GGA family: energy trait, autodiff harness, object-safe wrapper.
//!
//! The exact analogue of [`super::gga`], with the kinetic-energy density τ added
//! to the differentiated seed. The harness mirrors libxc's `work_mgga`
//! (golden-locked): screen on the total density (strict `<`), floor each spin
//! density to `dens_threshold`, each σ_aa/σ_bb to `sigma_threshold²`, clamp σ_ab
//! to `[−s_ave, +s_ave]`, and floor each τ_σ to `tau_threshold` (libxc default
//! `1e-20`) — then seed forward-AD at those floored values so vrho/vsigma/vtau
//! are derivatives w.r.t. the floored inputs, as libxc does.
//!
//! **Laplacian:** the public `XcInput::lapl` field is accepted but unused — every
//! v0.2 meta-GGA (TPSS, r2SCAN, M06-L) is `needs_lapl = false`. The reduced
//! Laplacian `u_σ` is therefore not seeded into the AD vector and the `lapl` fxc
//! blocks (`v2rholapl`, `v2sigmalapl`, `v2lapl2`, `v2lapltau`) are left empty;
//! both are deferred until a Laplacian-dependent functional is added (at which
//! point the seed grows from 7 to 9 components — an additive change).

use nalgebra::SVector;
use num_dual::{gradient, hessian, Dual2SVec64, DualNum, DualSVec64};

use super::{check_len, XcEval};
use crate::error::XcError;
use crate::func::{FunctionalInfo, Spin};
use crate::io::{XcInput, XcResult};
use crate::reduced::vars;

/// libxc's default `tau_threshold` (`functionals.c`): the floor applied to each
/// spin kinetic-energy density τ_σ before evaluation. None of the v0.2 functionals
/// override it.
const TAU_THRESHOLD: f64 = 1e-20;

/// Whether to enforce the Fermi-hole-curvature constraint `1 − x_σ²/(8 t_σ) ≥ 0`
/// by clamping each spin σ to `min(σ_σσ, 8·n_σ·τ_σ)` — libxc's `work_mgga`
/// `XC_FLAGS_ENFORCE_FHC` step. This is **on** here because it is on in the pinned
/// conda-forge libxc 6.1.0 build that the golden snapshots are generated from
/// (compiled with `XC_ENFORCE_FERMI_HOLE_CURVATURE`; FFI-confirmed: TPSS `exc` at
/// `σ = 1e-8` equals `exc` at `σ = 8nτ`, i.e. σ is clamped). Per the match-libxc
/// policy (CLAUDE.md §2) the golden build is the single source of truth, so xcx
/// reproduces this clamp. Like the other `work_mgga` floors/clamps it is applied to
/// the **seed value**, and AD then differentiates w.r.t. the (clamped) σ slot
/// without chaining through the `min` — matching libxc's emitted derivatives.
const ENFORCE_FHC: bool = true;

/// Apply the Fermi-hole-curvature clamp to one spin's σ given its floored density
/// and τ: `σ ← min(σ, 8·n·τ)` when [`ENFORCE_FHC`].
#[inline]
fn fhc_clamp(sigma: f64, n: f64, tau: f64) -> f64 {
    if ENFORCE_FHC {
        sigma.min(8.0 * n * tau)
    } else {
        sigma
    }
}

/// Reduced variables passed to a meta-GGA energy expression. Extends
/// [`super::gga::GgaVars`] with the per-spin reduced kinetic-energy density
/// `t_σ = τ_σ/n_σ^(5/3)` (τ enters directly, sqrt-free — see [`vars::reduced_tau`]).
/// As in the GGA harness, `opz`/`omz` are the cancellation-free `1 ± z = 2n_{a,b}/n`
/// and all reduced gradients are carried **squared** (`xt2`, `xs0_sq`, `xs1_sq`).
#[derive(Clone, Copy)]
pub(crate) struct MggaVars<N> {
    /// Wigner–Seitz radius of the total density.
    pub rs: N,
    /// Relative spin polarization `z = (n_a − n_b)/n`.
    pub z: N,
    /// `1 + z`, computed cancellation-free as `2·n_a/n`.
    pub opz: N,
    /// `1 − z`, computed cancellation-free as `2·n_b/n`.
    pub omz: N,
    /// Spin-up density (floored), for direct per-channel screening like libxc.
    pub na: N,
    /// Spin-down density (floored).
    pub nb: N,
    /// Squared total reduced gradient `x_t² = (σ_aa+2σ_ab+σ_bb)/n^(8/3)`.
    pub xt2: N,
    /// Squared spin-up reduced gradient `x_{s0}² = σ_aa/n_a^(8/3)`, sqrt-free.
    pub xs0_sq: N,
    /// Squared spin-down reduced gradient `x_{s1}² = σ_bb/n_b^(8/3)`, sqrt-free.
    pub xs1_sq: N,
    /// Spin-up reduced kinetic-energy density `t_0 = τ_a/n_a^(5/3)` (τ direct).
    pub t0: N,
    /// Spin-down reduced kinetic-energy density `t_1 = τ_b/n_b^(5/3)`.
    pub t1: N,
}

/// Meta-GGA-exchange skeleton shared by every meta-GGA exchange functional — the
/// Rust mirror of libxc's `util.mpl` `mgga_exchange(func, rs, z, xs0, xs1, u0, u1,
/// t0, t1)`. Each spin channel is the LDA exchange of that channel
/// (`lda_x_spin`, cancellation-free `opz`/`omz`) times an enhancement factor
/// `F_x` of that channel's **squared** reduced gradient `x_σ²` and reduced kinetic
/// energy `t_σ`, screened independently on the floored spin density — exactly as
/// `lda_x`. A functional supplies only `enhancement`, the dimensionless `F_x` as a
/// function `(x_σ², t_σ) → F`. (The Laplacian argument `u_σ` of libxc's
/// `mgga_exchange` is omitted: no v0.2 functional uses it.) Provenance:
/// ported-from-libxc (MPL-2.0), `maple/util.mpl` `mgga_exchange`.
pub(crate) fn mgga_exchange<N, F>(
    v: &MggaVars<N>,
    dens_threshold: f64,
    zeta_threshold: f64,
    enhancement: F,
) -> N
where
    N: DualNum<f64> + Copy,
    F: Fn(N, N) -> N,
{
    let up = if vars::screen_dens(v.na, dens_threshold) {
        N::from(0.0)
    } else {
        vars::lda_x_spin(v.rs, v.opz, zeta_threshold) * enhancement(v.xs0_sq, v.t0)
    };
    let dn = if vars::screen_dens(v.nb, dens_threshold) {
        N::from(0.0)
    } else {
        vars::lda_x_spin(v.rs, v.omz, zeta_threshold) * enhancement(v.xs1_sq, v.t1)
    };
    up + dn
}

/// The rSCAN / r2SCAN interpolation switch `f(α)`, shared by r2SCAN **exchange**
/// and **correlation** — they differ only in the constants `c1`/`c2`/`d` and the
/// polynomial coefficients, so per the reuse rule this is one parameterized source
/// (guarded by each functional's golden recovery). Three branches selected on the
/// real part of `α` (libxc's `my_piecewise5`): the left tail `exp(−c1·α/(1−α))` for
/// `α ≤ 0`, the degree-7 polynomial `Σ_{i=0}^{7} coeffs[7−i]·α^i` (Horner) for
/// `0 < α ≤ 2.5`, and the right tail `−d·exp(c2/(1−α))` for `α > 2.5`. `coeffs` is
/// libxc's **reversed**-order list (`coeffs[7]` is the constant term).
///
/// AD-safety: each tail's `(1−α)` denominator is bounded away from 0 in its branch
/// (`≥ 1` for `α ≤ 0`, `≤ −1.5` for `α > 2.5`), so neither `exp` argument has a
/// pole; within a branch the maple `m_min`/`m_max` argument clamps are the identity,
/// so forward-AD of the selected branch equals libxc's emitted derivative. The
/// branch jumps in the *derivative* across the α = 0 / α = 2.5 seams are libxc's
/// exact (continuous-value, switch-class) behavior, reproduced by the same
/// real-part branch selection. Provenance: ported-from-libxc (MPL-2.0),
/// `maple/mgga_exc/mgga_x_rscan.mpl` / `mgga_c_rscan.mpl` (`rscan_f_alpha_small`/
/// `_large`) + `mgga_x_r2scan.mpl` / `mgga_c_r2scan.mpl` (`r2scan_f_alpha_neg`).
pub(crate) fn rscan_f_alpha<N: DualNum<f64> + Copy>(
    a: N,
    c1: f64,
    c2: f64,
    d: f64,
    coeffs: &[f64; 8],
) -> N {
    if a.re() <= 0.0 {
        (-N::from(c1) * a / (N::from(1.0) - a)).exp()
    } else if a.re() <= 2.5 {
        let mut poly = N::from(coeffs[0]);
        for &c in &coeffs[1..] {
            poly = poly * a + N::from(c);
        }
        poly
    } else {
        -N::from(d) * (N::from(c2) / (N::from(1.0) - a)).exp()
    }
}

/// The VS98 / Minnesota "gvt4" working term `gtv4(α, d, w, z)`, shared by M06-L
/// **exchange** (its VS98-type local correction) and **correlation** (the VSXC
/// parallel/perpendicular pieces) — they differ only in `α`, the six `d`
/// coefficients, and the arguments, so per the reuse rule this is one
/// parameterized source. libxc's `maple/gvt4.mpl`:
/// ```text
/// gvt4_gamm(α, x, z) = 1 + α·(x² + z)
/// gtv4(α, d, x, z)  = d₁/γ + (d₂x² + d₃z)/γ² + (d₄x⁴ + d₅x²z + d₆z²)/γ³
/// ```
/// Only the **even** powers `x²`, `x⁴` of the reduced gradient appear, so this
/// takes the **squared** reduced gradient `w = x²` directly (sqrt-free,
/// AD-safe; `x⁴ = w²`) — never `√σ`. The denominator `γ = 1 + α·(w + z)` is
/// strictly positive over the physical domain (`w ≥ 0`, `α` tiny, `z = 2(t−K)`
/// bounded below by `−2K` since `t ≥ 0`), so there is no pole. `d` is libxc's
/// 0-indexed coefficient array (`d[0]` is the maple's `d₁`). Provenance:
/// ported-from-libxc (MPL-2.0), `maple/gvt4.mpl`.
pub(crate) fn gtv4<N: DualNum<f64> + Copy>(alpha: f64, d: &[f64; 6], w: N, z: N) -> N {
    let gamm = N::from(1.0) + N::from(alpha) * (w + z);
    let g2 = gamm * gamm;
    let g3 = g2 * gamm;
    N::from(d[0]) / gamm
        + (N::from(d[1]) * w + N::from(d[2]) * z) / g2
        + (N::from(d[3]) * (w * w) + N::from(d[4]) * (w * z) + N::from(d[5]) * (z * z)) / g3
}

/// A meta-GGA functional's energy per particle, written generically over a
/// dual-number scalar. Mirrors libxc's `f := (rs, z, xt, xs0, xs1, u0, u1, t0, t1)
/// -> ...` (with the unused Laplacian arguments dropped).
pub(crate) trait MggaEnergy: Send + Sync {
    fn info(&self) -> &FunctionalInfo;
    fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N;
}

/// Object-safe wrapper turning any [`MggaEnergy`] into an [`XcEval`].
pub(crate) struct Mgga<F: MggaEnergy>(pub F);

impl<F: MggaEnergy> XcEval for Mgga<F> {
    fn info(&self) -> &FunctionalInfo {
        self.0.info()
    }

    fn eval(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let sigma = input.sigma.ok_or(XcError::MissingInput("sigma"))?;
        let tau = input.tau.ok_or(XcError::MissingInput("tau"))?;
        match spin {
            Spin::Unpolarized => self.eval_unpol(np, input.rho, sigma, tau),
            Spin::Polarized => self.eval_pol(np, input.rho, sigma, tau),
        }
    }

    fn eval_fxc(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let sigma = input.sigma.ok_or(XcError::MissingInput("sigma"))?;
        let tau = input.tau.ok_or(XcError::MissingInput("tau"))?;
        match spin {
            Spin::Unpolarized => self.eval_fxc_unpol(np, input.rho, sigma, tau),
            Spin::Polarized => self.eval_fxc_pol(np, input.rho, sigma, tau),
        }
    }
}

impl<F: MggaEnergy> Mgga<F> {
    /// libxc's σ floor: `sigma_threshold² = (dens_threshold^(4/3))²`.
    fn sigma_floor(&self) -> f64 {
        let st = self.0.info().dens_threshold.powf(4.0 / 3.0);
        st * st
    }

    /// Unpolarized energy density `e = n·f` at one point, generic over the dual
    /// scalar `N` so the *same* expression feeds both the gradient (vxc) and the
    /// Hessian (fxc). Seed vector is `[n, σ, τ]` (floored). Per the unpolarized
    /// convention each channel has `n_σ = n/2`, `σ_σσ = σ/4`, `τ_σ = τ/2`.
    fn energy_unpol<N: DualNum<f64> + Copy>(&self, x: &SVector<N, 3>) -> N {
        let n = x[0];
        let s = x[1];
        let tau = x[2];
        let rs = vars::rs_from_n(n);
        let half = n / N::from(2.0);
        let xt2 = vars::reduced_grad_sq(s, n);
        let xs_sq = vars::reduced_grad_sq(s / N::from(4.0), half);
        let t = vars::reduced_tau(tau / N::from(2.0), half);
        n * self.0.f(MggaVars {
            rs,
            z: N::from(0.0),
            opz: N::from(1.0),
            omz: N::from(1.0),
            na: half,
            nb: half,
            xt2,
            xs0_sq: xs_sq,
            xs1_sq: xs_sq,
            t0: t,
            t1: t,
        })
    }

    /// Polarized energy density `e = n·f` at one point, generic over `N`. Seed
    /// vector is `[n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b]` (floored/clamped).
    fn energy_pol<N: DualNum<f64> + Copy>(&self, x: &SVector<N, 7>) -> N {
        let na = x[0];
        let nb = x[1];
        let saa = x[2];
        let sab = x[3];
        let sbb = x[4];
        let ta = x[5];
        let tb = x[6];
        let n = na + nb;
        let rs = vars::rs_from_n(n);
        let z = (na - nb) / n;
        let opz = (na + na) / n; // 1 + z, cancellation-free
        let omz = (nb + nb) / n; // 1 − z, cancellation-free
                                 // σ_tot = σ_aa + 2σ_ab + σ_bb = |∇n_a + ∇n_b|² is mathematically ≥ 0. The
                                 // σ_ab clamp drives it to exactly 0, where f64 cancellation can leave a tiny
                                 // *negative* noise residual; divided by n^(8/3) at very low density that
                                 // becomes a large spurious negative reduced gradient, which is NaN-producing
                                 // for log1p-domain consumers (r2SCAN-c's `log1p(4·(y−dy))`). Floor at the
                                 // exact lower bound 0 — branch on the real part like the other harness
                                 // clamps (no chain through the max), so the seeded derivative is unaffected
                                 // in-domain (where σ_tot > 0 and the floor is a no-op). This keeps the energy
                                 // finite out-of-domain where libxc itself returns NaN (a divergence-C-class
                                 // robustness asymmetry; the reduced gradient there, s ≫ 10³, is non-physical).
        let sigma_tot_raw = saa + sab + sab + sbb;
        let sigma_tot = if sigma_tot_raw.re() < 0.0 {
            N::from(0.0)
        } else {
            sigma_tot_raw
        };
        let xt2 = vars::reduced_grad_sq(sigma_tot, n);
        let xs0_sq = vars::reduced_grad_sq(saa, na);
        let xs1_sq = vars::reduced_grad_sq(sbb, nb);
        let t0 = vars::reduced_tau(ta, na);
        let t1 = vars::reduced_tau(tb, nb);
        n * self.0.f(MggaVars {
            rs,
            z,
            opz,
            omz,
            na,
            nb,
            xt2,
            xs0_sq,
            xs1_sq,
            t0,
            t1,
        })
    }

    /// Floor and clamp the polarized inputs exactly as libxc's `work_mgga` does,
    /// returning the seed vector `[n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b]` and the
    /// floored total density `n_f`. Shared by the vxc and fxc harnesses so both
    /// seed at identical points.
    #[allow(clippy::too_many_arguments)]
    fn seed_pol(
        &self,
        na: f64,
        nb: f64,
        saa: f64,
        sab: f64,
        sbb: f64,
        ta: f64,
        tb: f64,
    ) -> (SVector<f64, 7>, f64) {
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let na_f = na.max(thr);
        let nb_f = nb.max(thr);
        let ta_f = ta.max(TAU_THRESHOLD);
        let tb_f = tb.max(TAU_THRESHOLD);
        // Floor σ_aa/σ_bb, then apply the Fermi-hole-curvature clamp per spin
        // (libxc work_mgga: σ_σσ ← min(σ_σσ, 8 n_σ τ_σ)) — before s_ave, so the
        // σ_ab clamp window is built from the FHC-clamped diagonal σ.
        let saa_f = fhc_clamp(saa.max(sfloor), na_f, ta_f);
        let sbb_f = fhc_clamp(sbb.max(sfloor), nb_f, tb_f);
        let s_ave = 0.5 * (saa_f + sbb_f);
        let sab = if sab >= -s_ave { sab } else { -s_ave };
        let sab_c = if sab <= s_ave { sab } else { s_ave };
        (
            SVector::<f64, 7>::from([na_f, nb_f, saa_f, sab_c, sbb_f, ta_f, tb_f]),
            na_f + nb_f,
        )
    }

    fn eval_unpol(
        &self,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Result<XcResult, XcError> {
        check_len(rho, np)?;
        check_len(sigma, np)?;
        check_len(tau, np)?;
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; np];
        let mut vsigma = vec![0.0; np];
        let mut vtau = vec![0.0; np];
        for i in 0..np {
            let n = rho[i];
            if n < thr || n.is_nan() {
                continue;
            }
            let nf = n.max(thr);
            let tf = tau[i].max(TAU_THRESHOLD);
            // Floor σ, then apply the Fermi-hole-curvature clamp (libxc work_mgga).
            let sf = fhc_clamp(sigma[i].max(sfloor), nf, tf);
            let (e, g) = gradient(
                |v: SVector<DualSVec64<3>, 3>| self.energy_unpol(&v),
                &SVector::<f64, 3>::from([nf, sf, tf]),
            );
            exc[i] = e / nf;
            vrho[i] = g[0];
            vsigma[i] = g[1];
            vtau[i] = g[2];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            vtau,
            ..Default::default()
        })
    }

    fn eval_pol(
        &self,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Result<XcResult, XcError> {
        check_len(rho, 2 * np)?;
        check_len(sigma, 3 * np)?;
        check_len(tau, 2 * np)?;
        let thr = self.0.info().dens_threshold;
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; 2 * np];
        let mut vsigma = vec![0.0; 3 * np];
        let mut vtau = vec![0.0; 2 * np];
        for i in 0..np {
            let na = rho[2 * i];
            let nb = rho[2 * i + 1];
            let n = na + nb;
            if n < thr || n.is_nan() {
                continue;
            }
            let (seed, n_f) = self.seed_pol(
                na,
                nb,
                sigma[3 * i],
                sigma[3 * i + 1],
                sigma[3 * i + 2],
                tau[2 * i],
                tau[2 * i + 1],
            );
            let (e, g) = gradient(|v: SVector<DualSVec64<7>, 7>| self.energy_pol(&v), &seed);
            exc[i] = e / n_f;
            vrho[2 * i] = g[0];
            vrho[2 * i + 1] = g[1];
            vsigma[3 * i] = g[2];
            vsigma[3 * i + 1] = g[3];
            vsigma[3 * i + 2] = g[4];
            vtau[2 * i] = g[5];
            vtau[2 * i + 1] = g[6];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            vtau,
            ..Default::default()
        })
    }

    /// Second-order (`fxc`) unpolarized harness. The 3×3 Hessian over `[n, σ, τ]`
    /// maps to the six unpolarized meta-GGA second derivatives: `v2rho2 = h(0,0)`,
    /// `v2rhosigma = h(0,1)`, `v2sigma2 = h(1,1)`, `v2rhotau = h(0,2)`,
    /// `v2sigmatau = h(1,2)`, `v2tau2 = h(2,2)`.
    fn eval_fxc_unpol(
        &self,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Result<XcResult, XcError> {
        check_len(rho, np)?;
        check_len(sigma, np)?;
        check_len(tau, np)?;
        let thr = self.0.info().dens_threshold;
        let sfloor = self.sigma_floor();
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; np];
        let mut vsigma = vec![0.0; np];
        let mut vtau = vec![0.0; np];
        let mut v2rho2 = vec![0.0; np];
        let mut v2rhosigma = vec![0.0; np];
        let mut v2sigma2 = vec![0.0; np];
        let mut v2rhotau = vec![0.0; np];
        let mut v2sigmatau = vec![0.0; np];
        let mut v2tau2 = vec![0.0; np];
        for i in 0..np {
            let n = rho[i];
            if n < thr || n.is_nan() {
                continue;
            }
            let nf = n.max(thr);
            let tf = tau[i].max(TAU_THRESHOLD);
            // Floor σ, then apply the Fermi-hole-curvature clamp (libxc work_mgga).
            let sf = fhc_clamp(sigma[i].max(sfloor), nf, tf);
            let (e, g, h) = hessian(
                |v: SVector<Dual2SVec64<3>, 3>| self.energy_unpol(&v),
                &SVector::<f64, 3>::from([nf, sf, tf]),
            );
            exc[i] = e / nf;
            vrho[i] = g[0];
            vsigma[i] = g[1];
            vtau[i] = g[2];
            v2rho2[i] = h[(0, 0)];
            v2rhosigma[i] = h[(0, 1)];
            v2sigma2[i] = h[(1, 1)];
            v2rhotau[i] = h[(0, 2)];
            v2sigmatau[i] = h[(1, 2)];
            v2tau2[i] = h[(2, 2)];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            vtau,
            v2rho2,
            v2rhosigma,
            v2sigma2,
            v2rhotau,
            v2sigmatau,
            v2tau2,
            ..Default::default()
        })
    }

    /// Second-order (`fxc`) polarized harness. The 7×7 Hessian over
    /// `[n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b]` (indices 0..6) is packed into
    /// libxc's xc.h ordering: `v2rho2 = [aa, ab, bb]`; `v2rhosigma = [a·aa, a·ab,
    /// a·bb, b·aa, b·ab, b·bb]` (ρ major); `v2sigma2 = [aa·aa, aa·ab, aa·bb, ab·ab,
    /// ab·bb, bb·bb]` (symmetric); `v2rhotau = [a·τa, a·τb, b·τa, b·τb]` (ρ major ×
    /// τ minor); `v2sigmatau = [aa·τa, aa·τb, ab·τa, ab·τb, bb·τa, bb·τb]` (σ major ×
    /// τ minor); `v2tau2 = [τa·τa, τa·τb, τb·τb]` (symmetric).
    fn eval_fxc_pol(
        &self,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
    ) -> Result<XcResult, XcError> {
        check_len(rho, 2 * np)?;
        check_len(sigma, 3 * np)?;
        check_len(tau, 2 * np)?;
        let thr = self.0.info().dens_threshold;
        let mut exc = vec![0.0; np];
        let mut vrho = vec![0.0; 2 * np];
        let mut vsigma = vec![0.0; 3 * np];
        let mut vtau = vec![0.0; 2 * np];
        let mut v2rho2 = vec![0.0; 3 * np];
        let mut v2rhosigma = vec![0.0; 6 * np];
        let mut v2sigma2 = vec![0.0; 6 * np];
        let mut v2rhotau = vec![0.0; 4 * np];
        let mut v2sigmatau = vec![0.0; 6 * np];
        let mut v2tau2 = vec![0.0; 3 * np];
        for i in 0..np {
            let na = rho[2 * i];
            let nb = rho[2 * i + 1];
            let n = na + nb;
            if n < thr || n.is_nan() {
                continue;
            }
            let (seed, n_f) = self.seed_pol(
                na,
                nb,
                sigma[3 * i],
                sigma[3 * i + 1],
                sigma[3 * i + 2],
                tau[2 * i],
                tau[2 * i + 1],
            );
            let (e, g, h) = hessian(|v: SVector<Dual2SVec64<7>, 7>| self.energy_pol(&v), &seed);
            exc[i] = e / n_f;
            vrho[2 * i] = g[0];
            vrho[2 * i + 1] = g[1];
            vsigma[3 * i] = g[2];
            vsigma[3 * i + 1] = g[3];
            vsigma[3 * i + 2] = g[4];
            vtau[2 * i] = g[5];
            vtau[2 * i + 1] = g[6];
            // density-density block (0=n_a, 1=n_b)
            v2rho2[3 * i] = h[(0, 0)];
            v2rho2[3 * i + 1] = h[(0, 1)];
            v2rho2[3 * i + 2] = h[(1, 1)];
            // density-sigma block (ρ major: a×{aa,ab,bb}, b×{aa,ab,bb}; σ 2,3,4)
            v2rhosigma[6 * i] = h[(0, 2)];
            v2rhosigma[6 * i + 1] = h[(0, 3)];
            v2rhosigma[6 * i + 2] = h[(0, 4)];
            v2rhosigma[6 * i + 3] = h[(1, 2)];
            v2rhosigma[6 * i + 4] = h[(1, 3)];
            v2rhosigma[6 * i + 5] = h[(1, 4)];
            // sigma-sigma block (symmetric upper triangle over {aa,ab,bb})
            v2sigma2[6 * i] = h[(2, 2)];
            v2sigma2[6 * i + 1] = h[(2, 3)];
            v2sigma2[6 * i + 2] = h[(2, 4)];
            v2sigma2[6 * i + 3] = h[(3, 3)];
            v2sigma2[6 * i + 4] = h[(3, 4)];
            v2sigma2[6 * i + 5] = h[(4, 4)];
            // density-tau block (ρ major × τ minor; τ 5,6)
            v2rhotau[4 * i] = h[(0, 5)];
            v2rhotau[4 * i + 1] = h[(0, 6)];
            v2rhotau[4 * i + 2] = h[(1, 5)];
            v2rhotau[4 * i + 3] = h[(1, 6)];
            // sigma-tau block (σ major × τ minor)
            v2sigmatau[6 * i] = h[(2, 5)];
            v2sigmatau[6 * i + 1] = h[(2, 6)];
            v2sigmatau[6 * i + 2] = h[(3, 5)];
            v2sigmatau[6 * i + 3] = h[(3, 6)];
            v2sigmatau[6 * i + 4] = h[(4, 5)];
            v2sigmatau[6 * i + 5] = h[(4, 6)];
            // tau-tau block (symmetric upper triangle over {τa,τb})
            v2tau2[3 * i] = h[(5, 5)];
            v2tau2[3 * i + 1] = h[(5, 6)];
            v2tau2[3 * i + 2] = h[(6, 6)];
        }
        Ok(XcResult {
            exc,
            vrho,
            vsigma,
            vtau,
            v2rho2,
            v2rhosigma,
            v2sigma2,
            v2rhotau,
            v2sigmatau,
            v2tau2,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::{Family, FunctionalId, Kind};

    struct DummyMgga(FunctionalInfo);
    impl MggaEnergy for DummyMgga {
        fn info(&self) -> &FunctionalInfo {
            &self.0
        }
        fn f<N: DualNum<f64> + Copy>(&self, v: MggaVars<N>) -> N {
            // Depends on rs, the squared reduced gradients, and the reduced τ so
            // vrho/vsigma/vtau are all nonzero (a smooth harness validator).
            let c = N::from(1e-3);
            -v.rs.recip() + (v.xs0_sq + v.xs1_sq) * c + (v.t0 + v.t1) * c
        }
    }

    fn dummy() -> Mgga<DummyMgga> {
        Mgga(DummyMgga(FunctionalInfo {
            id: Some(FunctionalId::MggaXTpss),
            name: "dummy_mgga",
            family: Family::Mgga,
            kind: Kind::Exchange,
            needs_sigma: true,
            needs_lapl: false,
            needs_tau: true,
            dens_threshold: 1e-15,
            hybrid: None,
        }))
    }

    #[test]
    fn missing_tau_errors() {
        let f = dummy();
        let rho = [0.3];
        let sigma = [0.01];
        let err = f
            .eval(Spin::Unpolarized, 1, &XcInput::gga(&rho, &sigma))
            .unwrap_err();
        assert_eq!(err, XcError::MissingInput("tau"));
    }

    #[test]
    fn unpol_runs_finite() {
        let f = dummy();
        let rho = [0.1, 0.5];
        let sigma = [0.01, 0.2];
        let tau = [0.05, 0.3];
        let out = f
            .eval(
                Spin::Unpolarized,
                2,
                &XcInput::gga(&rho, &sigma).with_tau(&tau),
            )
            .unwrap();
        assert_eq!(out.vtau.len(), 2);
        assert!(out
            .exc
            .iter()
            .chain(&out.vrho)
            .chain(&out.vsigma)
            .chain(&out.vtau)
            .all(|v| v.is_finite()));
    }

    #[test]
    fn pol_runs_finite() {
        let f = dummy();
        let rho = [0.05, 0.05, 0.3, 0.2];
        let sigma = [0.01, 0.005, 0.01, 0.2, 0.1, 0.15];
        let tau = [0.02, 0.02, 0.1, 0.08];
        let out = f
            .eval(
                Spin::Polarized,
                2,
                &XcInput::gga(&rho, &sigma).with_tau(&tau),
            )
            .unwrap();
        assert_eq!(out.vrho.len(), 4);
        assert_eq!(out.vsigma.len(), 6);
        assert_eq!(out.vtau.len(), 4);
        assert!(out
            .vrho
            .iter()
            .chain(&out.vsigma)
            .chain(&out.vtau)
            .all(|v| v.is_finite()));
    }

    /// fxc Hessian indexing: a smooth energy makes every second-derivative block
    /// finite, with the documented lengths (both spins).
    #[test]
    fn fxc_blocks_have_right_lengths() {
        let f = dummy();
        let rho = [0.6, 0.3];
        let sigma = [0.1, 0.04, 0.08];
        let tau = [0.2, 0.1];
        let out = f
            .eval_fxc(
                Spin::Polarized,
                1,
                &XcInput::gga(&rho, &sigma).with_tau(&tau),
            )
            .unwrap();
        assert_eq!(out.v2rho2.len(), 3);
        assert_eq!(out.v2rhosigma.len(), 6);
        assert_eq!(out.v2sigma2.len(), 6);
        assert_eq!(out.v2rhotau.len(), 4);
        assert_eq!(out.v2sigmatau.len(), 6);
        assert_eq!(out.v2tau2.len(), 3);
        assert!(out
            .v2rho2
            .iter()
            .chain(&out.v2rhosigma)
            .chain(&out.v2sigma2)
            .chain(&out.v2rhotau)
            .chain(&out.v2sigmatau)
            .chain(&out.v2tau2)
            .all(|v| v.is_finite()));
    }
}
