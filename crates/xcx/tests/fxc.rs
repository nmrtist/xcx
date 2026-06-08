// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Second-derivative (`fxc`) correctness gate for the public `xcx` API.
//!
//! These tests are **libxc-free** (public surface only, like `fuzz.rs`), so they
//! ship with the crate. They establish that `eval_fxc`'s second derivatives are
//! self-consistent with the (golden-verified) first derivatives, independent of
//! any libxc cross-check:
//!
//! - **Finite differences:** every `v2*` component equals a central FD of the
//!   corresponding first derivative, at smooth interior points, both spins. This
//!   is the AD-vs-FD check the golden suite cannot give (golden pins to libxc; FD
//!   pins to xcx's own first derivatives).
//! - **Symmetry of mixed partials:** the FD is taken in *one* direction per pair;
//!   the Hessian is symmetric by construction, so e.g. `v2rhosigma` (= ∂²/∂n∂σ)
//!   FD-checks against both `∂vrho/∂σ` and `∂vsigma/∂n`.
//! - **unpol/pol consistency at z = 0**, **exchange has identically-zero σ_ab
//!   derivatives**, and **spin-swap symmetry** of `v2rho2`/`v2sigma2`.

use xcx::{Functional, FunctionalId, Spin, XcInput};

/// All functionals that need σ (GGA + hybrids) — the ones with a full `fxc`
/// tensor (`v2rho2` + `v2rhosigma` + `v2sigma2`).
const GGA_LIKE: &[FunctionalId] = &[
    FunctionalId::GgaXPbe,
    FunctionalId::GgaXB88,
    FunctionalId::GgaCPbe,
    FunctionalId::GgaCLyp,
    FunctionalId::GgaXPbeR,
    FunctionalId::GgaXPbeSol,
    FunctionalId::GgaXRpbe,
    FunctionalId::GgaCPbeSol,
    FunctionalId::HybGgaXcPbeh,
    FunctionalId::HybGgaXcB3lyp,
    FunctionalId::HybGgaXcB3lyp5,
];

/// Pure-exchange GGAs: their energy has no σ_ab dependence at all, so every
/// derivative w.r.t. σ_ab must be *identically* zero.
const PURE_X_GGA: &[FunctionalId] = &[
    FunctionalId::GgaXPbe,
    FunctionalId::GgaXB88,
    FunctionalId::GgaXPbeR,
    FunctionalId::GgaXPbeSol,
    FunctionalId::GgaXRpbe,
];

const LDA_ALL: &[FunctionalId] = &[
    FunctionalId::LdaX,
    FunctionalId::LdaCPw,
    FunctionalId::LdaCVwn,
    FunctionalId::LdaCVwn3,
    FunctionalId::LdaCVwnRpa,
];

/// meta-GGAs: the full 6-block `fxc` tensor (`v2rho2`/`v2rhosigma`/`v2sigma2`
/// plus the τ blocks `v2rhotau`/`v2sigmatau`/`v2tau2`).
const MGGA_ALL: &[FunctionalId] = &[
    FunctionalId::MggaXTpss,
    FunctionalId::MggaCTpss,
    FunctionalId::MggaXR2scan,
    FunctionalId::MggaCR2scan,
    FunctionalId::MggaXM06L,
    FunctionalId::MggaCM06L,
];

/// Mixed relative/absolute closeness for FD comparisons (central FD on a first
/// derivative: truncation O(h²) plus subtractive roundoff O(ε/h)).
fn close(ad: f64, fd: f64, rtol: f64, atol: f64) -> bool {
    (ad - fd).abs() <= rtol * ad.abs().max(fd.abs()) + atol
}

// ---------------------------------------------------------------------------
// Finite-difference checks: v2* vs central FD of the first derivatives.
// ---------------------------------------------------------------------------

/// Unpolarized GGA: 2×2 Hessian over `[n, σ]`.
fn fd_unpol_gga(id: FunctionalId) {
    let f = Functional::new(id, Spin::Unpolarized).unwrap();
    // Smooth interior points (away from screening, σ = 0, and huge gradients);
    // `(1.0, 0.02)` reaches toward the small-σ band the sqrt-free fix repaired.
    for &(n, s) in &[
        (0.5, 0.1),
        (1.0, 0.3),
        (2.0, 1.5),
        (0.3, 0.05),
        (5.0, 8.0),
        (1.0, 0.02),
    ] {
        let r = f.eval_fxc(1, &XcInput::gga(&[n], &[s])).unwrap();
        let vrho = |n: f64, s: f64| f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().vrho[0];
        let vsig = |n: f64, s: f64| f.eval(1, &XcInput::gga(&[n], &[s])).unwrap().vsigma[0];
        let hn = 1e-6 * n;
        let hs = 1e-6 * s;
        let fd_v2rho2 = (vrho(n + hn, s) - vrho(n - hn, s)) / (2.0 * hn);
        // mixed partial ∂²e/∂n∂σ from both directions (Hessian symmetry)
        let fd_rs_a = (vsig(n + hn, s) - vsig(n - hn, s)) / (2.0 * hn); // ∂vsigma/∂n
        let fd_rs_b = (vrho(n, s + hs) - vrho(n, s - hs)) / (2.0 * hs); // ∂vrho/∂σ
        let fd_v2sigma2 = (vsig(n, s + hs) - vsig(n, s - hs)) / (2.0 * hs);
        let tag = f.info().name;
        assert!(
            close(r.v2rho2[0], fd_v2rho2, 1e-5, 1e-7),
            "{tag} unpol v2rho2 @({n},{s}): ad {} vs fd {fd_v2rho2}",
            r.v2rho2[0]
        );
        assert!(
            close(r.v2rhosigma[0], fd_rs_a, 1e-5, 1e-7)
                && close(r.v2rhosigma[0], fd_rs_b, 1e-5, 1e-7),
            "{tag} unpol v2rhosigma @({n},{s}): ad {} vs fd {fd_rs_a}/{fd_rs_b}",
            r.v2rhosigma[0]
        );
        assert!(
            close(r.v2sigma2[0], fd_v2sigma2, 1e-5, 1e-7),
            "{tag} unpol v2sigma2 @({n},{s}): ad {} vs fd {fd_v2sigma2}",
            r.v2sigma2[0]
        );
    }
}

/// Polarized GGA: the full 5×5 Hessian over `[n_a, n_b, σ_aa, σ_ab, σ_bb]`. The
/// packed `fxc` is checked against a central FD of the first-derivative
/// 5-gradient `g = [vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb]`.
fn fd_pol_gga(id: FunctionalId) {
    let f = Functional::new(id, Spin::Polarized).unwrap();
    // (n_a, n_b, σ_aa, σ_ab, σ_bb), interior and physical (|σ_ab| < √(σaa·σbb)).
    let points: &[[f64; 5]] = &[
        [0.6, 0.3, 0.10, 0.04, 0.08],
        [1.0, 0.7, 0.30, 0.10, 0.20],
        [2.0, 1.5, 1.00, 0.40, 0.80],
        [0.4, 0.5, 0.05, -0.02, 0.06],
    ];
    // g[i] = the packed first-derivative gradient (vrho ++ vsigma).
    let grad = |x: &[f64; 5]| {
        let r = f
            .eval(1, &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]))
            .unwrap();
        [r.vrho[0], r.vrho[1], r.vsigma[0], r.vsigma[1], r.vsigma[2]]
    };
    // central FD of g[i] w.r.t. variable j.
    let fd = |x: &[f64; 5], i: usize, j: usize| {
        let h = 1e-6 * x[j].abs().max(1e-3);
        let mut xp = *x;
        let mut xm = *x;
        xp[j] += h;
        xm[j] -= h;
        (grad(&xp)[i] - grad(&xm)[i]) / (2.0 * h)
    };
    let tag = f.info().name;
    for x in points {
        let r = f
            .eval_fxc(1, &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]))
            .unwrap();
        // packed component -> (gradient index i, variable index j)
        let v2rho2 = [(0usize, 0usize), (0, 1), (1, 1)];
        let v2rhosigma = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
        let v2sigma2 = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)];
        for (k, &(i, j)) in v2rho2.iter().enumerate() {
            let f_fd = fd(x, i, j);
            assert!(
                close(r.v2rho2[k], f_fd, 2e-5, 1e-7),
                "{tag} pol v2rho2[{k}] @{x:?}: ad {} vs fd {f_fd}",
                r.v2rho2[k]
            );
        }
        for (k, &(i, j)) in v2rhosigma.iter().enumerate() {
            let f_fd = fd(x, i, j);
            assert!(
                close(r.v2rhosigma[k], f_fd, 2e-5, 1e-7),
                "{tag} pol v2rhosigma[{k}] @{x:?}: ad {} vs fd {f_fd}",
                r.v2rhosigma[k]
            );
        }
        for (k, &(i, j)) in v2sigma2.iter().enumerate() {
            let f_fd = fd(x, i, j);
            assert!(
                close(r.v2sigma2[k], f_fd, 2e-5, 1e-7),
                "{tag} pol v2sigma2[{k}] @{x:?}: ad {} vs fd {f_fd}",
                r.v2sigma2[k]
            );
        }
    }
}

/// Unpolarized LDA: 1×1 Hessian (just `v2rho2 = ∂²e/∂n²`).
fn fd_unpol_lda(id: FunctionalId) {
    let f = Functional::new(id, Spin::Unpolarized).unwrap();
    for &n in &[0.05, 0.5, 1.0, 3.0, 50.0] {
        let r = f.eval_fxc(1, &XcInput::lda(&[n])).unwrap();
        let vrho = |n: f64| f.eval(1, &XcInput::lda(&[n])).unwrap().vrho[0];
        let h = 1e-6 * n;
        let fd = (vrho(n + h) - vrho(n - h)) / (2.0 * h);
        assert!(
            close(r.v2rho2[0], fd, 1e-5, 1e-8),
            "{} unpol v2rho2 @{n}: ad {} vs fd {fd}",
            f.info().name,
            r.v2rho2[0]
        );
    }
}

/// Polarized LDA: 2×2 density Hessian packed `[aa, ab, bb]`.
fn fd_pol_lda(id: FunctionalId) {
    let f = Functional::new(id, Spin::Polarized).unwrap();
    let points: &[[f64; 2]] = &[[0.6, 0.3], [1.0, 0.7], [2.0, 1.5], [0.4, 0.5]];
    let grad = |x: &[f64; 2]| {
        let r = f.eval(1, &XcInput::lda(&[x[0], x[1]])).unwrap();
        [r.vrho[0], r.vrho[1]]
    };
    let fd = |x: &[f64; 2], i: usize, j: usize| {
        let h = 1e-6 * x[j];
        let mut xp = *x;
        let mut xm = *x;
        xp[j] += h;
        xm[j] -= h;
        (grad(&xp)[i] - grad(&xm)[i]) / (2.0 * h)
    };
    let tag = f.info().name;
    for x in points {
        let r = f.eval_fxc(1, &XcInput::lda(&[x[0], x[1]])).unwrap();
        for (k, (i, j)) in [(0usize, 0usize), (0, 1), (1, 1)].into_iter().enumerate() {
            let f_fd = fd(x, i, j);
            assert!(
                close(r.v2rho2[k], f_fd, 1e-5, 1e-8),
                "{tag} pol v2rho2[{k}] @{x:?}: ad {} vs fd {f_fd}",
                r.v2rho2[k]
            );
        }
    }
}

/// Unpolarized meta-GGA: 3×3 Hessian over `[n, σ, τ]`. Every `v2*` block (incl.
/// the τ blocks) is FD-checked against the corresponding first derivative, with
/// the mixed partials cross-checked from both directions (Hessian symmetry).
/// Points are smooth physical interiors that include the τ-ratio hazards α ≈ 0
/// (τ ≈ τ_W) and α ≈ 1.
fn fd_unpol_mgga(id: FunctionalId) {
    let f = Functional::new(id, Spin::Unpolarized).unwrap();
    let g = |n: f64, s: f64, t: f64| {
        let r = f.eval(1, &XcInput::gga(&[n], &[s]).with_tau(&[t])).unwrap();
        [r.vrho[0], r.vsigma[0], r.vtau[0]]
    };
    let fd = |n: f64, s: f64, t: f64, i: usize, var: usize| {
        let (mut p, mut m) = ([n, s, t], [n, s, t]);
        let h = 1e-6 * [n, s, t][var].abs().max(1e-3);
        p[var] += h;
        m[var] -= h;
        (g(p[0], p[1], p[2])[i] - g(m[0], m[1], m[2])[i]) / (2.0 * h)
    };
    let tag = f.info().name;
    for &(n, s, t) in &[
        (0.5, 0.1, 0.3),
        (1.0, 0.3, 0.8),
        (2.0, 1.5, 3.0),
        (1.0, 0.4, 0.06), // α ≈ 0 (τ ≈ τ_W)
        (1.0, 0.4, 4.6),  // α ≈ 1
        (3.0, 2.0, 9.0),
    ] {
        let r = f
            .eval_fxc(1, &XcInput::gga(&[n], &[s]).with_tau(&[t]))
            .unwrap();
        // block -> (gradient component i, seed variable var). 0=n,1=σ,2=τ.
        for (val, i, var, name) in [
            (r.v2rho2[0], 0usize, 0usize, "v2rho2"),
            (r.v2rhosigma[0], 0, 1, "v2rhosigma"),
            (r.v2sigma2[0], 1, 1, "v2sigma2"),
            (r.v2rhotau[0], 0, 2, "v2rhotau"),
            (r.v2sigmatau[0], 1, 2, "v2sigmatau"),
            (r.v2tau2[0], 2, 2, "v2tau2"),
        ] {
            let f_fd = fd(n, s, t, i, var);
            assert!(
                close(val, f_fd, 1e-5, 1e-7),
                "{tag} unpol {name} @({n},{s},{t}): ad {val} vs fd {f_fd}"
            );
        }
    }
}

/// Polarized meta-GGA: the full 7×7 Hessian over `[n_a, n_b, σ_aa, σ_ab, σ_bb,
/// τ_a, τ_b]`, packed `fxc` vs central FD of the 7-component first-derivative
/// gradient `g = [vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb, vtau_a, vtau_b]`.
fn fd_pol_mgga(id: FunctionalId) {
    let f = Functional::new(id, Spin::Polarized).unwrap();
    // (n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b), interior and physical.
    let points: &[[f64; 7]] = &[
        [0.6, 0.3, 0.10, 0.04, 0.08, 0.40, 0.25],
        [1.0, 0.7, 0.30, 0.10, 0.20, 1.20, 0.90],
        [2.0, 1.5, 1.00, 0.40, 0.80, 3.00, 2.00],
        [0.9, 0.6, 0.20, 0.05, 0.15, 0.70, 0.50],
    ];
    let grad = |x: &[f64; 7]| {
        let r = f
            .eval(
                1,
                &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]).with_tau(&[x[5], x[6]]),
            )
            .unwrap();
        [
            r.vrho[0],
            r.vrho[1],
            r.vsigma[0],
            r.vsigma[1],
            r.vsigma[2],
            r.vtau[0],
            r.vtau[1],
        ]
    };
    let fd = |x: &[f64; 7], i: usize, j: usize| {
        let h = 1e-6 * x[j].abs().max(1e-3);
        let (mut xp, mut xm) = (*x, *x);
        xp[j] += h;
        xm[j] -= h;
        (grad(&xp)[i] - grad(&xm)[i]) / (2.0 * h)
    };
    let tag = f.info().name;
    // packed block component -> (gradient index i, variable index j).
    let v2rho2 = [(0usize, 0usize), (0, 1), (1, 1)];
    let v2rhosigma = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
    let v2sigma2 = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)];
    let v2rhotau = [(0, 5), (0, 6), (1, 5), (1, 6)];
    let v2sigmatau = [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6)];
    let v2tau2 = [(5, 5), (5, 6), (6, 6)];
    for x in points {
        let r = f
            .eval_fxc(
                1,
                &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]).with_tau(&[x[5], x[6]]),
            )
            .unwrap();
        #[allow(clippy::type_complexity)]
        let blocks: [(&str, &[f64], &[(usize, usize)]); 6] = [
            ("v2rho2", &r.v2rho2, &v2rho2),
            ("v2rhosigma", &r.v2rhosigma, &v2rhosigma),
            ("v2sigma2", &r.v2sigma2, &v2sigma2),
            ("v2rhotau", &r.v2rhotau, &v2rhotau),
            ("v2sigmatau", &r.v2sigmatau, &v2sigmatau),
            ("v2tau2", &r.v2tau2, &v2tau2),
        ];
        for (name, got, idx) in blocks {
            for (k, &(i, j)) in idx.iter().enumerate() {
                let f_fd = fd(x, i, j);
                assert!(
                    close(got[k], f_fd, 2e-5, 1e-7),
                    "{tag} pol {name}[{k}] @{x:?}: ad {} vs fd {f_fd}",
                    got[k]
                );
            }
        }
    }
}

#[test]
fn fxc_matches_finite_difference() {
    for &id in LDA_ALL {
        fd_unpol_lda(id);
        fd_pol_lda(id);
    }
    for &id in GGA_LIKE {
        fd_unpol_gga(id);
        fd_pol_gga(id);
    }
    for &id in MGGA_ALL {
        fd_unpol_mgga(id);
        fd_pol_mgga(id);
    }
}

/// A meta-GGA's `v2sigma2` must stay finite and ~constant (approaching its σ → 0
/// limit) as σ → 0 at fixed τ — not blow up through a residual `√`/`t`-ratio second
/// derivative. The libxc-free regression lock for the AD-safe sqrt-free reduced
/// gradient: a naive `√σ`-based form would diverge `~ σ^(-3/2)` here. Covers
/// TPSS-x (the `√(½(9/25 z² + p²))` factoring, the meta-GGA analogue of
/// divergence #4) and both r2SCAN functionals (the `scan_gx` damping + smooth
/// rational forms). Pinned against the σ = 1e-4 value, where the AD is
/// unquestionably accurate (golden pins the *value* to libxc — at σ ≥ 1e-5 for
/// TPSS-x, down to exact σ = 0 for r2SCAN).
#[test]
fn mgga_v2sigma2_stable_into_small_sigma() {
    let small = [1e-5, 1e-7, 1e-9, 0.0];
    for &id in &[
        FunctionalId::MggaXTpss,
        FunctionalId::MggaXR2scan,
        FunctionalId::MggaCR2scan,
        FunctionalId::MggaXM06L,
        FunctionalId::MggaCM06L,
    ] {
        let f = Functional::new(id, Spin::Unpolarized).unwrap();
        let tag = f.info().name;
        for &(n, tau) in &[(0.3, 0.2), (1.0, 0.8), (10.0, 30.0)] {
            let refv = f
                .eval_fxc(1, &XcInput::gga(&[n], &[1e-4]).with_tau(&[tau]))
                .unwrap()
                .v2sigma2[0];
            for &s in &small {
                let v = f
                    .eval_fxc(1, &XcInput::gga(&[n], &[s]).with_tau(&[tau]))
                    .unwrap()
                    .v2sigma2[0];
                assert!(
                    v.is_finite(),
                    "{tag} unpol v2sigma2 non-finite @(n={n}, σ={s})"
                );
                assert!(
                    close(v, refv, 1e-2, 1e-10),
                    "{tag} unpol v2sigma2 @(n={n}, σ={s}) = {v:e} drifted from σ=1e-4 ref {refv:e}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small-σ stability: the sqrt-free per-spin reduced gradient (old divergence #4).
// ---------------------------------------------------------------------------

/// `v2sigma2` must stay **finite and ~constant** (approaching its LDA-limit
/// value) as σ → 0 — not blow up `~σ^(-3/2)` through a `√σ` second derivative.
/// This is the libxc-free regression lock for the sqrt-free per-spin reduced
/// gradient: the harness now carries every reduced gradient squared, so the
/// second derivative is accurate down to σ = 0 (old divergence #4). A reverted,
/// `√σ`-based harness gives ~1e21 at the floor and fails here by ~24 orders. The
/// golden suite pins the *value* to libxc at σ = 1e-6/1e-8; this pins it (libxc-
/// free) against its own σ = 1e-4 value, where the AD is unquestionably accurate.
#[test]
fn v2sigma2_stable_into_small_sigma() {
    let small = [1e-6, 1e-8, 1e-10, 0.0];
    // unpolarized: the single v2sigma2 component.
    for &id in GGA_LIKE {
        let f = Functional::new(id, Spin::Unpolarized).unwrap();
        let tag = f.info().name;
        for &n in &[0.3, 1.0, 10.0] {
            let refv = f
                .eval_fxc(1, &XcInput::gga(&[n], &[1e-4]))
                .unwrap()
                .v2sigma2[0];
            for &s in &small {
                let v = f.eval_fxc(1, &XcInput::gga(&[n], &[s])).unwrap().v2sigma2[0];
                assert!(
                    v.is_finite(),
                    "{tag} unpol v2sigma2 non-finite @(n={n}, σ={s})"
                );
                assert!(
                    close(v, refv, 1e-2, 1e-10),
                    "{tag} unpol v2sigma2 @(n={n}, σ={s}) = {v:e} drifted from σ=1e-4 ref {refv:e}"
                );
            }
        }
    }
    // polarized: the per-spin self-terms aa·aa (idx 0) and bb·bb (idx 5), with
    // σ_aa = σ_bb = s, σ_ab = 0 (drives both per-spin reduced gradients → 0).
    for &id in GGA_LIKE {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let tag = f.info().name;
        for &(na, nb) in &[(0.6, 0.4), (2.0, 1.0)] {
            let v2s = |s: f64| {
                f.eval_fxc(1, &XcInput::gga(&[na, nb], &[s, 0.0, s]))
                    .unwrap()
                    .v2sigma2
            };
            let refv = v2s(1e-4);
            for &s in &small {
                let v = v2s(s);
                for &k in &[0usize, 5] {
                    assert!(
                        v[k].is_finite(),
                        "{tag} pol v2sigma2[{k}] non-finite @(na={na}, nb={nb}, σ={s})"
                    );
                    assert!(
                        close(v[k], refv[k], 1e-2, 1e-10),
                        "{tag} pol v2sigma2[{k}] @(na={na}, nb={nb}, σ={s}) = {:e} vs ref {:e}",
                        v[k],
                        refv[k]
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// unpol/pol consistency at z = 0.
// ---------------------------------------------------------------------------

/// At zero polarization the polarized and unpolarized evaluations describe the
/// same physical point, so the second derivatives must be related by the chain
/// rule `n = n_a + n_b` (with `n_a = n_b = n/2`):
/// `d²e/dn² = ¼(e_aa + 2 e_ab + e_bb) = ½(v2rho2_aa + v2rho2_ab)` at z = 0
/// (using `e_aa = e_bb`). First derivatives must also agree.
#[test]
fn unpol_pol_consistent_at_zero_polarization() {
    let check = |id: FunctionalId, n: f64, s: f64, has_sigma: bool| {
        let up = Functional::new(id, Spin::Unpolarized).unwrap();
        let po = Functional::new(id, Spin::Polarized).unwrap();
        let (ru, rp) = if has_sigma {
            (
                up.eval_fxc(1, &XcInput::gga(&[n], &[s])).unwrap(),
                po.eval_fxc(
                    1,
                    &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0]),
                )
                .unwrap(),
            )
        } else {
            (
                up.eval_fxc(1, &XcInput::lda(&[n])).unwrap(),
                po.eval_fxc(1, &XcInput::lda(&[n / 2.0, n / 2.0])).unwrap(),
            )
        };
        let tag = up.info().name;
        // first derivatives agree at z = 0
        assert!(
            close(ru.exc[0], rp.exc[0], 1e-11, 1e-13),
            "{tag} exc consistency"
        );
        assert!(
            close(ru.vrho[0], rp.vrho[0], 1e-10, 1e-12)
                && close(ru.vrho[0], rp.vrho[1], 1e-10, 1e-12),
            "{tag} vrho consistency"
        );
        // density-density second derivative: unpol = ½(aa + ab)
        let want = 0.5 * (rp.v2rho2[0] + rp.v2rho2[1]);
        assert!(
            close(ru.v2rho2[0], want, 1e-9, 1e-12),
            "{tag} v2rho2 consistency: unpol {} vs ½(aa+ab) {want}",
            ru.v2rho2[0]
        );
        // and the polarized density Hessian is spin-symmetric (aa == bb) at z = 0
        assert!(
            close(rp.v2rho2[0], rp.v2rho2[2], 1e-10, 1e-12),
            "{tag} v2rho2 aa==bb at z=0"
        );
    };
    for &id in LDA_ALL {
        check(id, 0.7, 0.0, false);
        check(id, 2.5, 0.0, false);
    }
    for &id in GGA_LIKE {
        check(id, 0.8, 0.3, true);
        check(id, 2.0, 1.0, true);
    }
}

// ---------------------------------------------------------------------------
// Pure exchange: identically-zero σ_ab derivatives.
// ---------------------------------------------------------------------------

/// A pure-exchange GGA's energy is a sum of per-spin-channel terms depending only
/// on σ_aa / σ_bb; it never references σ_ab. So *every* derivative w.r.t. σ_ab is
/// exactly 0.0 — `vsigma_ab`, and the `fxc` components `v2rhosigma[a·ab]`,
/// `v2rhosigma[b·ab]`, `v2sigma2[aa·ab]`, `v2sigma2[ab·ab]`, `v2sigma2[ab·bb]`.
#[test]
fn pure_exchange_sigma_ab_derivatives_are_exactly_zero() {
    let points: &[[f64; 5]] = &[
        [0.6, 0.3, 0.10, 0.04, 0.08],
        [1.0, 0.0, 0.30, 0.00, 0.00], // full polarization
        [2.0, 1.5, 1.00, -0.40, 0.80],
    ];
    for &id in PURE_X_GGA {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let tag = f.info().name;
        for x in points {
            let r = f
                .eval_fxc(1, &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]))
                .unwrap();
            assert_eq!(r.vsigma[1], 0.0, "{tag} vsigma_ab @{x:?}");
            // v2rhosigma: a·ab (idx 1), b·ab (idx 4)
            assert_eq!(r.v2rhosigma[1], 0.0, "{tag} v2rhosigma[a·ab] @{x:?}");
            assert_eq!(r.v2rhosigma[4], 0.0, "{tag} v2rhosigma[b·ab] @{x:?}");
            // v2sigma2: aa·ab (1), ab·ab (3), ab·bb (4)
            assert_eq!(r.v2sigma2[1], 0.0, "{tag} v2sigma2[aa·ab] @{x:?}");
            assert_eq!(r.v2sigma2[3], 0.0, "{tag} v2sigma2[ab·ab] @{x:?}");
            assert_eq!(r.v2sigma2[4], 0.0, "{tag} v2sigma2[ab·bb] @{x:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Spin-swap symmetry of v2rho2 / v2sigma2.
// ---------------------------------------------------------------------------

/// Swapping the two spin channels (`n_a↔n_b`, `σ_aa↔σ_bb`, `σ_ab` fixed) must
/// permute the packed second derivatives accordingly. For `v2rho2 = [aa, ab, bb]`
/// the swap is `[bb, ab, aa]`; for `v2sigma2 = [aa·aa, aa·ab, aa·bb, ab·ab,
/// ab·bb, bb·bb]` it is `[bb·bb, ab·bb, aa·bb, ab·ab, aa·ab, aa·aa]`.
#[test]
fn spin_swap_symmetry() {
    let points: &[[f64; 5]] = &[
        [0.6, 0.3, 0.10, 0.04, 0.08],
        [1.2, 0.4, 0.50, -0.20, 0.30],
        [2.0, 1.5, 1.00, 0.40, 0.80],
    ];
    let rt = 1e-10;
    let at = 1e-13;
    // GGA-like: full v2rho2 + v2sigma2 swap.
    for &id in GGA_LIKE {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let tag = f.info().name;
        for x in points {
            let r = f
                .eval_fxc(1, &XcInput::gga(&[x[0], x[1]], &[x[2], x[3], x[4]]))
                .unwrap();
            let rs = f
                .eval_fxc(1, &XcInput::gga(&[x[1], x[0]], &[x[4], x[3], x[2]]))
                .unwrap();
            // v2rho2: swapped[aa]=orig[bb], [ab]=[ab], [bb]=[aa]
            assert!(
                close(rs.v2rho2[0], r.v2rho2[2], rt, at),
                "{tag} v2rho2 swap aa"
            );
            assert!(
                close(rs.v2rho2[1], r.v2rho2[1], rt, at),
                "{tag} v2rho2 swap ab"
            );
            assert!(
                close(rs.v2rho2[2], r.v2rho2[0], rt, at),
                "{tag} v2rho2 swap bb"
            );
            // v2sigma2 permutation [0..5] -> [5,4,2,3,1,0]
            for (a, b) in [(0, 5), (1, 4), (2, 2), (3, 3), (4, 1), (5, 0)] {
                assert!(
                    close(rs.v2sigma2[a], r.v2sigma2[b], rt, at),
                    "{tag} v2sigma2 swap [{a}]<->[{b}] @{x:?}"
                );
            }
        }
    }
    // LDA: v2rho2 only.
    for &id in LDA_ALL {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let tag = f.info().name;
        for x in points {
            let r = f.eval_fxc(1, &XcInput::lda(&[x[0], x[1]])).unwrap();
            let rs = f.eval_fxc(1, &XcInput::lda(&[x[1], x[0]])).unwrap();
            assert!(
                close(rs.v2rho2[0], r.v2rho2[2], rt, at),
                "{tag} lda v2rho2 swap aa"
            );
            assert!(
                close(rs.v2rho2[1], r.v2rho2[1], rt, at),
                "{tag} lda v2rho2 swap ab"
            );
            assert!(
                close(rs.v2rho2[2], r.v2rho2[0], rt, at),
                "{tag} lda v2rho2 swap bb"
            );
        }
    }
}
