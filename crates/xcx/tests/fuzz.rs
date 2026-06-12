// Copyright (c) 2026 Jiekang Tian and the xcx authors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Finiteness / fuzz gate for the public `xcx` API (contract: docs/api-convention.md §4).
//!
//! Contract under test (docs/api-convention.md §4): **every output component is
//! finite for every finite input** — not just `exc`, but each `vrho` channel and
//! all three `vsigma` (aa/ab/bb), and `vtau`/`vlapl` (empty until meta-GGA, but
//! checked anyway). The same finiteness check is extended to the second
//! derivatives (`fxc`: `v2rho2`/`v2rhosigma`/`v2sigma2`) — the NaN-derivative
//! lesson applies doubly to second derivatives. This is orthogonal to the golden
//! suite (which checks "numbers match libxc"); here we only check "no
//! NaN/Inf/panic," over every registered functional (`FunctionalId::ALL`) × both
//! spins.
//!
//! `fxc` finiteness is asserted only **inside the physical reduced-gradient
//! domain** (docs/api-convention.md §4 domain caveat / §8 divergence C): second
//! derivatives overflow f64 at a lower gradient than the first derivatives, and
//! out there neither xcx nor libxc is meaningful, so we do not chase f64-range
//! behavior. The σ range stays at the documented cap (1e12); the per-point domain
//! predicate ([`fxc_in_domain`]) additionally bounds the reduced gradient. The
//! census reports how many `fxc` checks ran vs. were skipped, per region, so the
//! coverage of the boundary regions (full polarization, σ_tot = 0, …) is
//! auditable.
//!
//! This file is **libxc-free** — it uses only the public `xcx` surface (no
//! `xcx-validation`, no FFI) — so it ships with the published crate.
//!
//! Strategy: uniform-random inputs essentially never land on the
//! exact boundaries that have actually produced NaNs in this project (`z = ±1`,
//! `σ_tot = 0`, the `σ_ab` clamp edge), so those regions are **constructed by
//! hand** and densified; uniform random is only a supplement. Every run prints
//! the seed and a per-region coverage census, so a green result is auditable — a
//! fuzz pass that never actually hit `σ_tot = 0` has not tested the thing that
//! broke before. Run with `-- --nocapture` to see the census.
//!
//! On any non-finite output the test panics with the functional id, spin, region,
//! the offending component, and the exact (ρ, σ) that triggered it.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use xcx::{Functional, FunctionalId, Spin, XcInput};

/// Fixed RNG seed — determinism so any failure reproduces bit-for-bit. Arbitrary
/// constant; change only with intent (and re-report coverage).
const SEED: u64 = 0xF02D_C0FF_EE5E_ED42;

/// Uniform-random supplement counts (per spin). These are a *supplement* to the
/// hand-constructed pathological regions, not the main event.
const RANDOM_UNPOL: usize = 2000;
const RANDOM_POL: usize = 3000;

/// Documented physical σ cap for the `fxc` finiteness check (matches the fuzz's
/// `sigma_huge_abs` ceiling and docs/api-convention.md §4). Above this the input
/// is non-physical and `fxc` may overflow (divergence C) — not checked.
const FXC_SIGMA_CAP: f64 = 1e12;

/// Reduced-gradient ceiling for the `fxc` finiteness check. Real densities have
/// `s ≲ 5`; this is an order of magnitude beyond, so the whole physical domain
/// (and the boundary regions that have produced NaNs before) is covered, while
/// the pathological large-gradient corner where second derivatives overflow f64
/// (divergence C, worse for `fxc` than vxc) is excluded by design.
const FXC_REDUCED_GRAD_MAX: f64 = 50.0;

// ---------------------------------------------------------------------------
// Coverage census + the per-point finiteness check.
// ---------------------------------------------------------------------------

/// Tracks how many evaluations touched each named region, plus totals, so the
/// run can prove which pathological regions were actually exercised.
struct Census {
    evals: u64,
    components: u64,
    by_region: BTreeMap<&'static str, u64>,
    /// `fxc` evaluations actually checked (point inside the physical domain).
    fxc_evals: u64,
    /// `fxc` finiteness checks performed.
    fxc_components: u64,
    /// Points skipped for `fxc` (outside the physical reduced-gradient domain).
    fxc_skipped: u64,
    /// Per-region count of `fxc` evaluations checked — proves the boundary
    /// regions (full polarization, σ_tot = 0, …) got second-derivative coverage.
    fxc_by_region: BTreeMap<&'static str, u64>,
}

impl Census {
    fn new() -> Self {
        Self {
            evals: 0,
            components: 0,
            by_region: BTreeMap::new(),
            fxc_evals: 0,
            fxc_components: 0,
            fxc_skipped: 0,
            fxc_by_region: BTreeMap::new(),
        }
    }

    /// Evaluate one point (`np = 1`) and assert every output component finite.
    /// `regions` are the named pathological regions this input belongs to (e.g.
    /// `["full_pol_exact", "sigma_tot_zero_exact"]`); each is credited.
    #[allow(clippy::too_many_arguments)]
    fn check(
        &mut self,
        f: &Functional,
        id: FunctionalId,
        spin: Spin,
        rho: &[f64],
        sigma: Option<&[f64]>,
        tau: Option<&[f64]>,
        regions: &[&'static str],
    ) {
        let input = match (sigma, tau) {
            (Some(s), Some(t)) => XcInput::gga(rho, s).with_tau(t),
            (Some(s), None) => XcInput::gga(rho, s),
            (None, _) => XcInput::lda(rho),
        };
        let r = f.eval(1, &input).unwrap_or_else(|e| {
            panic!(
                "{} (id {}) spin {spin:?}: eval errored on finite input {e:?}: rho=[{}] sigma={} tau={}",
                f.info().name,
                id.as_u32(),
                fmt_slice(rho),
                sigma.map_or("None".to_string(), |s| format!("[{}]", fmt_slice(s))),
                tau.map_or("None".to_string(), |t| format!("[{}]", fmt_slice(t))),
            )
        });

        let comps: [(&str, &[f64]); 5] = [
            ("exc", &r.exc),
            ("vrho", &r.vrho),
            ("vsigma", &r.vsigma),
            ("vtau", &r.vtau),
            ("vlapl", &r.vlapl),
        ];
        for (name, slice) in comps {
            for (k, &v) in slice.iter().enumerate() {
                self.components += 1;
                assert!(
                    v.is_finite(),
                    "NON-FINITE {name}[{k}] = {v} :: {} (id {}) spin {spin:?} region {regions:?}\n  \
                     rho   = [{}]\n  sigma = {}",
                    f.info().name,
                    id.as_u32(),
                    fmt_slice(rho),
                    sigma.map_or("None".to_string(), |s| format!("[{}]", fmt_slice(s))),
                );
            }
        }

        // Second derivatives (fxc), inside the physical reduced-gradient domain.
        if fxc_in_domain(rho, sigma, f.info().dens_threshold) {
            let r = f.eval_fxc(1, &input).unwrap_or_else(|e| {
                panic!(
                    "{} (id {}) spin {spin:?}: eval_fxc errored on finite input {e:?}: \
                     rho=[{}] sigma={} tau={}",
                    f.info().name,
                    id.as_u32(),
                    fmt_slice(rho),
                    sigma.map_or("None".to_string(), |s| format!("[{}]", fmt_slice(s))),
                    tau.map_or("None".to_string(), |t| format!("[{}]", fmt_slice(t))),
                )
            });
            let fxc: [(&str, &[f64]); 6] = [
                ("v2rho2", &r.v2rho2),
                ("v2rhosigma", &r.v2rhosigma),
                ("v2sigma2", &r.v2sigma2),
                ("v2rhotau", &r.v2rhotau),
                ("v2sigmatau", &r.v2sigmatau),
                ("v2tau2", &r.v2tau2),
            ];
            for (name, slice) in fxc {
                for (k, &v) in slice.iter().enumerate() {
                    self.fxc_components += 1;
                    assert!(
                        v.is_finite(),
                        "NON-FINITE fxc {name}[{k}] = {v} :: {} (id {}) spin {spin:?} region {regions:?}\n  \
                         rho   = [{}]\n  sigma = {}",
                        f.info().name,
                        id.as_u32(),
                        fmt_slice(rho),
                        sigma.map_or("None".to_string(), |s| format!("[{}]", fmt_slice(s))),
                    );
                }
            }
            self.fxc_evals += 1;
            for &reg in regions {
                *self.fxc_by_region.entry(reg).or_insert(0) += 1;
            }
        } else {
            self.fxc_skipped += 1;
        }

        self.evals += 1;
        for &reg in regions {
            *self.by_region.entry(reg).or_insert(0) += 1;
        }
    }
}

/// Whether a finite input is inside the **physical domain for second
/// derivatives**. The `fxc` finiteness contract is bounded to the physical
/// reduced-gradient range (docs/api-convention.md §4 domain caveat / §8
/// divergence C): `fxc` overflows f64 at a lower gradient than vxc, so out there
/// neither library is meaningful and the gate does not chase f64-range behavior.
///
/// In domain: screened points (total `< thr`, where every output is exactly 0);
/// any LDA point (no gradient to overflow, finite across the density range); and
/// GGA points whose σ is within the documented cap **and** whose per-channel and
/// total reduced gradients are `≤ FXC_REDUCED_GRAD_MAX`. Mirrors the harness's
/// flooring/clamping so the predicate sees the same values the evaluator does.
fn fxc_in_domain(rho: &[f64], sigma: Option<&[f64]>, thr: f64) -> bool {
    let total: f64 = rho.iter().sum();
    if total < thr {
        return true; // screened → exact 0, always finite
    }
    let sig = match sigma {
        Some(s) => s,
        None => return true, // LDA: smooth over the whole physical density range
    };
    if sig.iter().any(|&x| x.abs() > FXC_SIGMA_CAP) {
        return false;
    }
    let st = thr.powf(4.0 / 3.0);
    let sfloor = st * st; // libxc σ floor = sigma_threshold²
                          // reduced gradient sqrt(σ_floored)/n_floored^(4/3)
    let red = |s: f64, nch: f64| s.max(sfloor).sqrt() / nch.max(thr).powf(4.0 / 3.0);
    if rho.len() == 1 {
        // unpolarized: per-channel n_a = n/2, σ_aa = σ/4; total uses n, σ
        let n = rho[0];
        red(sig[0] / 4.0, n / 2.0) <= FXC_REDUCED_GRAD_MAX && red(sig[0], n) <= FXC_REDUCED_GRAD_MAX
    } else {
        let (na, nb) = (rho[0], rho[1]);
        let (saa, sab, sbb) = (sig[0], sig[1], sig[2]);
        let saa_f = saa.max(sfloor);
        let sbb_f = sbb.max(sfloor);
        let s_ave = 0.5 * (saa_f + sbb_f);
        let sab_c = sab.clamp(-s_ave, s_ave);
        let sigma_tot = (saa_f + 2.0 * sab_c + sbb_f).max(0.0);
        let nt = na + nb;
        red(saa, na) <= FXC_REDUCED_GRAD_MAX
            && red(sbb, nb) <= FXC_REDUCED_GRAD_MAX
            && sigma_tot.sqrt() / nt.max(thr).powf(4.0 / 3.0) <= FXC_REDUCED_GRAD_MAX
    }
}

/// Full-precision slice formatting for reproducible failure messages.
fn fmt_slice(s: &[f64]) -> String {
    s.iter()
        .map(|x| format!("{x:.17e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Hand-constructed pathological input regions (functional-independent).
// ---------------------------------------------------------------------------

/// Unpolarized total densities: extreme `rs` (ρ from ~1e-14 to ~1e3), the
/// `dens_threshold` straddle (just below / at / just above each functional's
/// threshold ∈ {1e-15, 1e-14, 1e-12}), and a few strictly sub-threshold values
/// (which must screen to a finite exact 0).
fn unpol_densities() -> Vec<(f64, &'static str)> {
    let mut v = Vec::new();
    for &n in &[
        1e-14, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 31.6,
        100.0, 316.0, 1000.0,
    ] {
        v.push((n, "extreme_rs"));
    }
    for &thr in &[1e-15, 1e-14, 1e-12] {
        // strict `<` screen: == thr is *evaluated* (at the floor), < thr is zeroed.
        v.push((0.5 * thr, "dens_threshold_straddle"));
        v.push((thr * (1.0 - 1e-6), "dens_threshold_straddle"));
        v.push((thr, "dens_threshold_straddle"));
        v.push((thr * (1.0 + 1e-6), "dens_threshold_straddle"));
        v.push((2.0 * thr, "dens_threshold_straddle"));
    }
    for &n in &[0.0, 1e-300, 1e-18] {
        v.push((n, "sub_threshold"));
    }
    v
}

/// Unpolarized σ for a given density: exact 0 and tiny (both floored internally),
/// the small-σ band the sqrt-free per-spin reduced gradient repaired (old
/// divergence #4 — `v2sigma2` there is now accurate, not just finite), a sweep
/// targeting reduced gradient `x ≈ s` for `s ∈ {0.01 … 1000}` ("large sigma"),
/// and absolute-huge values.
fn unpol_sigmas(n: f64) -> Vec<(f64, &'static str)> {
    let mut v = vec![(0.0, "sigma_zero"), (1e-30, "sigma_tiny")];
    for &s in &[1e-6, 1e-8, 1e-10] {
        v.push((s, "sigma_small_band")); // sqrt-free fxc fix band (divergence #4)
    }
    let nn = n.max(1e-12);
    for &s in &[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
        v.push((s * s * nn.powf(8.0 / 3.0), "reduced_grad_scaled"));
    }
    v.push((1e6, "sigma_huge_abs"));
    v.push((1e12, "sigma_huge_abs"));
    v
}

/// Polarized (n_a, n_b): exact full polarization (`n_b` literally 0, both
/// orderings, across `rs`), a near-full-polarization sweep (`n_b/n_a` from 1e-4
/// down to 1e-12), `z = 0`, generic asymmetric, the threshold straddle (incl. one
/// channel sub-threshold while the total is well above), and extremes.
fn pol_density_pairs() -> Vec<(f64, f64, &'static str)> {
    let mut v = Vec::new();
    for &n in &[1e-12, 1e-6, 1e-3, 1.0, 1e3] {
        v.push((n, 0.0, "full_pol_exact")); // z = +1, n_b == 0
        v.push((0.0, n, "full_pol_exact")); // z = -1, n_a == 0
    }
    for &r in &[1e-4, 1e-6, 1e-8, 1e-10, 1e-12] {
        v.push((1.0, r, "near_full_pol"));
        v.push((r, 1.0, "near_full_pol"));
        v.push((1e3, 1e3 * r, "near_full_pol")); // near full pol at small rs too
    }
    for &n in &[1e-12, 1e-6, 1e-2, 1.0, 100.0, 1000.0] {
        v.push((n / 2.0, n / 2.0, "equal_spin")); // z = 0
    }
    for &(a, b) in &[(0.7, 0.3), (0.3, 0.7), (0.9, 0.1), (0.1, 0.9), (0.6, 0.4)] {
        v.push((a, b, "asymmetric"));
    }
    for &thr in &[1e-15, 1e-14, 1e-12] {
        v.push((0.6 * thr, 0.6 * thr, "dens_threshold_straddle")); // total 1.2·thr (above)
        v.push((0.4 * thr, 0.4 * thr, "dens_threshold_straddle")); // total 0.8·thr (screened)
        v.push((thr, thr, "dens_threshold_straddle"));
        v.push((0.5, 0.1 * thr, "one_channel_subthreshold"));
        v.push((0.1 * thr, 0.5, "one_channel_subthreshold"));
    }
    v.push((5e-15, 5e-15, "extreme_rs"));
    v.push((500.0, 500.0, "extreme_rs"));
    v.push((1e-300, 1e-300, "sub_threshold")); // screened
    v
}

/// Per-channel base (σ_aa, σ_bb) magnitudes for a polarized point: zero, a
/// reduced-gradient-scaled sweep, and deliberately lopsided pairs (huge σ on one
/// channel — especially stressing a near-zero minority channel).
fn base_sigma_pairs(na: f64, nb: f64) -> Vec<(f64, f64)> {
    let scaled = |n: f64, s: f64| {
        let nn = n.max(1e-10);
        s * s * nn.powf(8.0 / 3.0)
    };
    let mut v = vec![(0.0, 0.0)];
    for &s in &[0.1, 1.0, 100.0] {
        v.push((scaled(na, s), scaled(nb, s)));
    }
    // small absolute σ — the per-spin small-σ band repaired by the sqrt-free
    // reduced gradient (old divergence #4), now accurate as well as finite.
    v.push((1e-6, 1e-6));
    v.push((1e-8, 1e-8));
    v.push((1e-3, 0.0));
    v.push((0.0, 1e-3));
    v.push((1.0, 1e-12));
    v.push((1e-12, 1.0));
    v.push((1e6, 1e-6));
    v.push((1e-6, 1e6));
    v
}

/// Polarized σ triples (σ_aa, σ_ab, σ_bb) for a density pair. For each base
/// (σ_aa, σ_bb) it walks σ_ab across:
/// - the family's `σ_ab` clamp edge `±s_ave = ±½(σ_aa+σ_bb)`, just inside / at /
///   just outside (both signs, two ε scales, plus far beyond);
/// - the Cauchy–Schwarz / physical boundary `σ_ab² = σ_aa·σ_bb` (`±√(σ_aa σ_bb)`),
///   just inside / at / just outside;
/// - `σ_ab = −s_ave` (and beyond), which drives the *total* gradient `σ_tot =
///   σ_aa + 2σ_ab + σ_bb` to 0 — the cancellation that bit PBE-c.
///
/// It also appends two clean `σ_tot = 0` points where the f64 sum is *exactly* 0.
fn pol_sigma_triples(na: f64, nb: f64) -> Vec<(f64, f64, f64, &'static str)> {
    let mut v = Vec::new();
    for (saa, sbb) in base_sigma_pairs(na, nb) {
        let s_ave = 0.5 * (saa + sbb);
        let gm = (saa * sbb).sqrt();
        v.push((saa, 0.0, sbb, "sab_zero"));
        v.push((saa, s_ave, sbb, "sab_clamp_edge_hi"));
        v.push((saa, -s_ave, sbb, "sab_clamp_edge_lo")); // σ_tot → 0
        for &eps in &[1e-12, 1e-3] {
            v.push((saa, s_ave * (1.0 + eps), sbb, "sab_clamp_outside_hi"));
            v.push((saa, s_ave * (1.0 - eps), sbb, "sab_clamp_inside_hi"));
            v.push((saa, -s_ave * (1.0 + eps), sbb, "sab_clamp_outside_lo")); // clamp → σ_tot = 0
            v.push((saa, -s_ave * (1.0 - eps), sbb, "sab_clamp_inside_lo"));
        }
        v.push((saa, 10.0 * s_ave, sbb, "sab_clamp_outside_hi"));
        v.push((saa, -10.0 * s_ave, sbb, "sab_clamp_outside_lo"));
        v.push((saa, gm, sbb, "gm_boundary"));
        v.push((saa, -gm, sbb, "gm_boundary"));
        v.push((saa, gm * (1.0 + 1e-3), sbb, "gm_outside"));
        v.push((saa, gm * (1.0 - 1e-3), sbb, "gm_inside"));
        v.push((saa, -gm * (1.0 + 1e-3), sbb, "gm_outside"));
        v.push((saa, -gm * (1.0 - 1e-3), sbb, "gm_inside"));
    }
    // exact σ_tot = 0 (equal channels → the f64 sum σ_aa + 2σ_ab + σ_bb is 0.0).
    v.push((0.2, -0.2, 0.2, "sigma_tot_zero_exact"));
    v.push((0.0, 0.0, 0.0, "sigma_tot_zero_exact"));
    v
}

/// `K_FACTOR_C = (3/10)(6π²)^(2/3)`: τ_unif,σ = K_FACTOR_C·n_σ^(5/3).
const K_FACTOR_C: f64 = 4.557_799_872_345_596;

/// τ candidates for an unpolarized meta-GGA point (n, σ): the τ floor, the
/// von-Weizsäcker edge τ_W = σ/(8n) (α ≈ 0, the τ = τ_W hazard), the iso-orbital
/// point τ_W + τ_unif (α ≈ 1), a large-α value, and an absolute-large τ — the
/// τ-ratio hazard class (docs/api-convention.md §8). The harness floors τ to 1e-20 and the FHC
/// clamp keeps σ ≤ 8nτ, so every value yields finite outputs.
fn unpol_taus(n: f64, sigma: f64) -> Vec<(f64, &'static str)> {
    let nn = n.max(1e-300);
    let tw = sigma / (8.0 * nn); // von Weizsäcker τ_W
    let tunif = K_FACTOR_C * nn.powf(5.0 / 3.0);
    vec![
        (0.0, "tau_zero"),
        (tw, "tau_vw_edge"),       // α ≈ 0
        (tw + tunif, "alpha_one"), // α ≈ 1
        (tw + 5.0 * tunif, "alpha_large"),
        (1e8, "tau_huge"),
    ]
}

/// (τ_a, τ_b) candidates for a polarized meta-GGA point: τ floor (both, and one
/// channel at the floor while the other is physical — the minority-τ edge),
/// iso-orbital-ish (τ ≈ τ_unif per channel), large-α, and absolute-large.
fn pol_taus(na: f64, nb: f64) -> Vec<(f64, f64, &'static str)> {
    let ka = K_FACTOR_C * na.max(1e-300).powf(5.0 / 3.0);
    let kb = K_FACTOR_C * nb.max(1e-300).powf(5.0 / 3.0);
    vec![
        (0.0, 0.0, "tau_zero"),
        (ka, kb, "alpha_one"),
        (5.0 * ka, 5.0 * kb, "alpha_large"),
        (ka, 0.0, "tau_minority_floor"),
        (1e8, 1e8, "tau_huge"),
    ]
}

// ---------------------------------------------------------------------------
// Uniform-random supplement (deterministic, fixed seed).
// ---------------------------------------------------------------------------

fn loguniform(rng: &mut StdRng, lo: f64, hi: f64) -> f64 {
    let (a, b) = (lo.ln(), hi.ln());
    (a + (b - a) * rng.gen::<f64>()).exp()
}

/// A random σ magnitude: exactly 0 with probability 0.15, else log-uniform.
fn pick_sigma(rng: &mut StdRng) -> f64 {
    if rng.gen::<f64>() < 0.15 {
        0.0
    } else {
        loguniform(rng, 1e-20, 1e8)
    }
}

/// Random unpolarized (n, σ): log-uniform density spanning sub-threshold to large
/// (1e-16 … 1e4), σ sometimes exactly 0 else log-uniform (1e-20 … 1e8).
fn random_unpol(rng: &mut StdRng, count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|_| {
            let n = loguniform(rng, 1e-16, 1e4);
            let s = if rng.gen::<f64>() < 0.15 {
                0.0
            } else {
                loguniform(rng, 1e-20, 1e8)
            };
            (n, s)
        })
        .collect()
}

/// Random polarized (n_a, n_b, σ_aa, σ_ab, σ_bb): independent log-uniform spin
/// densities (so wide `rs` and chance near-full-polarization both arise), σ_aa/σ_bb
/// log-uniform or 0, and σ_ab spanning inside-to-outside the clamp (`|σ_ab|` up to
/// ~3·s_ave), occasionally far beyond.
fn random_pol(rng: &mut StdRng, count: usize) -> Vec<(f64, f64, f64, f64, f64)> {
    (0..count)
        .map(|_| {
            let na = loguniform(rng, 1e-16, 1e4);
            let nb = loguniform(rng, 1e-16, 1e4);
            let saa = pick_sigma(rng);
            let sbb = pick_sigma(rng);
            let s_ave = 0.5 * (saa + sbb);
            let scale = if rng.gen::<f64>() < 0.1 { 50.0 } else { 3.0 };
            let sab = (2.0 * rng.gen::<f64>() - 1.0) * scale * s_ave;
            (na, nb, saa, sab, sbb)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

/// Every registered functional (`FunctionalId::ALL`) × both spins must produce
/// only finite outputs over the densified pathological regions plus a
/// uniform-random supplement.
#[test]
fn fuzz_all_functionals_finite() {
    let mut census = Census::new();

    let unpol_dens = unpol_densities();
    let pol_pairs = pol_density_pairs();

    // Deterministic random supplements, generated once and reused per functional.
    let mut rng = StdRng::seed_from_u64(SEED);
    let unpol_rand = random_unpol(&mut rng, RANDOM_UNPOL);
    let pol_rand = random_pol(&mut rng, RANDOM_POL);

    for &id in FunctionalId::ALL {
        for &spin in &[Spin::Unpolarized, Spin::Polarized] {
            let f = Functional::new(id, spin).expect("registered functional must build");
            let needs_sigma = f.info().needs_sigma;
            let needs_tau = f.info().needs_tau;

            match spin {
                Spin::Unpolarized => {
                    for &(n, dreg) in &unpol_dens {
                        if needs_sigma {
                            for (s, sreg) in unpol_sigmas(n) {
                                if needs_tau {
                                    for (t, treg) in unpol_taus(n, s) {
                                        census.check(
                                            &f,
                                            id,
                                            spin,
                                            &[n],
                                            Some(&[s]),
                                            Some(&[t]),
                                            &[dreg, sreg, treg],
                                        );
                                    }
                                } else {
                                    census.check(
                                        &f,
                                        id,
                                        spin,
                                        &[n],
                                        Some(&[s]),
                                        None,
                                        &[dreg, sreg],
                                    );
                                }
                            }
                        } else {
                            census.check(&f, id, spin, &[n], None, None, &[dreg]);
                        }
                    }
                    for &(n, s) in &unpol_rand {
                        if needs_sigma {
                            if needs_tau {
                                // a couple of τ per random point (floor + iso-orbital)
                                for (t, _) in unpol_taus(n, s).into_iter().take(3) {
                                    census.check(
                                        &f,
                                        id,
                                        spin,
                                        &[n],
                                        Some(&[s]),
                                        Some(&[t]),
                                        &["random"],
                                    );
                                }
                            } else {
                                census.check(&f, id, spin, &[n], Some(&[s]), None, &["random"]);
                            }
                        } else {
                            census.check(&f, id, spin, &[n], None, None, &["random"]);
                        }
                    }
                }
                Spin::Polarized => {
                    for &(na, nb, dreg) in &pol_pairs {
                        if needs_sigma {
                            for (saa, sab, sbb, sreg) in pol_sigma_triples(na, nb) {
                                if needs_tau {
                                    for (ta, tb, treg) in pol_taus(na, nb) {
                                        census.check(
                                            &f,
                                            id,
                                            spin,
                                            &[na, nb],
                                            Some(&[saa, sab, sbb]),
                                            Some(&[ta, tb]),
                                            &[dreg, sreg, treg],
                                        );
                                    }
                                } else {
                                    census.check(
                                        &f,
                                        id,
                                        spin,
                                        &[na, nb],
                                        Some(&[saa, sab, sbb]),
                                        None,
                                        &[dreg, sreg],
                                    );
                                }
                            }
                        } else {
                            census.check(&f, id, spin, &[na, nb], None, None, &[dreg]);
                        }
                    }
                    for &(na, nb, saa, sab, sbb) in &pol_rand {
                        if needs_sigma {
                            if needs_tau {
                                for (ta, tb, _) in pol_taus(na, nb).into_iter().take(3) {
                                    census.check(
                                        &f,
                                        id,
                                        spin,
                                        &[na, nb],
                                        Some(&[saa, sab, sbb]),
                                        Some(&[ta, tb]),
                                        &["random"],
                                    );
                                }
                            } else {
                                census.check(
                                    &f,
                                    id,
                                    spin,
                                    &[na, nb],
                                    Some(&[saa, sab, sbb]),
                                    None,
                                    &["random"],
                                );
                            }
                        } else {
                            census.check(&f, id, spin, &[na, nb], None, None, &["random"]);
                        }
                    }
                }
                // `Spin` is #[non_exhaustive]; no other variant exists today.
                _ => unreachable!("unknown Spin variant"),
            }
        }
    }

    // Coverage census — proves which regions were actually exercised (run with
    // `-- --nocapture` to see it).
    println!("\n=== xcx fuzz coverage (seed = 0x{SEED:016X}) ===");
    println!("functionals: {}  ×  spins: 2", FunctionalId::ALL.len());
    println!(
        "total evaluations: {}   finiteness checks: {}",
        census.evals, census.components
    );
    println!("evaluations crediting each region (summed over all functionals/spins):");
    for (region, n) in &census.by_region {
        let fxc = census.fxc_by_region.get(region).copied().unwrap_or(0);
        println!("  {region:<28} vxc {n:>9}   fxc {fxc:>9}");
    }
    println!(
        "fxc: {} evaluations checked, {} finiteness checks, {} skipped (out of physical domain)",
        census.fxc_evals, census.fxc_components, census.fxc_skipped
    );
    println!("=== all outputs finite ===\n");
}

/// Batch (`np > 1`) must agree bit-for-bit with single-point (`np = 1`) eval and
/// stay finite. Guards the family harness's per-point indexing (`2*i`, `3*i`),
/// which the np=1 sweep above does not exercise.
#[test]
fn batch_matches_single_point() {
    // A handful spanning normal, full-pol, σ_tot→0, threshold, and large σ.
    let pol: &[(f64, f64, f64, f64, f64)] = &[
        (0.6, 0.3, 0.1, 0.05, 0.08),   // generic
        (1.0, 0.0, 0.2, 0.0, 0.0),     // full polarization
        (0.5, 0.5, 0.2, -0.2, 0.2),    // σ_tot = 0 exactly
        (1.0, 1e-10, 0.1, 0.0, 1e-8),  // near full pol
        (1e-12, 1e-12, 0.0, 0.0, 0.0), // at threshold, σ = 0
        (2.0, 1.0, 1e6, 0.0, 1e6),     // large σ
    ];

    // Per-point τ (for meta-GGA): physical-ish iso-orbital values, plus the τ
    // floor where τ would otherwise be 0.
    let taus: &[(f64, f64)] = &[
        (0.4, 0.25),
        (0.5, 1e-20),
        (0.3, 0.3),
        (0.6, 1e-12),
        (1e-20, 1e-20),
        (5.0, 3.0),
    ];

    for &id in FunctionalId::ALL {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let needs_sigma = f.info().needs_sigma;
        let needs_tau = f.info().needs_tau;
        let np = pol.len();

        let mut rho = Vec::with_capacity(2 * np);
        let mut sig = Vec::with_capacity(3 * np);
        let mut tau = Vec::with_capacity(2 * np);
        for (&(na, nb, saa, sab, sbb), &(ta, tb)) in pol.iter().zip(taus) {
            rho.push(na);
            rho.push(nb);
            sig.push(saa);
            sig.push(sab);
            sig.push(sbb);
            tau.push(ta);
            tau.push(tb);
        }
        let input = match (needs_sigma, needs_tau) {
            (true, true) => XcInput::gga(&rho, &sig).with_tau(&tau),
            (true, false) => XcInput::gga(&rho, &sig),
            _ => XcInput::lda(&rho),
        };
        let batch = f.eval(np, &input).unwrap();

        // every batch component finite
        for v in batch
            .exc
            .iter()
            .chain(&batch.vrho)
            .chain(&batch.vsigma)
            .chain(&batch.vtau)
            .chain(&batch.vlapl)
        {
            assert!(
                v.is_finite(),
                "{}: non-finite batch output {v}",
                f.info().name
            );
        }

        // and identical to evaluating each point alone
        for (i, (&(na, nb, saa, sab, sbb), &(ta, tb))) in pol.iter().zip(taus).enumerate() {
            let rho1 = [na, nb];
            let sig1 = [saa, sab, sbb];
            let tau1 = [ta, tb];
            let one = match (needs_sigma, needs_tau) {
                (true, true) => f
                    .eval(1, &XcInput::gga(&rho1, &sig1).with_tau(&tau1))
                    .unwrap(),
                (true, false) => f.eval(1, &XcInput::gga(&rho1, &sig1)).unwrap(),
                _ => f.eval(1, &XcInput::lda(&rho1)).unwrap(),
            };
            assert_eq!(
                batch.exc[i].to_bits(),
                one.exc[0].to_bits(),
                "{} exc[{i}]",
                f.info().name
            );
            for c in 0..2 {
                assert_eq!(
                    batch.vrho[2 * i + c].to_bits(),
                    one.vrho[c].to_bits(),
                    "{} vrho[{i}][{c}]",
                    f.info().name
                );
            }
            if needs_sigma {
                for c in 0..3 {
                    assert_eq!(
                        batch.vsigma[3 * i + c].to_bits(),
                        one.vsigma[c].to_bits(),
                        "{} vsigma[{i}][{c}]",
                        f.info().name
                    );
                }
            }
            if needs_tau {
                for c in 0..2 {
                    assert_eq!(
                        batch.vtau[2 * i + c].to_bits(),
                        one.vtau[c].to_bits(),
                        "{} vtau[{i}][{c}]",
                        f.info().name
                    );
                }
            }
        }
    }
}
