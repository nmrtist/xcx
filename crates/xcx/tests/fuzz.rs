// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Finiteness / fuzz gate for the public `xcx` API (contract: docs/api-convention.md §4).
//!
//! Contract under test (docs/api-convention.md §4): **every output component is
//! finite for every finite input** — not just `exc`, but each `vrho` channel and
//! all three `vsigma` (aa/ab/bb), and `vtau`/`vlapl` (empty until meta-GGA, but
//! checked anyway). This is orthogonal to the golden suite (which checks "numbers
//! match libxc"); here we only check "no NaN/Inf/panic," over all 12 functionals
//! × both spins.
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

// ---------------------------------------------------------------------------
// Coverage census + the per-point finiteness check.
// ---------------------------------------------------------------------------

/// Tracks how many evaluations touched each named region, plus totals, so the
/// run can prove which pathological regions were actually exercised.
struct Census {
    evals: u64,
    components: u64,
    by_region: BTreeMap<&'static str, u64>,
}

impl Census {
    fn new() -> Self {
        Self {
            evals: 0,
            components: 0,
            by_region: BTreeMap::new(),
        }
    }

    /// Evaluate one point (`np = 1`) and assert every output component finite.
    /// `regions` are the named pathological regions this input belongs to (e.g.
    /// `["full_pol_exact", "sigma_tot_zero_exact"]`); each is credited.
    fn check(
        &mut self,
        f: &Functional,
        id: FunctionalId,
        spin: Spin,
        rho: &[f64],
        sigma: Option<&[f64]>,
        regions: &[&'static str],
    ) {
        let input = match sigma {
            Some(s) => XcInput::gga(rho, s),
            None => XcInput::lda(rho),
        };
        let r = f.eval(1, &input).unwrap_or_else(|e| {
            panic!(
                "{} (id {}) spin {spin:?}: eval errored on finite input {e:?}: rho=[{}] sigma={}",
                f.info().name,
                id.as_u32(),
                fmt_slice(rho),
                sigma.map_or("None".to_string(), |s| format!("[{}]", fmt_slice(s))),
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

        self.evals += 1;
        for &reg in regions {
            *self.by_region.entry(reg).or_insert(0) += 1;
        }
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
/// a sweep targeting reduced gradient `x ≈ s` for `s ∈ {0.01 … 1000}` ("large
/// sigma"), and absolute-huge values.
fn unpol_sigmas(n: f64) -> Vec<(f64, &'static str)> {
    let mut v = vec![(0.0, "sigma_zero"), (1e-30, "sigma_tiny")];
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

/// All 12 functionals × both spins must produce only finite outputs over the
/// densified pathological regions plus a uniform-random supplement.
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
            let f = Functional::new(id, spin).expect("v0.1 functional must build");
            let needs_sigma = f.info().needs_sigma;

            match spin {
                Spin::Unpolarized => {
                    for &(n, dreg) in &unpol_dens {
                        if needs_sigma {
                            for (s, sreg) in unpol_sigmas(n) {
                                census.check(&f, id, spin, &[n], Some(&[s]), &[dreg, sreg]);
                            }
                        } else {
                            census.check(&f, id, spin, &[n], None, &[dreg]);
                        }
                    }
                    for &(n, s) in &unpol_rand {
                        if needs_sigma {
                            census.check(&f, id, spin, &[n], Some(&[s]), &["random"]);
                        } else {
                            census.check(&f, id, spin, &[n], None, &["random"]);
                        }
                    }
                }
                Spin::Polarized => {
                    for &(na, nb, dreg) in &pol_pairs {
                        if needs_sigma {
                            for (saa, sab, sbb, sreg) in pol_sigma_triples(na, nb) {
                                census.check(
                                    &f,
                                    id,
                                    spin,
                                    &[na, nb],
                                    Some(&[saa, sab, sbb]),
                                    &[dreg, sreg],
                                );
                            }
                        } else {
                            census.check(&f, id, spin, &[na, nb], None, &[dreg]);
                        }
                    }
                    for &(na, nb, saa, sab, sbb) in &pol_rand {
                        if needs_sigma {
                            census.check(
                                &f,
                                id,
                                spin,
                                &[na, nb],
                                Some(&[saa, sab, sbb]),
                                &["random"],
                            );
                        } else {
                            census.check(&f, id, spin, &[na, nb], None, &["random"]);
                        }
                    }
                }
                // `Spin` is #[non_exhaustive]; no other variant exists in v0.1.
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
        println!("  {region:<28} {n:>9}");
    }
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

    for &id in FunctionalId::ALL {
        let f = Functional::new(id, Spin::Polarized).unwrap();
        let needs_sigma = f.info().needs_sigma;
        let np = pol.len();

        let mut rho = Vec::with_capacity(2 * np);
        let mut sig = Vec::with_capacity(3 * np);
        for &(na, nb, saa, sab, sbb) in pol {
            rho.push(na);
            rho.push(nb);
            sig.push(saa);
            sig.push(sab);
            sig.push(sbb);
        }
        let input = if needs_sigma {
            XcInput::gga(&rho, &sig)
        } else {
            XcInput::lda(&rho)
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
        for (i, &(na, nb, saa, sab, sbb)) in pol.iter().enumerate() {
            let rho1 = [na, nb];
            let sig1 = [saa, sab, sbb];
            let one = if needs_sigma {
                f.eval(1, &XcInput::gga(&rho1, &sig1)).unwrap()
            } else {
                f.eval(1, &XcInput::lda(&rho1)).unwrap()
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
        }
    }
}
