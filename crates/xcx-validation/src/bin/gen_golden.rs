// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Regenerate golden snapshots from a conda-forge libxc.
//!
//! ```text
//! conda create -n xcref -c conda-forge libxc
//! $env:XCX_LIBXC_DLL = "<env>\Library\bin\xc.dll"   # or set CONDA_PREFIX
//! cargo run -p xcx-validation --features libxc-ffi --bin gen_golden
//! # optional: regenerate only some functionals (per-functional commits)
//! cargo run -p xcx-validation --features libxc-ffi --bin gen_golden -- lda_x
//! ```
//!
//! The committed snapshots under `testdata/` are then used by the (default,
//! libxc-free) golden test in CI.
//!
//! Each functional's snapshot holds **vxc cases** on the extreme/edge points
//! (full polarization, screening straddles, the σ_ab clamp, very large σ) and
//! **fxc cases** on a curated interior set that now **includes the small-σ band**
//! (down to σ = 1e-8 and exact 0) the sqrt-free per-spin reduced gradient fixed. The fxc points still avoid |ζ| → 1 (where libxc's
//! analytic fxc loses accuracy). Exact σ = 0 is pinned only where libxc itself is
//! accurate there (see [`sigma0_pinnable`]): B88's analytic `v2sigma2` truncates
//! at its σ-floor, so B88 and the B88-containing hybrids are excluded at *exact*
//! σ = 0 — the lone, far narrower remnant of divergence #4 (docs/api-convention.md §8).

#[cfg(feature = "libxc-ffi")]
fn main() {
    use std::path::Path;
    use xcx_validation::ffi::Libxc;

    let xc = Libxc::load();
    let (vmaj, vmin, vmic) = xc.version();
    let version = format!("{vmaj}.{vmin}.{vmic}");
    eprintln!("libxc version {version}");

    // Optional functional-name filter so a single functional can be regenerated
    // and committed at a time (leaving the other snapshots — and the working
    // tree — untouched between per-functional fxc commits).
    let filter: Vec<String> = std::env::args().skip(1).collect();
    let want = |name: &str| filter.is_empty() || filter.iter().any(|f| f == name);

    let outdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    std::fs::create_dir_all(&outdir).unwrap();

    // (name, family): 0 = LDA, 1 = GGA/hybrid-GGA, 2 = meta-GGA
    let functionals: &[(&str, u8)] = &[
        ("lda_x", 0),
        ("lda_c_pw", 0),
        ("lda_c_vwn", 0),
        ("lda_c_vwn_3", 0),
        ("lda_c_vwn_rpa", 0),
        ("gga_x_pbe", 1),
        ("gga_x_b88", 1),
        ("gga_c_pbe", 1),
        ("gga_c_lyp", 1),
        ("gga_x_pbe_r", 1),
        ("gga_x_pbe_sol", 1),
        ("gga_x_rpbe", 1),
        ("gga_c_pbe_sol", 1),
        ("hyb_gga_xc_pbeh", 1),
        ("hyb_gga_xc_b3lyp5", 1),
        ("hyb_gga_xc_b3lyp", 1),
        ("mgga_x_tpss", 2),
        ("mgga_c_tpss", 2),
        ("mgga_x_r2scan", 2),
        ("mgga_c_r2scan", 2),
        ("mgga_x_m06_l", 2),
        ("mgga_c_m06_l", 2),
        ("hyb_mgga_x_m06_2x", 2),
        ("mgga_c_m06_2x", 2),
        ("hyb_mgga_xc_pw6b95", 2),
        ("mgga_xc_b97m_v", 2),
        ("hyb_gga_xc_wb97x_v", 1),
        ("hyb_mgga_xc_wb97m_v", 2),
        ("gga_c_p86", 1),
        ("gga_xc_b97_3c", 1),
    ];
    for &(name, family) in functionals {
        if !want(name) {
            continue;
        }
        match family {
            0 => gen_lda(&xc, &version, &outdir, name),
            1 => gen_gga(&xc, &version, &outdir, name, sigma0_pinnable(name)),
            _ => gen_mgga(&xc, &version, &outdir, name),
        }
    }

    // --- Double-hybrid semilocal cores: component-weighted libxc mixes. ---
    //
    // libxc (any release) ships no double hybrids, so these snapshots are
    // built by evaluating each libxc *component* (optionally with
    // `xc_func_set_ext_params` overrides) and accumulating the weighted sum —
    // genuinely libxc-derived reference data for the registered mixes.
    // `libxc_id` records the xcx-private id (>= 100000); `libxc_version` is
    // tagged "(component mix)".
    //
    // B2PLYP (Grimme 2006): 0.47·B88-x + 0.73·LYP-c. Contains B88 ⇒ exact-σ=0
    // fxc not pinnable (same as b3lyp).
    if want("hyb_gga_xc_b2plyp") {
        gen_gga_mix(
            &xc,
            &version,
            &outdir,
            "hyb_gga_xc_b2plyp",
            100_001,
            &[(0.47, "gga_x_b88", None), (0.73, "gga_c_lyp", None)],
            false,
        );
    }
    // revDSD-PBEP86-D4 (Santra/Sylvetsky/Martin 2019): 0.31·PBE-x + 0.4210·P86-c.
    // P86's odd-x_t gradient term ⇒ exact-σ=0 fxc not pinnable (σ^(-1/2) class).
    if want("hyb_gga_xc_revdsd_pbep86_d4") {
        gen_gga_mix(
            &xc,
            &version,
            &outdir,
            "hyb_gga_xc_revdsd_pbep86_d4",
            100_002,
            &[(0.31, "gga_x_pbe", None), (0.4210, "gga_c_p86", None)],
            false,
        );
    }
    // PWPB95 (Goerigk & Grimme 2011): 0.50·mPW-x(reopt.) + 0.731·B95-c(reopt.),
    // via libxc ext-params overrides: mpw91 {bt, alpha(s-space), expo} and
    // bc95 {css, copp}. Paper values (cross-checked vs Psi4's PWPB95 tweak):
    // bt = 0.004440, c = 0.32620 (w-space), expo = 3.7868.
    if want("hyb_mgga_xc_pwpb95") {
        // libxc util.h X2S = 1/(2·(6π²)^(1/3)); PWPB95's α is passed in
        // s-space exactly as Psi4/libxc's pw6b95-style init does: c/X2S².
        let x2s = 1.0 / (2.0 * (6.0 * std::f64::consts::PI.powi(2)).cbrt());
        let par_x = vec![0.004440, 0.32620 / (x2s * x2s), 3.7868];
        let par_c = vec![0.03241, 0.00250];
        gen_mgga_mix(
            &xc,
            &version,
            &outdir,
            "hyb_mgga_xc_pwpb95",
            100_003,
            &[
                (0.50, "gga_x_mpw91", MixExt::Gga(par_x)),
                (1.0 - 0.269, "mgga_c_bc95", MixExt::Mgga(par_c)),
            ],
        );
    }
    // PBEh-3c (Grimme et al. 2015) semilocal core: libxc 6.1.0 ships B97-3c but
    // not PBEh-3c, so the snapshot is a libxc *component mix* through
    // `xc_func_set_ext_params`: 0.58·gga_x_pbe{_kappa = 1.0245, _mu = 10/81}
    // + 1.0·gga_c_pbe{_beta = 0.03, _gamma/_B/_tscale at defaults}. PBE family
    // ⇒ exact-σ=0 fxc pinnable.
    if want("hyb_gga_xc_pbeh_3c") {
        gen_gga_mix(
            &xc,
            &version,
            &outdir,
            "hyb_gga_xc_pbeh_3c",
            100_005,
            &[
                (0.58, "gga_x_pbe", Some(&[1.0245, 10.0 / 81.0])),
                (
                    1.0,
                    "gga_c_pbe",
                    Some(&[0.03, 0.031090690869654895034, 1.0, 1.0]),
                ),
            ],
            true,
        );
    }
    // Constructor cross-check corpus (testdata/constructor/, loaded by
    // tests/constructor.rs rather than the by-name golden test): the
    // *semilocal part* of libxc's original B97 (hyb_gga_xc_b97, id 407),
    // pinned against the public `Functional::b97_xc` series constructor fed
    // Becke's original coefficients.
    if want("hyb_gga_xc_b97") {
        let cdir = outdir.join("constructor");
        std::fs::create_dir_all(&cdir).unwrap();
        gen_gga(&xc, &version, &cdir, "hyb_gga_xc_b97", true);
    }
}

/// Component spec for a meta-GGA-family mix: whether the component is
/// evaluated through the GGA or meta-GGA libxc entry points, with optional
/// `xc_func_set_ext_params` overrides.
#[cfg(feature = "libxc-ffi")]
enum MixExt {
    Gga(Vec<f64>),
    Mgga(Vec<f64>),
}

/// Whether libxc's *analytic* `v2sigma2` is accurate at **exact** σ = 0 for this
/// functional, so an exact-σ=0 fxc golden point can be pinned to it. `false` for
/// B88 and the B88-containing hybrids: libxc's B88 `v2sigma2` truncates to (5/8)×
/// the true limit at/below its σ-floor (an FFI-measured libxc artifact — libxc
/// emits the correct limit at σ = 1e-8…1e-20, then floors; xcx computes the true
/// limit at σ = 0). See `docs/api-convention.md` §8.
/// PBE-x's rational enhancement is smooth through 0 (σ=0 value == limit), so
/// PBE-x / PBE-c / LYP / PBEH are accurate there and *are* pinned at σ = 0. The
/// PBE-x family variants share that smoothness: revPBE/PBEsol-x are the same
/// rational form (κ/μ swapped), RPBE's `1 + κ(1 − exp(−μs²/κ))` is entire, and
/// PBEsol-c is PBE-c's rational H with β swapped — all accurate at σ = 0 and pinned.
#[cfg(feature = "libxc-ffi")]
fn sigma0_pinnable(name: &str) -> bool {
    !matches!(name, "gga_x_b88" | "hyb_gga_xc_b3lyp" | "hyb_gga_xc_b3lyp5")
}

/// Whether **every** fxc block of this meta-GGA matches libxc's analytic fxc to
/// ≤1e-10 down to (and including) exact σ = 0 — so the small-σ fxc band can be
/// pinned all the way to 0, not just to σ = 1e-5. Established by FFI sweep
/// (`bin/sweep_fxc`, meta-GGA section): `mgga_c_tpss`, `mgga_x_r2scan`,
/// `mgga_c_r2scan`, and **both M06-L functionals** all match to ~1e-16 across the
/// six blocks down to exact σ = 0 (their smooth, sqrt-free enhancements — r2SCAN's
/// damping, M06-L's PBE/`gtv4`/`b97_g`/`Fermi_D` rationals — have no `v2sigma2`
/// σ-floor artifact on *either* side), so they are pinned to 0. `mgga_x_tpss` is
/// **not**: its
/// `v2sigma2` is the lone block where libxc's analytic second derivative degrades
/// toward its σ-floor below σ ≈ 1e-6 (the TPSS analogue of divergence #4 — xcx is
/// the accurate side, locked libxc-free by
/// `tests/fxc.rs::mgga_v2sigma2_stable_into_small_sigma`). TPSS-x's other five
/// blocks are clean to σ = 0, but the per-case golden compares all blocks at once,
/// so TPSS-x's small-σ fxc stays pinned only at σ ≥ 1e-5.
#[cfg(feature = "libxc-ffi")]
fn mgga_fxc_smallsigma_to_zero(name: &str) -> bool {
    matches!(
        name,
        "mgga_c_tpss" | "mgga_x_r2scan" | "mgga_c_r2scan" | "mgga_x_m06_l" | "mgga_c_m06_l"
    )
}

/// Minimum **total** density at which a meta-GGA's first derivatives match libxc to
/// ≤1e-10, so the extreme-low-density vxc points can be pinned. `mgga_c_r2scan`'s
/// low-density gradient derivatives (`vsigma`/`vtau`) hit the analytic-vs-AD
/// cancellation (divergence #1 class) earlier than the others — below n ≈ 1e-8 the
/// divergence exceeds 1e-10 while the energy still matches — so its extreme points
/// (n < 1e-8) are dropped from the golden vxc set (FFI-measured crossover ≈
/// 1e-10…1e-11; the FD-of-libxc-energy anchor is too imprecise at low density to
/// declare an xcx-accurate side, so this is match-libxc, and fuzz covers finiteness
/// to n = 1e-14). All others (TPSS x/c, r2SCAN-x) are accurate to the n = 1e-14
/// floor and keep every point.
#[cfg(feature = "libxc-ffi")]
fn mgga_vxc_dens_floor(name: &str) -> f64 {
    match name {
        "mgga_c_r2scan" => 1e-8,
        // B97M-V's per-spin x_σ²/t_σ series hit the same extreme-low-density
        // analytic-vs-AD cancellation on vsigma (FFI-measured rel ~1.3e-9 at
        // n = 1e-14 and ~1.5e-10 at n = 1e-10, energy matching throughout);
        // pinned from n = 1e-8 up like r2SCAN-c. ωB97M-V (threshold 1e-13)
        // matches ≤ 1e-10 at every extreme point and keeps them all.
        "mgga_xc_b97m_v" => 1e-8,
        // PW6B95's B95 part hits the same floored-edge analytic-vs-AD
        // cancellation at the n = 1e-14 extreme point (rel ~3e-10 on vsigma,
        // FFI-measured); energy still matches. Drop only the sub-1e-12 extreme.
        "hyb_mgga_xc_pw6b95" => 1e-12,
        _ => 0.0,
    }
}

/// Whether to drop the **exact** full-polarization points (a spin channel
/// literally 0, floored to `dens_threshold`) from this meta-GGA's vxc golden set.
/// `true` for `mgga_c_m06_l`: its kinetic enhancements use the **raw** per-spin
/// reduced KE `t_σ = τ_σ/n_σ^(5/3)` and reduced gradient `x_σ² = σ_σσ/n_σ^(8/3)`
/// (unlike TPSS/r2SCAN's `(n_σ/n)^(5/3)`-weighted `t_total`, which cancels the
/// `n_σ^(-5/3)` factor). As a minority density `n_b → 0`, those per-spin
/// derivatives blow up `∝ n_b^(-8/3)`, and libxc's analytic derivative and xcx's
/// forward-AD diverge by f64 cancellation amplified by the floor (FFI-measured
/// minority-channel crossover ≈ `n_b` 1e-6; rel error ~1e-6…1e-5 at the floored
/// exact edge). The **energy** matches throughout, and the *physical* near-full-pol
/// point `[1.0, 1e-4]` matches ≤1e-10 and is kept — only the non-physical exact
/// edge (`n_b = 0`) is dropped. This is the M06-L analogue of the low-density
/// first-derivative divergence #1/D (and the GGA minority-vrho divergence B); fuzz
/// still covers finiteness at exact full polarization. See `docs/api-convention.md`
/// §8.
#[cfg(feature = "libxc-ffi")]
fn mgga_vxc_drop_full_pol(name: &str) -> bool {
    // mgga_c_m06_2x shares M06-L's per-spin raw `t_σ`/`x_σ²` form (and PW6B95's
    // B95 part the same class), so the same exact-full-polarization floored-edge
    // analytic-vs-AD artifact applies.
    // mgga_xc_b97m_v: same raw per-spin x_σ²/t_σ class; additionally its
    // opposite-spin kinetic variable w_os is evaluated in xcx's symmetric,
    // cancellation-free regrouping (algebraically identical to libxc's
    // (K(t₀+t₁) − 2t₀t₁)/(…) form, which piles up 2K² before cancelling), so at
    // the floored minority edge the two sides differ by amplified f64 noise
    // (FFI-measured: vtau_b rel ~0.15 at the τ-floor corner, vrho ~9e-4) while
    // the energy and all physical points match ≤ 1e-10. ωB97M-V (threshold
    // 1e-13) matches even at the exact edge and stays fully pinned.
    matches!(
        name,
        "mgga_c_m06_l" | "mgga_c_m06_2x" | "hyb_mgga_xc_pw6b95" | "mgga_xc_b97m_v"
    )
}

/// Build one LDA `GoldenCase`: libxc vxc always, plus `v2rho2` (fxc) when
/// `want_fxc`. `rho` is the packed input (`np` unpolarized, `2*np` polarized).
#[cfg(feature = "libxc-ffi")]
#[allow(clippy::too_many_arguments)]
fn lda_case(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    name: &str,
    id: i32,
    spin: &str,
    nspin: i32,
    rho: Vec<f64>,
    want_fxc: bool,
) -> xcx_validation::GoldenCase {
    use xcx_validation::GoldenCase;
    let np = rho.len() / nspin as usize;
    let (exc, vrho) = xc.lda_exc_vxc(id, nspin, np, &rho);
    let v2rho2 = if want_fxc {
        xc.lda_fxc(id, nspin, np, &rho)
    } else {
        vec![]
    };
    GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: spin.into(),
        np,
        rho,
        sigma: vec![],
        tau: vec![],
        exc,
        vrho,
        vsigma: vec![],
        vtau: vec![],
        v2rho2,
        v2rhosigma: vec![],
        v2sigma2: vec![],
        v2rhotau: vec![],
        v2sigmatau: vec![],
        v2tau2: vec![],
    }
}

/// Build one GGA `GoldenCase`: libxc vxc always, plus the full fxc tensor when
/// `want_fxc`.
#[cfg(feature = "libxc-ffi")]
#[allow(clippy::too_many_arguments)]
fn gga_case(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    name: &str,
    id: i32,
    spin: &str,
    nspin: i32,
    rho: Vec<f64>,
    sigma: Vec<f64>,
    want_fxc: bool,
) -> xcx_validation::GoldenCase {
    use xcx_validation::GoldenCase;
    let np = rho.len() / nspin as usize;
    let (exc, vrho, vsigma) = xc.gga_exc_vxc(id, nspin, np, &rho, &sigma);
    let (v2rho2, v2rhosigma, v2sigma2) = if want_fxc {
        xc.gga_fxc(id, nspin, np, &rho, &sigma)
    } else {
        (vec![], vec![], vec![])
    };
    GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: spin.into(),
        np,
        rho,
        sigma,
        tau: vec![],
        exc,
        vrho,
        vsigma,
        vtau: vec![],
        v2rho2,
        v2rhosigma,
        v2sigma2,
        v2rhotau: vec![],
        v2sigmatau: vec![],
        v2tau2: vec![],
    }
}

/// Flatten `(n_a, n_b)` pairs into a packed polarized `rho`.
#[cfg(feature = "libxc-ffi")]
fn flat_pairs(pairs: &[(f64, f64)]) -> Vec<f64> {
    pairs.iter().flat_map(|&(a, b)| [a, b]).collect()
}

/// Snapshot one LDA functional: vxc on the extreme/edge points (both spins),
/// then fxc on a curated interior point set (both spins).
#[cfg(feature = "libxc-ffi")]
fn gen_lda(xc: &xcx_validation::ffi::Libxc, version: &str, outdir: &std::path::Path, name: &str) {
    let id = xc.number(name);
    assert!(id > 0, "libxc does not know `{name}` (got id {id})");

    // --- vxc: extreme points (unchanged from the original snapshots) ---
    let unpol_rho: Vec<f64> = vec![
        1e-16, 1e-15, 1e-14, 1e-10, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0,
    ];
    let pol_pairs: &[(f64, f64)] = &[
        (0.5, 0.5),
        (0.7, 0.3),
        (0.9, 0.1),
        (1.0, 1e-10),
        (1e-10, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (3.0, 2.0),
        (1e-3, 1e-4),
        (100.0, 50.0),
        (1e-13, 1e-14),
    ];

    // --- fxc: curated interior points (no exact full polarization) ---
    let fxc_unpol_rho: Vec<f64> = vec![1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0];
    let fxc_pol_pairs: &[(f64, f64)] = &[
        (0.5, 0.5),
        (0.7, 0.3),
        (0.9, 0.1),
        (0.6, 0.4),
        (3.0, 2.0),
        (100.0, 50.0),
        (1e-3, 1e-4),
        (2.0, 0.5),
    ];

    let cases = vec![
        lda_case(xc, version, name, id, "unpolarized", 1, unpol_rho, false),
        lda_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            flat_pairs(pol_pairs),
            false,
        ),
        lda_case(xc, version, name, id, "unpolarized", 1, fxc_unpol_rho, true),
        lda_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            flat_pairs(fxc_pol_pairs),
            true,
        ),
    ];
    write_cases(outdir, name, &cases);
}

/// Snapshot one GGA functional: vxc on the extreme/edge points (both spins),
/// then fxc on a curated interior point set (both spins). `sigma0_ok` adds the
/// exact-σ=0 fxc points (only where libxc is accurate there — see
/// [`sigma0_pinnable`]).
#[cfg(feature = "libxc-ffi")]
fn gen_gga(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    outdir: &std::path::Path,
    name: &str,
    sigma0_ok: bool,
) {
    let id = xc.number(name);
    assert!(id > 0, "libxc does not know `{name}` (got id {id})");

    // --- vxc: extreme points (unchanged from the original snapshots) ---
    let unpol: &[(f64, f64)] = &[
        (1e-16, 0.0),
        (1e-15, 0.0),
        (1e-10, 1e-25),
        (1e-4, 1e-8),
        (0.1, 0.0),
        (0.1, 0.01),
        (0.5, 0.1),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 5.0),
        (10.0, 50.0),
        (100.0, 1e3),
        (1000.0, 1e6),
    ];
    let pol_all: &[(f64, f64, f64, f64, f64)] = &[
        (0.5, 0.5, 0.1, 0.05, 0.1),
        (0.7, 0.3, 0.2, 0.1, 0.05),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.3, 0.0, 0.0),
        (1.0, 1e-4, 0.2, 0.0, 1e-6),
        (0.6, 0.3, 0.1, 0.05, 0.08),
        (3.0, 2.0, 1.0, 0.5, 0.8),
        (0.5, 0.5, 0.1, 10.0, 0.1),
        (0.5, 0.5, 0.1, -10.0, 0.1),
        (1e-13, 1e-14, 1e-26, 0.0, 1e-28),
        (100.0, 50.0, 1e3, 500.0, 800.0),
    ];
    // ωB97X-V's B97 correlation uses the *raw* per-spin reduced gradient
    // x_σ² = σ_σσ/n_σ^(8/3) (the M06-family form, not the (n_σ/n)-weighted
    // combiner), so its exact-full-polarization floored-edge minority
    // derivatives are the same analytic-vs-AD artifact class as
    // `mgga_vxc_drop_full_pol` (FFI-measured rel ~1e-4 on the floored
    // minority vrho/vsigma; the *energy* and the physical near-edge
    // (1.0, 1e-4) point match ≤ 1e-10 and stay pinned). Its extreme-low-density
    // point (n ≈ 1.1e-13) hits the divergence-D class on vsigma (rel ~5e-9,
    // PW6B95-precedent), so it is floored like `mgga_vxc_dens_floor`.
    // The B97 power-series functionals (gga_xc_b97_3c and the constructor-corpus
    // hyb_gga_xc_b97) share ωB97X-V's raw per-spin x_σ² B97 correlation, so the
    // same exact-full-polarization floored-edge and extreme-low-density
    // exclusions apply (FFI-measured: minority vrho rel ~1e-4…2e-5 at the
    // floored n_σ = 0 edge; energy and the physical near-edge points match).
    let drop_full_pol = matches!(
        name,
        "hyb_gga_xc_wb97x_v" | "gga_xc_b97_3c" | "hyb_gga_xc_b97"
    );
    let dens_floor = match name {
        "hyb_gga_xc_wb97x_v" | "gga_xc_b97_3c" | "hyb_gga_xc_b97" => 1e-12,
        _ => 0.0,
    };
    let pol: Vec<(f64, f64, f64, f64, f64)> = pol_all
        .iter()
        .copied()
        .filter(|p| p.0 + p.1 >= dens_floor)
        .filter(|p| !(drop_full_pol && (p.0 == 0.0 || p.1 == 0.0)))
        .collect();
    let pol = &pol[..];

    // --- fxc: curated interior points + the small-σ band the sqrt-free per-spin
    // reduced gradient fixed. Moderate polarization (no |ζ| → 1), physical
    // σ_ab (|σ_ab| ≤ √(σ_aa σ_bb)). The large-σ points still have small *reduced*
    // gradient (σ scales as n^(8/3)), so they stay inside the fxc-finite domain.
    //
    // The harness now carries every reduced gradient **squared/sqrt-free**, so
    // `v2sigma2` is accurate as σ → 0 (old divergence #4 — the √σ trap — is gone).
    // σ = 1e-6 / 1e-8 are pinned for *all* GGA/hybrids as the regression lock.
    // Exact σ = 0 is pinned only where libxc itself is accurate there
    // (`sigma0_ok`): PBE-x/PBE-c/LYP/PBEH match to ≤1e-10. B88 and the
    // B88-containing b3lyp/b3lyp5 are excluded at *exact* σ = 0 — libxc's analytic
    // B88 `v2sigma2` truncates to (5/8)× the true limit at its σ-floor (an
    // FFI-measured libxc artifact; xcx is the accurate one — the new, far narrower
    // divergence #4 in docs/api-convention.md §8). ---
    let mut fxc_unpol: Vec<(f64, f64)> = vec![
        (0.1, 0.01),
        (0.5, 0.1),
        (1.0, 0.3),
        (2.0, 1.5),
        (5.0, 8.0),
        (10.0, 50.0),
        (100.0, 1e3),
        (0.3, 0.02),
        // small-σ band (divergence-#4 fix; the crossover used to be σ ≲ 1e-6)
        (0.3, 1e-6),
        (1.0, 1e-6),
        (1.0, 1e-8),
        (0.5, 1e-8),
    ];
    let mut fxc_pol: Vec<(f64, f64, f64, f64, f64)> = vec![
        (0.6, 0.3, 0.1, 0.04, 0.08),
        (1.0, 0.7, 0.3, 0.1, 0.2),
        (2.0, 1.5, 1.0, 0.4, 0.8),
        (0.5, 0.5, 0.1, 0.0, 0.1),
        (3.0, 2.0, 1.0, 0.5, 0.8),
        (0.9, 0.6, 0.2, 0.05, 0.15),
        (100.0, 50.0, 1e3, 500.0, 800.0),
        // small-σ band, per spin (σ_ab = 0 drives both per-spin gradients → 0)
        (0.6, 0.4, 1e-6, 0.0, 1e-6),
        (0.6, 0.4, 1e-8, 0.0, 1e-8),
        (1.0, 0.7, 1e-8, 0.0, 1e-6),
    ];
    // Exact σ = 0 only where libxc's analytic fxc is accurate there.
    if sigma0_ok {
        fxc_unpol.push((0.3, 0.0));
        fxc_unpol.push((1.0, 0.0));
        fxc_pol.push((0.6, 0.4, 0.0, 0.0, 0.0));
        fxc_pol.push((1.0, 0.7, 0.0, 0.0, 0.0));
    }
    let fxc_unpol = &fxc_unpol[..];
    let fxc_pol = &fxc_pol[..];

    let unpol_rho: Vec<f64> = unpol.iter().map(|&(r, _)| r).collect();
    let unpol_sigma: Vec<f64> = unpol.iter().map(|&(_, s)| s).collect();
    let pol_rho = flat_pol_rho(pol);
    let pol_sigma = flat_pol_sigma(pol);
    let fxc_unpol_rho: Vec<f64> = fxc_unpol.iter().map(|&(r, _)| r).collect();
    let fxc_unpol_sigma: Vec<f64> = fxc_unpol.iter().map(|&(_, s)| s).collect();
    let fxc_pol_rho = flat_pol_rho(fxc_pol);
    let fxc_pol_sigma = flat_pol_sigma(fxc_pol);

    let cases = vec![
        gga_case(
            xc,
            version,
            name,
            id,
            "unpolarized",
            1,
            unpol_rho,
            unpol_sigma,
            false,
        ),
        gga_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            pol_rho,
            pol_sigma,
            false,
        ),
        gga_case(
            xc,
            version,
            name,
            id,
            "unpolarized",
            1,
            fxc_unpol_rho,
            fxc_unpol_sigma,
            true,
        ),
        gga_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            fxc_pol_rho,
            fxc_pol_sigma,
            true,
        ),
    ];
    write_cases(outdir, name, &cases);
}

/// Flatten polarized `(n_a, n_b, σ_aa, σ_ab, σ_bb)` rows into packed `rho`.
#[cfg(feature = "libxc-ffi")]
fn flat_pol_rho(pol: &[(f64, f64, f64, f64, f64)]) -> Vec<f64> {
    pol.iter().flat_map(|&(a, b, _, _, _)| [a, b]).collect()
}

/// Flatten polarized rows into packed `sigma` `[σ_aa, σ_ab, σ_bb]`.
#[cfg(feature = "libxc-ffi")]
fn flat_pol_sigma(pol: &[(f64, f64, f64, f64, f64)]) -> Vec<f64> {
    pol.iter()
        .flat_map(|&(_, _, saa, sab, sbb)| [saa, sab, sbb])
        .collect()
}

/// Build one meta-GGA `GoldenCase`: libxc energy + vxc (vrho/vsigma/vtau) always,
/// plus the six non-Laplacian fxc blocks when `want_fxc`. A zeroed `lapl` array is
/// supplied (TPSS is `needs_lapl = false`); `tau` is stored in the snapshot so the
/// golden test can rebuild the input.
#[cfg(feature = "libxc-ffi")]
#[allow(clippy::too_many_arguments)]
fn mgga_case(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    name: &str,
    id: i32,
    spin: &str,
    nspin: i32,
    rho: Vec<f64>,
    sigma: Vec<f64>,
    tau: Vec<f64>,
    want_fxc: bool,
) -> xcx_validation::GoldenCase {
    use xcx_validation::GoldenCase;
    let np = rho.len() / nspin as usize;
    let lapl = vec![0.0; np * nspin as usize];
    let (exc, vrho, vsigma, _vlapl, vtau) =
        xc.mgga_exc_vxc(id, nspin, np, &rho, &sigma, &lapl, &tau);
    let (v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau, v2tau2) = if want_fxc {
        xc.mgga_fxc(id, nspin, np, &rho, &sigma, &lapl, &tau)
    } else {
        (vec![], vec![], vec![], vec![], vec![], vec![])
    };
    GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: spin.into(),
        np,
        rho,
        sigma,
        tau,
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
    }
}

/// Flatten meta-GGA polarized `(n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b)` rows into
/// packed `(rho, sigma, tau)`.
#[cfg(feature = "libxc-ffi")]
#[allow(clippy::type_complexity)]
fn flat_mgga_pol(pol: &[[f64; 7]]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let rho = pol.iter().flat_map(|x| [x[0], x[1]]).collect();
    let sigma = pol.iter().flat_map(|x| [x[2], x[3], x[4]]).collect();
    let tau = pol.iter().flat_map(|x| [x[5], x[6]]).collect();
    (rho, sigma, tau)
}

/// Snapshot one meta-GGA functional: vxc on extreme/edge points (full
/// polarization, σ_ab clamp, large σ, α < 0 from τ ≪ τ_W, large α) then fxc on a
/// **physical** interior set (both spins). The fxc set keeps σ ≥ 1e-6 and a
/// physical τ (τ ≳ τ_W, well above the 1e-20 floor): it deliberately includes the
/// τ-ratio hazard regions that *do* match libxc — α near 0 (τ ≈ τ_W), α near 1,
/// and small σ down to 1e-6 — but excludes exact σ = 0 (libxc's analytic
/// `v2sigma2` is a σ-floor artifact there, the TPSS analogue of divergence #4) and
/// the τ-floor corner (non-physical, huge in both libraries — divergence-C class).
#[cfg(feature = "libxc-ffi")]
fn gen_mgga(xc: &xcx_validation::ffi::Libxc, version: &str, outdir: &std::path::Path, name: &str) {
    let id = xc.number(name);
    assert!(id > 0, "libxc does not know `{name}` (got id {id})");

    // --- vxc: extreme/edge points (n, σ, τ) ---
    // Drop points below this functional's low-density vxc floor (only r2SCAN-c has
    // one; see `mgga_vxc_dens_floor`).
    let floor = mgga_vxc_dens_floor(name);
    let unpol_all: &[(f64, f64, f64)] = &[
        (1e-14, 0.0, 1e-15),
        (1e-10, 1e-25, 1e-12),
        (1e-4, 1e-8, 1e-6),
        (0.1, 0.0, 0.05),
        (0.1, 0.01, 0.05),
        (0.5, 0.1, 0.3),
        (1.0, 0.0, 0.8),
        (1.0, 1.0, 0.8),
        (1.0, 5.0, 0.6),   // τ < τ_W ⇒ α < 0
        (1.0, 0.2, 100.0), // α ≫ 1
        (2.0, 5.0, 3.0),
        (10.0, 50.0, 20.0),
        (100.0, 1e3, 200.0),
        (1000.0, 1e6, 2e3),
    ];
    // (n_a, n_b, σ_aa, σ_ab, σ_bb, τ_a, τ_b)
    let pol_all: &[[f64; 7]] = &[
        [0.5, 0.5, 0.1, 0.05, 0.1, 0.4, 0.4],
        [0.7, 0.3, 0.2, 0.1, 0.05, 0.5, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], // full polarization
        [1.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0],
        [1.0, 1e-4, 0.2, 0.0, 1e-6, 0.6, 1e-5], // near full pol
        [0.6, 0.3, 0.1, 0.05, 0.08, 0.4, 0.25],
        [3.0, 2.0, 1.0, 0.5, 0.8, 4.0, 3.0],
        [0.5, 0.5, 0.1, 10.0, 0.1, 0.5, 0.5],  // σ_ab clamp hi
        [0.5, 0.5, 0.1, -10.0, 0.1, 0.5, 0.5], // σ_ab clamp lo
        // low densities, physical reduced gradient + τ. Kept at 1e-8 (not 1e-13):
        // below ~1e-9 the TPSS-c τ-derivative hits an extreme-rs analytic/AD
        // cancellation wall (|Δ| crosses 1e-10), the correlation analogue of
        // divergence #1 — not a tau-hazard. Finiteness to 1e-14 is the fuzz gate's
        // job (libxc-free); golden pins where both libraries agree ≤ 1e-10.
        [1e-8, 1e-9, 2e-20, 0.0, 4e-23, 7e-13, 1e-14],
        [100.0, 50.0, 1e3, 500.0, 800.0, 2e3, 1e3],
    ];
    // Apply the per-functional low-density vxc floor (drops n < floor; only
    // r2SCAN-c sets one — see `mgga_vxc_dens_floor`). Others keep every point.
    let unpol: Vec<(f64, f64, f64)> = unpol_all
        .iter()
        .copied()
        .filter(|&(n, _, _)| n >= floor)
        .collect();
    // Also drop the exact full-polarization points (a spin channel literally 0)
    // for functionals whose minority-channel derivatives there are an
    // analytic-vs-AD floor artifact (only mgga_c_m06_l — see
    // `mgga_vxc_drop_full_pol`); the physical near-full-pol point is retained.
    let drop_full_pol = mgga_vxc_drop_full_pol(name);
    let pol: Vec<[f64; 7]> = pol_all
        .iter()
        .copied()
        .filter(|p| p[0] + p[1] >= floor)
        .filter(|p| !(drop_full_pol && (p[0] == 0.0 || p[1] == 0.0)))
        .collect();
    let unpol = &unpol[..];
    let pol = &pol[..];

    // --- fxc: physical interior set (σ ≥ 1e-6, physical τ; α spans 0…≫1) ---
    let mut fxc_unpol: Vec<(f64, f64, f64)> = vec![
        (0.5, 0.1, 0.3),
        (1.0, 0.3, 0.8),
        (2.0, 1.5, 1.5),
        (5.0, 8.0, 10.0),
        (0.3, 0.02, 0.2),
        (10.0, 50.0, 30.0),
        (100.0, 1e3, 300.0),
        (1.0, 0.4, 0.06), // α ≈ 0 (τ ≈ τ_W) — r2SCAN switch seam
        (1.0, 0.4, 4.6),  // α ≈ 1
        (2.0, 1.0, 8.0),  // α > 1
        (1.0, 0.3, 10.0), // α ≈ 2.2 — just below the r2SCAN α = 2.5 switch seam (poly side)
        (1.0, 0.2, 13.0), // α ≈ 2.85 — just above the seam (exp-tail side)
        (1.0, 0.2, 50.0), // α ≫ 2.5 — deep in the large-α exp tail
        (1.0, 1e-5, 0.8), // small σ band (TPSS-x analytic v2sigma2 degrades < ~1e-6)
        (0.3, 1e-5, 0.2),
    ];
    let mut fxc_pol: Vec<[f64; 7]> = vec![
        [0.6, 0.3, 0.10, 0.04, 0.08, 0.40, 0.25],
        [1.0, 0.7, 0.30, 0.10, 0.20, 1.20, 0.90],
        [2.0, 1.5, 1.00, 0.40, 0.80, 3.00, 2.00],
        [0.5, 0.5, 0.10, 0.00, 0.10, 0.40, 0.40],
        [3.0, 2.0, 1.00, 0.50, 0.80, 5.00, 3.00],
        [0.9, 0.6, 0.20, 0.05, 0.15, 0.70, 0.50],
        [100.0, 50.0, 1e3, 500.0, 800.0, 2e3, 1e3],
        [1.0, 0.7, 0.20, 0.05, 0.15, 10.0, 7.0], // α ≈ 2…3 per channel (r2SCAN seam, pol)
        [0.6, 0.4, 1e-5, 0.0, 1e-5, 0.50, 0.30], // small σ band per spin
        [1.0, 0.7, 1e-5, 0.0, 1e-5, 0.80, 0.60],
    ];
    // Extend the small-σ band down to σ = 1e-8 and exact σ = 0 for functionals
    // whose *every* fxc block stays ≤1e-10 vs libxc there (FFI-established —
    // `mgga_fxc_smallsigma_to_zero`; currently mgga_c_tpss). This tightens the
    // formerly-conservative σ ≥ 1e-5 meta-GGA pin to match the GGA small-σ band
    // (and exact 0), locking the σ → 0 behavior as a regression net. TPSS-x keeps
    // σ ≥ 1e-5 only because its `v2sigma2` block alone degrades on libxc's side
    // below ~1e-6 (divergence-#4 class); its other five blocks are clean to 0.
    if mgga_fxc_smallsigma_to_zero(name) {
        fxc_unpol.push((1.0, 1e-8, 0.8));
        fxc_unpol.push((0.3, 1e-8, 0.2));
        fxc_unpol.push((1.0, 0.0, 0.8)); // exact σ = 0
        fxc_unpol.push((0.3, 0.0, 0.2));
        fxc_pol.push([0.6, 0.4, 1e-8, 0.0, 1e-8, 0.50, 0.30]);
        fxc_pol.push([1.0, 0.7, 1e-8, 0.0, 1e-8, 0.80, 0.60]);
        fxc_pol.push([0.6, 0.4, 0.0, 0.0, 0.0, 0.50, 0.30]); // exact σ = 0
        fxc_pol.push([1.0, 0.7, 0.0, 0.0, 0.0, 0.80, 0.60]);
    }
    let fxc_unpol = &fxc_unpol[..];
    let fxc_pol = &fxc_pol[..];

    let unpol_rho: Vec<f64> = unpol.iter().map(|&(r, _, _)| r).collect();
    let unpol_sigma: Vec<f64> = unpol.iter().map(|&(_, s, _)| s).collect();
    let unpol_tau: Vec<f64> = unpol.iter().map(|&(_, _, t)| t).collect();
    let (pol_rho, pol_sigma, pol_tau) = flat_mgga_pol(pol);
    let fxc_unpol_rho: Vec<f64> = fxc_unpol.iter().map(|&(r, _, _)| r).collect();
    let fxc_unpol_sigma: Vec<f64> = fxc_unpol.iter().map(|&(_, s, _)| s).collect();
    let fxc_unpol_tau: Vec<f64> = fxc_unpol.iter().map(|&(_, _, t)| t).collect();
    let (fxc_pol_rho, fxc_pol_sigma, fxc_pol_tau) = flat_mgga_pol(fxc_pol);

    let cases = vec![
        mgga_case(
            xc,
            version,
            name,
            id,
            "unpolarized",
            1,
            unpol_rho,
            unpol_sigma,
            unpol_tau,
            false,
        ),
        mgga_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            pol_rho,
            pol_sigma,
            pol_tau,
            false,
        ),
        mgga_case(
            xc,
            version,
            name,
            id,
            "unpolarized",
            1,
            fxc_unpol_rho,
            fxc_unpol_sigma,
            fxc_unpol_tau,
            true,
        ),
        mgga_case(
            xc,
            version,
            name,
            id,
            "polarized",
            2,
            fxc_pol_rho,
            fxc_pol_sigma,
            fxc_pol_tau,
            true,
        ),
    ];
    write_cases(outdir, name, &cases);
}

/// Accumulate `dst += w·src` (resizing to fit — components may produce more
/// blocks than allocated so far, e.g. a GGA component inside a meta-GGA mix).
#[cfg(feature = "libxc-ffi")]
fn acc(dst: &mut Vec<f64>, w: f64, src: &[f64]) {
    if src.len() > dst.len() {
        dst.resize(src.len(), 0.0);
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += w * s;
    }
}

/// Snapshot a GGA-family double-hybrid semilocal core as the weighted sum of
/// libxc components (each `(weight, name, ext_params)`). Mirrors [`gen_gga`]'s
/// point sets exactly (vxc extremes + fxc interior/small-σ band; `sigma0_ok`
/// as in [`sigma0_pinnable`]). `xcx_id` is the xcx-private id recorded in the
/// snapshot; `libxc_version` is tagged `"(component mix)"`.
#[cfg(feature = "libxc-ffi")]
fn gen_gga_mix(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    outdir: &std::path::Path,
    name: &str,
    xcx_id: u32,
    parts: &[(f64, &str, Option<&[f64]>)],
    sigma0_ok: bool,
) {
    use xcx_validation::GoldenCase;

    // vxc extreme points — identical to gen_gga.
    let unpol: &[(f64, f64)] = &[
        (1e-16, 0.0),
        (1e-15, 0.0),
        (1e-10, 1e-25),
        (1e-4, 1e-8),
        (0.1, 0.0),
        (0.1, 0.01),
        (0.5, 0.1),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 5.0),
        (10.0, 50.0),
        (100.0, 1e3),
        (1000.0, 1e6),
    ];
    let pol: &[(f64, f64, f64, f64, f64)] = &[
        (0.5, 0.5, 0.1, 0.05, 0.1),
        (0.7, 0.3, 0.2, 0.1, 0.05),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.3, 0.0, 0.0),
        (1.0, 1e-4, 0.2, 0.0, 1e-6),
        (0.6, 0.3, 0.1, 0.05, 0.08),
        (3.0, 2.0, 1.0, 0.5, 0.8),
        (0.5, 0.5, 0.1, 10.0, 0.1),
        (0.5, 0.5, 0.1, -10.0, 0.1),
        (1e-13, 1e-14, 1e-26, 0.0, 1e-28),
        (100.0, 50.0, 1e3, 500.0, 800.0),
    ];
    // fxc interior + small-σ band — identical to gen_gga.
    let mut fxc_unpol: Vec<(f64, f64)> = vec![
        (0.1, 0.01),
        (0.5, 0.1),
        (1.0, 0.3),
        (2.0, 1.5),
        (5.0, 8.0),
        (10.0, 50.0),
        (100.0, 1e3),
        (0.3, 0.02),
        (0.3, 1e-6),
        (1.0, 1e-6),
        (1.0, 1e-8),
        (0.5, 1e-8),
    ];
    let mut fxc_pol: Vec<(f64, f64, f64, f64, f64)> = vec![
        (0.6, 0.3, 0.1, 0.04, 0.08),
        (1.0, 0.7, 0.3, 0.1, 0.2),
        (2.0, 1.5, 1.0, 0.4, 0.8),
        (0.5, 0.5, 0.1, 0.0, 0.1),
        (3.0, 2.0, 1.0, 0.5, 0.8),
        (0.9, 0.6, 0.2, 0.05, 0.15),
        (100.0, 50.0, 1e3, 500.0, 800.0),
        (0.6, 0.4, 1e-6, 0.0, 1e-6),
        (0.6, 0.4, 1e-8, 0.0, 1e-8),
        (1.0, 0.7, 1e-8, 0.0, 1e-6),
    ];
    if sigma0_ok {
        fxc_unpol.push((0.3, 0.0));
        fxc_unpol.push((1.0, 0.0));
        fxc_pol.push((0.6, 0.4, 0.0, 0.0, 0.0));
        fxc_pol.push((1.0, 0.7, 0.0, 0.0, 0.0));
    }

    let mix_case = |spin: &str, nspin: i32, rho: Vec<f64>, sigma: Vec<f64>, want_fxc: bool| {
        let np = rho.len() / nspin as usize;
        let mut exc = vec![0.0; np];
        let mut vrho = Vec::new();
        let mut vsigma = Vec::new();
        let mut v2rho2 = Vec::new();
        let mut v2rhosigma = Vec::new();
        let mut v2sigma2 = Vec::new();
        for &(w, comp, ext) in parts {
            let id = xc.number(comp);
            assert!(id > 0, "libxc does not know `{comp}` (got id {id})");
            let (e, vr, vs) = xc.gga_exc_vxc_ext(id, nspin, np, &rho, &sigma, ext);
            acc(&mut exc, w, &e);
            acc(&mut vrho, w, &vr);
            acc(&mut vsigma, w, &vs);
            if want_fxc {
                let (r2, rs, s2) = xc.gga_fxc_ext(id, nspin, np, &rho, &sigma, ext);
                acc(&mut v2rho2, w, &r2);
                acc(&mut v2rhosigma, w, &rs);
                acc(&mut v2sigma2, w, &s2);
            }
        }
        GoldenCase {
            functional: name.into(),
            libxc_id: xcx_id,
            libxc_version: format!("{version} (component mix)"),
            spin: spin.into(),
            np,
            rho,
            sigma,
            tau: vec![],
            exc,
            vrho,
            vsigma,
            vtau: vec![],
            v2rho2,
            v2rhosigma,
            v2sigma2,
            v2rhotau: vec![],
            v2sigmatau: vec![],
            v2tau2: vec![],
        }
    };

    let cases = vec![
        mix_case(
            "unpolarized",
            1,
            unpol.iter().map(|&(r, _)| r).collect(),
            unpol.iter().map(|&(_, s)| s).collect(),
            false,
        ),
        mix_case(
            "polarized",
            2,
            flat_pol_rho(pol),
            flat_pol_sigma(pol),
            false,
        ),
        mix_case(
            "unpolarized",
            1,
            fxc_unpol.iter().map(|&(r, _)| r).collect(),
            fxc_unpol.iter().map(|&(_, s)| s).collect(),
            true,
        ),
        mix_case(
            "polarized",
            2,
            flat_pol_rho(&fxc_pol),
            flat_pol_sigma(&fxc_pol),
            true,
        ),
    ];
    write_cases(outdir, name, &cases);
}

/// Snapshot a meta-GGA-family double-hybrid semilocal core as the weighted sum
/// of libxc components (GGA or meta-GGA, with ext-params overrides). Mirrors
/// [`gen_mgga`]'s point sets, applying PW6B95's exclusions (drop exact
/// full-polarization rows; density floor 1e-12 — the B95 floored-edge class).
#[cfg(feature = "libxc-ffi")]
fn gen_mgga_mix(
    xc: &xcx_validation::ffi::Libxc,
    version: &str,
    outdir: &std::path::Path,
    name: &str,
    xcx_id: u32,
    parts: &[(f64, &str, MixExt)],
) {
    use xcx_validation::GoldenCase;

    let floor = 1e-12;
    let unpol: Vec<(f64, f64, f64)> = vec![
        (1e-10, 1e-25, 1e-12),
        (1e-4, 1e-8, 1e-6),
        (0.1, 0.0, 0.05),
        (0.1, 0.01, 0.05),
        (0.5, 0.1, 0.3),
        (1.0, 0.0, 0.8),
        (1.0, 1.0, 0.8),
        (1.0, 5.0, 0.6),
        (1.0, 0.2, 100.0),
        (2.0, 5.0, 3.0),
        (10.0, 50.0, 20.0),
        (100.0, 1e3, 200.0),
        (1000.0, 1e6, 2e3),
    ];
    let pol: Vec<[f64; 7]> = vec![
        [0.5, 0.5, 0.1, 0.05, 0.1, 0.4, 0.4],
        [0.7, 0.3, 0.2, 0.1, 0.05, 0.5, 0.2],
        [1.0, 1e-4, 0.2, 0.0, 1e-6, 0.6, 1e-5],
        [0.6, 0.3, 0.1, 0.05, 0.08, 0.4, 0.25],
        [3.0, 2.0, 1.0, 0.5, 0.8, 4.0, 3.0],
        [0.5, 0.5, 0.1, 10.0, 0.1, 0.5, 0.5],
        [0.5, 0.5, 0.1, -10.0, 0.1, 0.5, 0.5],
        [1e-8, 1e-9, 2e-20, 0.0, 4e-23, 7e-13, 1e-14],
        [100.0, 50.0, 1e3, 500.0, 800.0, 2e3, 1e3],
    ];
    let _ = floor; // point sets above already respect the floor / full-pol drop
    let fxc_unpol: Vec<(f64, f64, f64)> = vec![
        (0.5, 0.1, 0.3),
        (1.0, 0.3, 0.8),
        (2.0, 1.5, 1.5),
        (5.0, 8.0, 10.0),
        (0.3, 0.02, 0.2),
        (10.0, 50.0, 30.0),
        (100.0, 1e3, 300.0),
        (1.0, 0.4, 0.06),
        (1.0, 0.4, 4.6),
        (2.0, 1.0, 8.0),
        (1.0, 0.3, 10.0),
        (1.0, 0.2, 13.0),
        (1.0, 0.2, 50.0),
        (1.0, 1e-5, 0.8),
        (0.3, 1e-5, 0.2),
    ];
    let fxc_pol: Vec<[f64; 7]> = vec![
        [0.6, 0.3, 0.10, 0.04, 0.08, 0.40, 0.25],
        [1.0, 0.7, 0.30, 0.10, 0.20, 1.20, 0.90],
        [2.0, 1.5, 1.00, 0.40, 0.80, 3.00, 2.00],
        [0.5, 0.5, 0.10, 0.00, 0.10, 0.40, 0.40],
        [3.0, 2.0, 1.00, 0.50, 0.80, 5.00, 3.00],
        [0.9, 0.6, 0.20, 0.05, 0.15, 0.70, 0.50],
        [100.0, 50.0, 1e3, 500.0, 800.0, 2e3, 1e3],
        [1.0, 0.7, 0.20, 0.05, 0.15, 10.0, 7.0],
        [0.6, 0.4, 1e-5, 0.0, 1e-5, 0.50, 0.30],
        [1.0, 0.7, 1e-5, 0.0, 1e-5, 0.80, 0.60],
    ];

    let mix_case =
        |spin: &str, nspin: i32, rho: Vec<f64>, sigma: Vec<f64>, tau: Vec<f64>, want_fxc: bool| {
            let np = rho.len() / nspin as usize;
            let lapl = vec![0.0; np * nspin as usize];
            let mut exc = vec![0.0; np];
            let mut vrho = Vec::new();
            let mut vsigma = Vec::new();
            let mut vtau = Vec::new();
            let mut v2rho2 = Vec::new();
            let mut v2rhosigma = Vec::new();
            let mut v2sigma2 = Vec::new();
            let mut v2rhotau = Vec::new();
            let mut v2sigmatau = Vec::new();
            let mut v2tau2 = Vec::new();
            for (w, comp, ext) in parts {
                let id = xc.number(comp);
                assert!(id > 0, "libxc does not know `{comp}` (got id {id})");
                match ext {
                    MixExt::Gga(p) => {
                        let (e, vr, vs) = xc.gga_exc_vxc_ext(id, nspin, np, &rho, &sigma, Some(p));
                        acc(&mut exc, *w, &e);
                        acc(&mut vrho, *w, &vr);
                        acc(&mut vsigma, *w, &vs);
                        if want_fxc {
                            let (r2, rs, s2) = xc.gga_fxc_ext(id, nspin, np, &rho, &sigma, Some(p));
                            acc(&mut v2rho2, *w, &r2);
                            acc(&mut v2rhosigma, *w, &rs);
                            acc(&mut v2sigma2, *w, &s2);
                        }
                    }
                    MixExt::Mgga(p) => {
                        let (e, vr, vs, _vl, vt) =
                            xc.mgga_exc_vxc_ext(id, nspin, np, &rho, &sigma, &lapl, &tau, Some(p));
                        acc(&mut exc, *w, &e);
                        acc(&mut vrho, *w, &vr);
                        acc(&mut vsigma, *w, &vs);
                        acc(&mut vtau, *w, &vt);
                        if want_fxc {
                            let (r2, rs, s2, rt, st, t2) =
                                xc.mgga_fxc_ext(id, nspin, np, &rho, &sigma, &lapl, &tau, Some(p));
                            acc(&mut v2rho2, *w, &r2);
                            acc(&mut v2rhosigma, *w, &rs);
                            acc(&mut v2sigma2, *w, &s2);
                            acc(&mut v2rhotau, *w, &rt);
                            acc(&mut v2sigmatau, *w, &st);
                            acc(&mut v2tau2, *w, &t2);
                        }
                    }
                }
            }
            // a GGA component leaves the τ blocks short/empty: pad to the meta-GGA
            // lengths so the golden test compares the full (zero-extended) tensors
            let ns = nspin as usize;
            let nsig = 2 * ns - 1;
            vtau.resize(np * ns, 0.0);
            if want_fxc {
                v2rhotau.resize(np * ns * ns, 0.0);
                v2sigmatau.resize(np * nsig * ns, 0.0);
                v2tau2.resize(np * ns * (ns + 1) / 2, 0.0);
            }
            GoldenCase {
                functional: name.into(),
                libxc_id: xcx_id,
                libxc_version: format!("{version} (component mix)"),
                spin: spin.into(),
                np,
                rho,
                sigma,
                tau,
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
            }
        };

    let (pol_rho, pol_sigma, pol_tau) = flat_mgga_pol(&pol);
    let (fxc_pol_rho, fxc_pol_sigma, fxc_pol_tau) = flat_mgga_pol(&fxc_pol);
    let cases = vec![
        mix_case(
            "unpolarized",
            1,
            unpol.iter().map(|&(r, _, _)| r).collect(),
            unpol.iter().map(|&(_, s, _)| s).collect(),
            unpol.iter().map(|&(_, _, t)| t).collect(),
            false,
        ),
        mix_case("polarized", 2, pol_rho, pol_sigma, pol_tau, false),
        mix_case(
            "unpolarized",
            1,
            fxc_unpol.iter().map(|&(r, _, _)| r).collect(),
            fxc_unpol.iter().map(|&(_, s, _)| s).collect(),
            fxc_unpol.iter().map(|&(_, _, t)| t).collect(),
            true,
        ),
        mix_case(
            "polarized",
            2,
            fxc_pol_rho,
            fxc_pol_sigma,
            fxc_pol_tau,
            true,
        ),
    ];
    write_cases(outdir, name, &cases);
}

#[cfg(feature = "libxc-ffi")]
fn write_cases(outdir: &std::path::Path, name: &str, cases: &[xcx_validation::GoldenCase]) {
    let json = serde_json::to_string_pretty(cases).unwrap();
    let path = outdir.join(format!("{name}.json"));
    std::fs::write(&path, json).unwrap();
    eprintln!("wrote {}", path.display());
}

#[cfg(not(feature = "libxc-ffi"))]
fn main() {
    eprintln!(
        "gen_golden requires `--features libxc-ffi` and a libxc shared library \
         (set XCX_LIBXC_DLL or CONDA_PREFIX). See crates/xcx-validation/README.md."
    );
}
