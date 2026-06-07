// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Regenerate golden snapshots from a conda-forge libxc.
//!
//! ```text
//! conda create -n xcref -c conda-forge libxc
//! $env:XCX_LIBXC_DLL = "<env>\Library\bin\xc.dll"   # or set CONDA_PREFIX
//! cargo run -p xcx-validation --features libxc-ffi --bin gen_golden
//! ```
//!
//! The committed snapshots under `testdata/` are then used by the (default,
//! libxc-free) golden test in CI.

#[cfg(feature = "libxc-ffi")]
fn main() {
    use std::path::Path;
    use xcx_validation::ffi::Libxc;

    let xc = Libxc::load();
    let (vmaj, vmin, vmic) = xc.version();
    let version = format!("{vmaj}.{vmin}.{vmic}");
    eprintln!("libxc version {version}");

    let outdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    std::fs::create_dir_all(&outdir).unwrap();

    gen_lda(&xc, &version, &outdir, "lda_x");
    gen_lda(&xc, &version, &outdir, "lda_c_pw");
    gen_lda(&xc, &version, &outdir, "lda_c_vwn");
    gen_lda(&xc, &version, &outdir, "lda_c_vwn_3");
    gen_lda(&xc, &version, &outdir, "lda_c_vwn_rpa");
    gen_gga(&xc, &version, &outdir, "gga_x_pbe");
    gen_gga(&xc, &version, &outdir, "gga_x_b88");
    gen_gga(&xc, &version, &outdir, "gga_c_pbe");
    gen_gga(&xc, &version, &outdir, "gga_c_lyp");
    gen_gga(&xc, &version, &outdir, "hyb_gga_xc_pbeh");
    gen_gga(&xc, &version, &outdir, "hyb_gga_xc_b3lyp5");
    gen_gga(&xc, &version, &outdir, "hyb_gga_xc_b3lyp");
}

/// Snapshot one LDA functional in both spin modes, including screening / full
/// polarization edge cases.
#[cfg(feature = "libxc-ffi")]
fn gen_lda(xc: &xcx_validation::ffi::Libxc, version: &str, outdir: &std::path::Path, name: &str) {
    use xcx_validation::GoldenCase;

    let id = xc.number(name);
    assert!(id > 0, "libxc does not know `{name}` (got id {id})");

    // Unpolarized densities, spanning rho in [1e-16, 1e3] incl. screening edges.
    let unpol_rho: Vec<f64> = vec![
        1e-16, 1e-15, 1e-14, 1e-10, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0,
    ];
    let (exc, vrho) = xc.lda_exc_vxc(id, 1, unpol_rho.len(), &unpol_rho);
    let unpol = GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: "unpolarized".into(),
        np: unpol_rho.len(),
        rho: unpol_rho,
        sigma: vec![],
        exc,
        vrho,
        vsigma: vec![],
    };

    // Polarized (n_a, n_b) pairs incl. z = ±1 and sub-threshold channels.
    let pairs: &[(f64, f64)] = &[
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
    let mut pol_rho = Vec::with_capacity(pairs.len() * 2);
    for &(a, b) in pairs {
        pol_rho.push(a);
        pol_rho.push(b);
    }
    let np = pairs.len();
    let (exc, vrho) = xc.lda_exc_vxc(id, 2, np, &pol_rho);
    let pol = GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: "polarized".into(),
        np,
        rho: pol_rho,
        sigma: vec![],
        exc,
        vrho,
        vsigma: vec![],
    };

    let json = serde_json::to_string_pretty(&vec![unpol, pol]).unwrap();
    let path = outdir.join(format!("{name}.json"));
    std::fs::write(&path, json).unwrap();
    eprintln!("wrote {}", path.display());
}

/// Snapshot one GGA functional in both spin modes. Covers screening / full
/// polarization edges, σ = 0 (GGA→LDA limit), tiny and very large σ, and (for
/// the polarized set) σ_ab beyond ±s_ave to exercise libxc's work_gga clamp.
#[cfg(feature = "libxc-ffi")]
fn gen_gga(xc: &xcx_validation::ffi::Libxc, version: &str, outdir: &std::path::Path, name: &str) {
    use xcx_validation::GoldenCase;

    let id = xc.number(name);
    assert!(id > 0, "libxc does not know `{name}` (got id {id})");

    // Unpolarized (rho, sigma): screening edges, σ = 0, small/moderate/large σ.
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
    let mut urho = Vec::with_capacity(unpol.len());
    let mut usigma = Vec::with_capacity(unpol.len());
    for &(r, s) in unpol {
        urho.push(r);
        usigma.push(s);
    }
    let (exc, vrho, vsigma) = xc.gga_exc_vxc(id, 1, unpol.len(), &urho, &usigma);
    let unpol_case = GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: "unpolarized".into(),
        np: unpol.len(),
        rho: urho,
        sigma: usigma,
        exc,
        vrho,
        vsigma,
    };

    // Polarized (n_a, n_b, σ_aa, σ_ab, σ_bb): full polarization, σ = 0, minority
    // channel with σ>0, σ_ab beyond ±s_ave (clamp), and small/large edges.
    let pol: &[(f64, f64, f64, f64, f64)] = &[
        (0.5, 0.5, 0.1, 0.05, 0.1),
        (0.7, 0.3, 0.2, 0.1, 0.05),
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.3, 0.0, 0.0),
        // Near (not at) full polarization: minority channel unscreened, but n_b
        // kept at 1e-4 (not 1e-10). At n_b ≪ n libxc's *analytic* GGA-x minority
        // vrho cancels (FD-verified ~2.76e-8 error at 1e-10; xcx is correct);
        // see docs/api-convention.md §8, divergence B. 1e-4 still exercises the
        // minority path where libxc stays accurate.
        (1.0, 1e-4, 0.2, 0.0, 1e-6),
        (0.6, 0.3, 0.1, 0.05, 0.08),
        (3.0, 2.0, 1.0, 0.5, 0.8),
        (0.5, 0.5, 0.1, 10.0, 0.1),
        (0.5, 0.5, 0.1, -10.0, 0.1),
        (1e-13, 1e-14, 1e-26, 0.0, 1e-28),
        (100.0, 50.0, 1e3, 500.0, 800.0),
    ];
    let mut prho = Vec::with_capacity(pol.len() * 2);
    let mut psigma = Vec::with_capacity(pol.len() * 3);
    for &(a, b, saa, sab, sbb) in pol {
        prho.push(a);
        prho.push(b);
        psigma.push(saa);
        psigma.push(sab);
        psigma.push(sbb);
    }
    let np = pol.len();
    let (exc, vrho, vsigma) = xc.gga_exc_vxc(id, 2, np, &prho, &psigma);
    let pol_case = GoldenCase {
        functional: name.into(),
        libxc_id: id as u32,
        libxc_version: version.into(),
        spin: "polarized".into(),
        np,
        rho: prho,
        sigma: psigma,
        exc,
        vrho,
        vsigma,
    };

    let json = serde_json::to_string_pretty(&vec![unpol_case, pol_case]).unwrap();
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
