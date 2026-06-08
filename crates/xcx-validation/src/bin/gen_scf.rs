// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! SCF end-to-end gate (v0.1.0 DoD §6): turn PySCF full-grid dumps into the
//! committed SCF artifacts, with **pinned libxc 6.1.0 as the XC truth** (via FFI),
//! and run both verification layers.
//!
//! Requires `--features libxc-ffi` and the full-grid JSONs produced by
//! `python/gen_scf_ref.py` (run in WSL). PySCF is only the host (grid + converged
//! density + its own total/semilocal energies, with its bundled libxc 7.0.0); the
//! XC reference stays on the same pinned 6.1.0 the golden suite uses.
//!
//! Layer (a) — interface correctness: on PySCF's converged density+grid, compare
//! xcx vs libxc-6.1.0 component-wise (exc/vrho/vsigma). The full grid is checked
//! and a biased subset (density tails + near-nucleus cusp + high reduced gradient)
//! is written as `testdata/scf_grid_{case}.json` GoldenCases — so the libxc-free
//! golden test re-verifies xcx on real molecular-grid points in CI.
//!
//! Layer (b) — total energy: reconstruct E_tot by swapping only the *semilocal*
//! XC source on the converged density,
//!   E_tot(xcx) = E_tot(pyscf) − E_xc^sl(pyscf) + E_xc^sl(xcx),
//! and assert |ΔE_tot| < 1e-6 Eh. EXX (0.2·E_HFX) is held fixed and cancels — the
//! comparison is semilocal-to-semilocal. The residual also surfaces the libxc
//! 6.1.0-vs-7.0.0 version delta, which must stay ≪1e-6 (else the reconstruction
//! is mixing two libxc versions — stop and surface).
//!
//! Run: cargo run -p xcx-validation --features libxc-ffi --bin gen_scf

#[cfg(feature = "libxc-ffi")]
fn main() {
    imp::run();
}

#[cfg(not(feature = "libxc-ffi"))]
fn main() {
    eprintln!(
        "gen_scf requires `--features libxc-ffi` and the PySCF full-grid dumps \
         (run python/gen_scf_ref.py in WSL first)."
    );
}

#[cfg(feature = "libxc-ffi")]
mod imp {
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize};
    use xcx::{Functional, Spin, XcInput};
    use xcx_validation::ffi::Libxc;
    use xcx_validation::{rel_close, GoldenCase, ATOL, RTOL};

    /// libxc B3LYP id (VWN_RPA) — what both xcx and PySCF's `HYB_GGA_XC_B3LYP` use.
    const B3LYP_ID: i32 = 402;
    /// Target size of the committed biased real-grid subset, per case.
    const SUBSET: usize = 2000;
    /// DoD total-energy tolerance.
    const E_TOL: f64 = 1e-6;

    /// PySCF full-grid dump (subset of fields; serde ignores the rest).
    #[derive(Deserialize)]
    struct FullGrid {
        case: String,
        spin: String,
        molecule: String,
        basis: String,
        pyscf_version: String,
        libxc_version: String,
        libxc_id: u32,
        hybrid_coeff: f64,
        n_grid: usize,
        e_tot: f64,
        exc_sl_semilocal: f64,
        exx_energy: f64,
        weights: Vec<f64>,
        rho: Vec<f64>,
        sigma: Vec<f64>,
        exc_pyscf: Vec<f64>,
    }

    /// Committed record of the report-once layer-(b) run.
    #[derive(Serialize)]
    struct ScfScalars {
        case: String,
        molecule: String,
        basis: String,
        spin: String,
        n_grid: usize,
        pyscf_version: String,
        pyscf_libxc_version: String,
        xcx_libxc_version: String,
        libxc_id: u32,
        hybrid_coeff: f64,
        e_tot_pyscf: f64,
        exc_sl_pyscf: f64,
        exx_energy: f64,
        exc_sl_xcx_integrated: f64,
        exc_sl_libxc610_integrated: f64,
        e_tot_reconstructed_xcx: f64,
        delta_e_tot_xcx: f64,
        version_delta_610_vs_pyscf: f64,
        xcx_vs_610_integrated_abs: f64,
        fullgrid_max_rel_dev_xcx_vs_610: f64,
        fullgrid_points_over_rtol: usize,
        e_tol: f64,
        rtol: f64,
    }

    pub fn run() {
        let xc = Libxc::load();
        let (vmaj, vmin, vmic) = xc.version();
        let xcx_libxc = format!("{vmaj}.{vmin}.{vmic}");
        eprintln!("xcx-side libxc (truth) {xcx_libxc}");
        assert_eq!(
            xcx_libxc, "6.1.0",
            "expected pinned libxc 6.1.0 for the truth"
        );

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let testdata = manifest.join("testdata");
        let fullgrid_dir = testdata.join("scf").join("_fullgrid");
        let scalars_dir = testdata.join("scf");
        std::fs::create_dir_all(&scalars_dir).unwrap();

        println!("\n=== SCF end-to-end gate (libxc truth {xcx_libxc}) ===");
        let mut all_ok = true;
        for case in ["h2o_b3lyp", "oh_b3lyp"] {
            let path = fullgrid_dir.join(format!("{case}.fullgrid.json"));
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "missing {} ({e}); run python/gen_scf_ref.py in WSL first",
                    path.display()
                )
            });
            let fg: FullGrid = serde_json::from_str(&text).unwrap();
            assert_eq!(fg.libxc_id, B3LYP_ID as u32, "{case}: not libxc 402");
            all_ok &= process(&xc, &xcx_libxc, &fg, &testdata, &scalars_dir);
        }
        println!("=== SCF gate complete ===\n");
        assert!(all_ok, "SCF gate FAILED — see per-case report above");
    }

    fn process(
        xc: &Libxc,
        xcx_libxc: &str,
        fg: &FullGrid,
        testdata: &Path,
        scalars_dir: &Path,
    ) -> bool {
        let spin = if fg.spin == "polarized" {
            Spin::Polarized
        } else {
            Spin::Unpolarized
        };
        let nspin = spin.channels();
        let np = fg.n_grid;
        assert_eq!(fg.rho.len(), np * nspin, "{}: rho len", fg.case);
        assert_eq!(
            fg.sigma.len(),
            np * (2 * nspin - 1),
            "{}: sigma len",
            fg.case
        );
        assert_eq!(fg.weights.len(), np, "{}: weights len", fg.case);

        // Total density per point (for integration weighting + subset selection).
        let ntot: Vec<f64> = (0..np)
            .map(|i| match spin {
                Spin::Unpolarized => fg.rho[i],
                _ => fg.rho[2 * i] + fg.rho[2 * i + 1],
            })
            .collect();

        // --- xcx on the full grid ---
        let f = Functional::by_name("hyb_gga_xc_b3lyp", spin).unwrap();
        let out = f
            .eval(np, &XcInput::gga(&fg.rho, &fg.sigma))
            .expect("xcx eval");

        // --- libxc 6.1.0 on the full grid ---
        let (zk610, vrho610, vsig610) =
            xc.gga_exc_vxc(B3LYP_ID, nspin as i32, np, &fg.rho, &fg.sigma);

        // Layer (a) full-grid: max relative deviation across exc/vrho/vsigma.
        let mut max_dev = 0.0_f64;
        let mut over = 0usize;
        // `over` is the meaningful pass/fail count (rel_close, with the atol floor
        // the golden suite uses). `max_dev` is reported for headroom but gated to
        // magnitudes above the atol floor — a pure relative ratio on near-zero
        // tail/minority values is meaningless noise (|Δ| there is ~atol anyway).
        let bump = |a: f64, b: f64, max_dev: &mut f64, over: &mut usize| {
            if !rel_close(a, b, RTOL, ATOL) {
                *over += 1;
            }
            let scale = a.abs().max(b.abs());
            if scale > 1e-12 {
                let d = (a - b).abs() / scale;
                if d > *max_dev {
                    *max_dev = d;
                }
            }
        };
        for (a, b) in out.exc.iter().zip(&zk610) {
            bump(*a, *b, &mut max_dev, &mut over);
        }
        for (a, b) in out.vrho.iter().zip(&vrho610) {
            bump(*a, *b, &mut max_dev, &mut over);
        }
        for (a, b) in out.vsigma.iter().zip(&vsig610) {
            bump(*a, *b, &mut max_dev, &mut over);
        }

        // Integrated semilocal E_xc on the full grid.
        let integrate =
            |eps: &[f64]| -> f64 { (0..np).map(|i| fg.weights[i] * ntot[i] * eps[i]).sum() };
        let exc_xcx = integrate(&out.exc);
        let exc_610 = integrate(&zk610);
        let exc_pyscf_quad = integrate(&fg.exc_pyscf); // == fg.exc_sl_semilocal (PySCF 7.0.0)

        // Layer (b): swap only the semilocal XC source on the converged density.
        let e_tot_xcx = fg.e_tot - fg.exc_sl_semilocal + exc_xcx;
        let delta_e = (e_tot_xcx - fg.e_tot).abs();
        let version_delta = (exc_610 - fg.exc_sl_semilocal).abs();
        let xcx_vs_610 = (exc_xcx - exc_610).abs();

        // --- report ---
        println!(
            "\n[{}]  {} / {}  ({} spin, {} grid pts)",
            fg.case,
            fg.molecule_short(),
            fg.basis,
            fg.spin,
            np
        );
        println!(
            "  libxc: truth {xcx_libxc} (FFI)  |  pyscf {} / libxc {}  |  id {}  hyb {}",
            fg.pyscf_version, fg.libxc_version, fg.libxc_id, fg.hybrid_coeff
        );
        println!(
            "  layer (a) full grid: max rel dev xcx-vs-6.1.0 = {max_dev:.2e}  ({over} pts > rtol {RTOL:.0e})"
        );
        println!(
            "  E_xc^sl   xcx = {exc_xcx:.10}   libxc6.1.0 = {exc_610:.10}   pyscf7.0.0 = {:.10}",
            fg.exc_sl_semilocal
        );
        println!("            pyscf-eps quadrature = {exc_pyscf_quad:.10} (integration-convention check)");
        println!("  version delta |6.1.0 − 7.0.0| (integrated E_xc) = {version_delta:.3e}");
        println!("  xcx vs 6.1.0 (integrated)                       = {xcx_vs_610:.3e}");
        println!(
            "  layer (b) E_tot: pyscf = {:.10}   reconstructed(xcx) = {e_tot_xcx:.10}   ΔE = {delta_e:.3e}  (tol {E_TOL:.0e})",
            fg.e_tot
        );
        let verdict = if delta_e < E_TOL {
            "PASS"
        } else {
            "FAIL — investigate (version delta? seam?)"
        };
        println!("  => {verdict}");

        // --- write the biased real-grid golden subset (libxc-6.1.0 truth) ---
        let idx = select_subset(&ntot, &fg.sigma, spin, SUBSET);
        write_subset(xc, fg, spin, &idx, testdata);

        // --- write the scalar record ---
        let scal = ScfScalars {
            case: fg.case.clone(),
            molecule: fg.molecule.clone(),
            basis: fg.basis.clone(),
            spin: fg.spin.clone(),
            n_grid: np,
            pyscf_version: fg.pyscf_version.clone(),
            pyscf_libxc_version: fg.libxc_version.clone(),
            xcx_libxc_version: xcx_libxc.to_string(),
            libxc_id: fg.libxc_id,
            hybrid_coeff: fg.hybrid_coeff,
            e_tot_pyscf: fg.e_tot,
            exc_sl_pyscf: fg.exc_sl_semilocal,
            exx_energy: fg.exx_energy,
            exc_sl_xcx_integrated: exc_xcx,
            exc_sl_libxc610_integrated: exc_610,
            e_tot_reconstructed_xcx: e_tot_xcx,
            delta_e_tot_xcx: delta_e,
            version_delta_610_vs_pyscf: version_delta,
            xcx_vs_610_integrated_abs: xcx_vs_610,
            fullgrid_max_rel_dev_xcx_vs_610: max_dev,
            fullgrid_points_over_rtol: over,
            e_tol: E_TOL,
            rtol: RTOL,
        };
        let sp = scalars_dir.join(format!("{}.scalars.json", fg.case));
        std::fs::write(&sp, serde_json::to_string_pretty(&scal).unwrap()).unwrap();
        eprintln!("wrote {}", sp.display());

        delta_e < E_TOL && over == 0
    }

    /// Pick a subset biased toward density tails, near-nucleus cusp, and high
    /// reduced gradient (not a uniform sample), so the committed regression guards
    /// the edges — mirroring the fuzz densification discipline. Deterministic.
    fn select_subset(ntot: &[f64], sigma: &[f64], spin: Spin, target: usize) -> Vec<usize> {
        let n = ntot.len();
        if n <= target {
            return (0..n).collect();
        }
        let sig_tot = |i: usize| match spin {
            Spin::Unpolarized => sigma[i],
            _ => sigma[3 * i] + 2.0 * sigma[3 * i + 1] + sigma[3 * i + 2],
        };
        let red_grad = |i: usize| sig_tot(i).max(0.0).sqrt() / ntot[i].max(1e-300).powf(4.0 / 3.0);

        let mut by_rho: Vec<usize> = (0..n).collect();
        by_rho.sort_by(|&a, &b| ntot[a].partial_cmp(&ntot[b]).unwrap());
        let mut by_s: Vec<usize> = (0..n).collect();
        by_s.sort_by(|&a, &b| red_grad(a).partial_cmp(&red_grad(b)).unwrap());

        let mut sel = std::collections::BTreeSet::new();
        let q = target / 4;
        for i in 0..q {
            sel.insert(by_rho[i]); // density tail (lowest rho)
            sel.insert(by_rho[n - 1 - i]); // near-nucleus cusp (highest rho)
            sel.insert(by_s[n - 1 - i]); // highest reduced gradient
        }
        // fill the remainder with a stratified sweep across the rho-sorted order
        if sel.len() < target {
            let step = (n / (target - sel.len())).max(1);
            let mut i = 0;
            while i < n && sel.len() < target {
                sel.insert(by_rho[i]);
                i += step;
            }
        }
        sel.into_iter().collect()
    }

    /// Extract the subset in xcx packing, evaluate libxc-6.1.0 on it, and write a
    /// `GoldenCase` array to `testdata/scf_grid_{case}.json` (picked up by the
    /// libxc-free golden test).
    fn write_subset(xc: &Libxc, fg: &FullGrid, spin: Spin, idx: &[usize], testdata: &Path) {
        let nspin = spin.channels();
        let nps = idx.len();
        let mut rho = Vec::with_capacity(nps * nspin);
        let mut sigma = Vec::with_capacity(nps * (2 * nspin - 1));
        for &i in idx {
            match spin {
                Spin::Unpolarized => {
                    rho.push(fg.rho[i]);
                    sigma.push(fg.sigma[i]);
                }
                _ => {
                    rho.push(fg.rho[2 * i]);
                    rho.push(fg.rho[2 * i + 1]);
                    sigma.push(fg.sigma[3 * i]);
                    sigma.push(fg.sigma[3 * i + 1]);
                    sigma.push(fg.sigma[3 * i + 2]);
                }
            }
        }
        let (exc, vrho, vsigma) = xc.gga_exc_vxc(B3LYP_ID, nspin as i32, nps, &rho, &sigma);
        let case = GoldenCase {
            functional: "hyb_gga_xc_b3lyp".into(),
            libxc_id: B3LYP_ID as u32,
            libxc_version: "6.1.0".to_string(),
            spin: fg.spin.clone(),
            np: nps,
            rho,
            sigma,
            tau: vec![], // hyb-GGA: no τ
            exc,
            vrho,
            vsigma,
            vtau: vec![],
            // SCF-grid cases are vxc-only.
            v2rho2: vec![],
            v2rhosigma: vec![],
            v2sigma2: vec![],
            v2rhotau: vec![],
            v2sigmatau: vec![],
            v2tau2: vec![],
        };
        let path: PathBuf = testdata.join(format!("scf_grid_{}.json", fg.case));
        std::fs::write(&path, serde_json::to_string_pretty(&vec![case]).unwrap()).unwrap();
        eprintln!("wrote {} ({nps} real-grid points)", path.display());
    }

    impl FullGrid {
        fn molecule_short(&self) -> String {
            self.molecule.split(';').next().unwrap_or("").trim().into()
        }
    }
}
