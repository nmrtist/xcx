// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Generate **xcx-self regression snapshots** for clean-room functionals that
//! exist in **no libxc release** and cannot be assembled from libxc components
//! (currently ωB97M(2), whose refit {w, u}-series coefficient tables are not
//! in libxc). These snapshots are *not* libxc-verified: they lock xcx against
//! its own FD-validated implementation (the public derivative tests in
//! `crates/xcx/tests` establish correctness; this locks the values against
//! regressions). The convention: `libxc_version` is `"xcx-self <crate
//! version> (FD-validated; not libxc-verified)"` and `libxc_id` is the
//! xcx-private id (≥ 100000).
//!
//! ```text
//! cargo run -p xcx-validation --bin gen_self
//! ```

use std::path::Path;

use xcx::{Functional, FunctionalId, Spin, XcInput};
use xcx_validation::GoldenCase;

fn main() {
    let outdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    std::fs::create_dir_all(&outdir).unwrap();
    gen_mgga_self(
        &outdir,
        "hyb_mgga_xc_wb97m_2",
        FunctionalId::HybMggaXcWb97m2,
    );
}

/// Snapshot one meta-GGA functional from xcx itself, on the same vxc/fxc point
/// sets `gen_golden` uses for meta-GGAs (full sets — no libxc-divergence
/// exclusions apply, since both sides of the comparison are xcx).
fn gen_mgga_self(outdir: &Path, name: &str, id: FunctionalId) {
    let version = format!(
        "xcx-self {} (FD-validated; not libxc-verified)",
        env!("CARGO_PKG_VERSION")
    );

    let unpol: Vec<(f64, f64, f64)> = vec![
        (1e-14, 0.0, 1e-15),
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
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        [1.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0],
        [1.0, 1e-4, 0.2, 0.0, 1e-6, 0.6, 1e-5],
        [0.6, 0.3, 0.1, 0.05, 0.08, 0.4, 0.25],
        [3.0, 2.0, 1.0, 0.5, 0.8, 4.0, 3.0],
        [0.5, 0.5, 0.1, 10.0, 0.1, 0.5, 0.5],
        [0.5, 0.5, 0.1, -10.0, 0.1, 0.5, 0.5],
        [1e-8, 1e-9, 2e-20, 0.0, 4e-23, 7e-13, 1e-14],
        [100.0, 50.0, 1e3, 500.0, 800.0, 2e3, 1e3],
    ];
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

    let case =
        |spin: Spin, rho: Vec<f64>, sigma: Vec<f64>, tau: Vec<f64>, want_fxc: bool| -> GoldenCase {
            let ns = spin.channels();
            let np = rho.len() / ns;
            let f = Functional::new(id, spin).unwrap();
            let input = XcInput::gga(&rho, &sigma).with_tau(&tau);
            let out = if want_fxc {
                f.eval_fxc(np, &input).unwrap()
            } else {
                f.eval(np, &input).unwrap()
            };
            GoldenCase {
                functional: name.into(),
                libxc_id: id.as_u32(),
                libxc_version: version.clone(),
                spin: match spin {
                    Spin::Polarized => "polarized".into(),
                    _ => "unpolarized".into(),
                },
                np,
                rho,
                sigma,
                tau,
                exc: out.exc,
                vrho: out.vrho,
                vsigma: out.vsigma,
                vtau: out.vtau,
                v2rho2: out.v2rho2,
                v2rhosigma: out.v2rhosigma,
                v2sigma2: out.v2sigma2,
                v2rhotau: out.v2rhotau,
                v2sigmatau: out.v2sigmatau,
                v2tau2: out.v2tau2,
            }
        };

    let flat = |p: &[[f64; 7]]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            p.iter().flat_map(|x| [x[0], x[1]]).collect(),
            p.iter().flat_map(|x| [x[2], x[3], x[4]]).collect(),
            p.iter().flat_map(|x| [x[5], x[6]]).collect(),
        )
    };
    let (pol_rho, pol_sigma, pol_tau) = flat(&pol);
    let (fp_rho, fp_sigma, fp_tau) = flat(&fxc_pol);

    let cases = vec![
        case(
            Spin::Unpolarized,
            unpol.iter().map(|&(r, _, _)| r).collect(),
            unpol.iter().map(|&(_, s, _)| s).collect(),
            unpol.iter().map(|&(_, _, t)| t).collect(),
            false,
        ),
        case(Spin::Polarized, pol_rho, pol_sigma, pol_tau, false),
        case(
            Spin::Unpolarized,
            fxc_unpol.iter().map(|&(r, _, _)| r).collect(),
            fxc_unpol.iter().map(|&(_, s, _)| s).collect(),
            fxc_unpol.iter().map(|&(_, _, t)| t).collect(),
            true,
        ),
        case(Spin::Polarized, fp_rho, fp_sigma, fp_tau, true),
    ];
    let json = serde_json::to_string_pretty(&cases).unwrap();
    let path = outdir.join(format!("{name}.json"));
    std::fs::write(&path, json).unwrap();
    eprintln!("wrote {}", path.display());
}
