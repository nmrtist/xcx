// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Golden cross-check: compare `xcx` outputs against committed libxc snapshots.
//! Runs in CI without libxc present — snapshots live under `testdata/`.

use std::path::{Path, PathBuf};

use xcx::{Functional, Spin, XcInput};
use xcx_validation::{rel_close, GoldenCase, ATOL, RTOL};

fn snapshot_files() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn spin_of(s: &str) -> Spin {
    match s {
        "polarized" => Spin::Polarized,
        _ => Spin::Unpolarized,
    }
}

fn cmp(c: &GoldenCase, field: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{}/{}/{field} length: {} vs {}",
        c.functional,
        c.spin,
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            rel_close(*g, *w, RTOL, ATOL),
            "{}/{}/{field}[{i}]: xcx {g:.16e} vs libxc {w:.16e} (|Δ|={:.3e}, rtol={RTOL:e})",
            c.functional,
            c.spin,
            (g - w).abs()
        );
    }
}

#[test]
fn golden_snapshots_match() {
    let files = snapshot_files();
    if files.is_empty() {
        eprintln!(
            "no golden snapshots (testdata/ empty); regenerate with: \
             cargo run -p xcx-validation --features libxc-ffi --bin gen_golden"
        );
        return;
    }

    let mut checked = 0usize;
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let cases: Vec<GoldenCase> = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        for c in &cases {
            let f = Functional::by_name(&c.functional, spin_of(&c.spin))
                .unwrap_or_else(|e| panic!("building {} ({}): {e:?}", c.functional, c.spin));
            let input = if c.sigma.is_empty() {
                XcInput::lda(&c.rho)
            } else {
                XcInput::gga(&c.rho, &c.sigma)
            };
            let out = f.eval(c.np, &input).unwrap();
            cmp(c, "exc", &out.exc, &c.exc);
            cmp(c, "vrho", &out.vrho, &c.vrho);
            if !c.vsigma.is_empty() {
                cmp(c, "vsigma", &out.vsigma, &c.vsigma);
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "snapshots present but nothing compared");
    eprintln!("golden: {checked} case(s) matched within rtol={RTOL:e}");
}
