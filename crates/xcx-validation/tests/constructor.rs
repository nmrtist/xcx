// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Golden cross-checks for the **public parameterized constructors** against
//! libxc functionals that are not registered xcx ids — snapshots live under
//! `testdata/constructor/` (kept out of `testdata/*.json` so the by-name
//! golden test does not try to resolve them).
//!
//! Currently: the semilocal part of libxc's original **B97**
//! (`hyb_gga_xc_b97`, id 407; Becke, J. Chem. Phys. 107, 8554 (1997)) vs
//! `Functional::b97_xc` fed Becke's original coefficients — proving the
//! series constructor reproduces the canonical B97 family member, not just
//! the B97-3c refit.

use std::path::Path;

use xcx::{Functional, Spin, XcInput};
use xcx_validation::{rel_close, GoldenCase, ATOL, RTOL};

/// Becke's original B97 series coefficients (libxc `b97_values`; exact
/// exchange 0.1943 is *not* part of the semilocal series).
const B97_C_X: [f64; 3] = [0.8094, 0.5073, 0.7481];
const B97_C_SS: [f64; 3] = [0.1737, 2.3487, -2.4868];
const B97_C_OS: [f64; 3] = [0.9454, 0.7471, -4.5961];

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
fn b97_series_constructor_matches_libxc_b97() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("constructor")
        .join("hyb_gga_xc_b97.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let cases: Vec<GoldenCase> = serde_json::from_str(&text).unwrap();
    assert!(!cases.is_empty());

    let mut checked = 0usize;
    for c in &cases {
        let spin = match c.spin.as_str() {
            "polarized" => Spin::Polarized,
            _ => Spin::Unpolarized,
        };
        let f = Functional::b97_xc(&B97_C_X, &B97_C_SS, &B97_C_OS, spin);
        let input = XcInput::gga(&c.rho, &c.sigma);
        let out = f.eval(c.np, &input).unwrap();
        cmp(c, "exc", &out.exc, &c.exc);
        cmp(c, "vrho", &out.vrho, &c.vrho);
        cmp(c, "vsigma", &out.vsigma, &c.vsigma);
        if !c.v2rho2.is_empty() {
            let f2 = f.eval_fxc(c.np, &input).unwrap();
            cmp(c, "v2rho2", &f2.v2rho2, &c.v2rho2);
            cmp(c, "v2rhosigma", &f2.v2rhosigma, &c.v2rhosigma);
            cmp(c, "v2sigma2", &f2.v2sigma2, &c.v2sigma2);
        }
        checked += 1;
    }
    eprintln!("constructor golden: {checked} case(s) matched within rtol={RTOL:e}");
}
