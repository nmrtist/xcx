// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Dev-only verification harness for `xcx`. **Not published.**
//!
//! - Default build: compares `xcx` against committed golden snapshots under
//!   `testdata/`, requiring no libxc — this is what CI runs.
//! - With `--features libxc-ffi`: regenerates those snapshots from a
//!   conda-forge libxc via FFI (`bin/gen_golden`).
//!
//! The external libxc is used for verification only; no libxc source is vendored
//! or published, preserving the MPL provenance boundary.

use serde::{Deserialize, Serialize};

/// A single reference sample: inputs and libxc outputs for one functional,
/// one spin mode, at `np` points. Snapshots are arrays of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenCase {
    /// libxc functional name, e.g. `"gga_x_pbe"`.
    pub functional: String,
    /// libxc numeric id.
    pub libxc_id: u32,
    /// libxc version the snapshot was generated against (pinned).
    pub libxc_version: String,
    /// `"unpolarized"` or `"polarized"`.
    pub spin: String,
    /// Number of grid points.
    pub np: usize,
    /// Packed `rho` input.
    pub rho: Vec<f64>,
    /// Packed `sigma` input (empty for LDA).
    #[serde(default)]
    pub sigma: Vec<f64>,
    /// libxc `zk` (energy per particle).
    pub exc: Vec<f64>,
    /// libxc `vrho`.
    pub vrho: Vec<f64>,
    /// libxc `vsigma` (empty for LDA).
    #[serde(default)]
    pub vsigma: Vec<f64>,
}

/// Relative-with-absolute-floor closeness test used across the golden suite:
/// `|a − b| ≤ rtol·max(|a|,|b|) + atol`.
pub fn rel_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= rtol * a.abs().max(b.abs()) + atol
}

/// Tolerances for the golden comparison (v0.1 definition of done).
pub const RTOL: f64 = 1e-10;
/// Absolute floor so values screened to exactly zero compare cleanly.
pub const ATOL: f64 = 1e-12;

#[cfg(feature = "libxc-ffi")]
pub mod ffi;
