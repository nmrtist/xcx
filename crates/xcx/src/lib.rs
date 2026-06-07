// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! # xcx — exchange–correlation functionals for DFT (pure-Rust, libxc-compatible)
//!
//! `xcx` evaluates exchange–correlation (XC) functionals: given a density (and,
//! depending on the functional, its gradient / kinetic energy density), it
//! returns the XC energy per particle and its first derivatives, together with
//! metadata (family, requirements, exact-exchange fraction, range-separation and
//! VV10 parameters).
//!
//! Each functional is written once as a scalar energy expression; derivatives are
//! obtained by forward-mode automatic differentiation, so they are correct by
//! construction. Functional IDs match [libxc](https://libxc.gitlab.io/).
//!
//! ## Scope fence
//!
//! `xcx` maps `(rho, sigma, tau[, lapl]) → energy density + derivatives +
//! metadata + linear mixing` and nothing else — no grids, atomic-orbital
//! evaluation, SCF driver, or dispersion. For hybrids and VV10 it exposes the
//! parameters; it does not compute the exact-exchange or nonlocal integrals.
//!
//! The full, semver-stable contract lives in [`docs/api-convention.md`](https://github.com/nmrtist/xcx/blob/main/docs/api-convention.md).
//!
//! ## Example
//!
//! ```
//! use xcx::{Functional, FunctionalId, Spin, XcInput};
//!
//! // Spin-unpolarized LDA exchange over three grid points.
//! let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized)?;
//! let rho = [0.1_f64, 0.2, 0.3];
//! let out = f.eval(rho.len(), &XcInput::lda(&rho))?;
//!
//! assert_eq!(out.exc.len(), 3); // energy per particle at each point
//! assert_eq!(out.vrho.len(), 3); // ∂(n·ε_xc)/∂n at each point
//! # Ok::<(), xcx::XcError>(())
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod families;
mod func;
mod functionals;
mod io;
mod reduced;

pub use error::XcError;
pub use func::{
    CamParams, Family, Functional, FunctionalId, FunctionalInfo, HybridInfo, Kind, Spin, Vv10Params,
};
pub use io::{XcInput, XcResult};
