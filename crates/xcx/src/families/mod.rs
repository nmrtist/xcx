// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Family evaluation harnesses and the object-safe evaluator trait.
//!
//! Concrete functionals implement a per-family *energy* trait ([`lda::LdaEnergy`],
//! [`gga::GgaEnergy`]) carrying a generic scalar energy expression; the family
//! wrapper ([`lda::Lda`], [`gga::Gga`]) implements [`XcEval`] by running the
//! autodiff grid harness. This sidesteps the object-safety conflict of a generic
//! trait method while still permitting `Box<dyn XcEval>` runtime dispatch.

use crate::error::XcError;
use crate::func::{FunctionalInfo, Spin};
use crate::io::{XcInput, XcResult};

pub(crate) mod gga;
pub(crate) mod lda;
pub(crate) mod mgga;

/// Object-safe evaluator: the runtime-dispatched form of a functional.
pub(crate) trait XcEval: Send + Sync {
    fn info(&self) -> &FunctionalInfo;
    /// Energy + all available first derivatives.
    fn eval(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError>;
    /// Energy + first derivatives + second derivatives (`fxc`). Fills the same
    /// fields as [`eval`](XcEval::eval) plus `v2rho2`/`v2rhosigma`/`v2sigma2`.
    fn eval_fxc(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError>;
}

/// Validate that `slice.len() == expected`.
pub(crate) fn check_len(slice: &[f64], expected: usize) -> Result<(), XcError> {
    if slice.len() == expected {
        Ok(())
    } else {
        Err(XcError::LengthMismatch {
            expected,
            found: slice.len(),
        })
    }
}
