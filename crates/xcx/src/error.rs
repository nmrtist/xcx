// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::func::FunctionalId;

/// Errors returned by the public API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum XcError {
    /// The functional exists in the registry but is not yet implemented.
    NotImplemented(FunctionalId),
    /// No functional matches the requested name or numeric id.
    UnknownFunctional,
    /// A required input array was not supplied (e.g. a GGA called without `sigma`).
    MissingInput(&'static str),
    /// An input or output array had an unexpected length.
    LengthMismatch {
        /// Number of elements the packing convention requires.
        expected: usize,
        /// Number of elements actually supplied.
        found: usize,
    },
    /// Attempted to mix functionals of differing spin, or with an empty part list.
    SpinMismatch,
}

impl std::fmt::Display for XcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XcError::NotImplemented(id) => {
                write!(f, "functional `{}` is not yet implemented", id.name())
            }
            XcError::UnknownFunctional => write!(f, "unknown functional"),
            XcError::MissingInput(name) => write!(f, "missing required input `{name}`"),
            XcError::LengthMismatch { expected, found } => {
                write!(
                    f,
                    "array length mismatch: expected {expected}, found {found}"
                )
            }
            XcError::SpinMismatch => write!(f, "spin mismatch (or empty mix)"),
        }
    }
}

impl std::error::Error for XcError {}
