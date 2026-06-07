// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Concrete functional registry.
//!
//! The public contract (types + metadata, see `docs/api-convention.md`) is
//! frozen before any functional math is written. Functionals land in subsequent
//! steps; each file will be tagged with its provenance
//! (`Provenance: ported-from-libxc` / `Provenance: clean-room`).

use crate::error::XcError;
use crate::families::XcEval;
use crate::func::FunctionalId;

mod gga_c_lyp;
mod gga_c_pbe;
mod gga_x_b88;
mod gga_x_pbe;
mod hybrids;
mod lda_c_pw;
mod lda_c_vwn;
mod lda_c_vwn_3;
mod lda_c_vwn_rpa;
mod lda_x;

/// Build the boxed evaluator for a functional id. The full v0.1 set is
/// implemented, so this matches every [`FunctionalId`] variant explicitly (no
/// catch-all): adding a future variant deliberately fails to compile here until
/// it is wired up or routed to [`XcError::NotImplemented`].
pub(crate) fn build(id: FunctionalId) -> Result<Box<dyn XcEval>, XcError> {
    use FunctionalId::*;
    match id {
        LdaX => Ok(lda_x::LdaX::boxed()),
        LdaCPw => Ok(lda_c_pw::LdaCPw::boxed()),
        LdaCVwn => Ok(lda_c_vwn::LdaCVwn::boxed()),
        LdaCVwn3 => Ok(lda_c_vwn_3::LdaCVwn3::boxed()),
        LdaCVwnRpa => Ok(lda_c_vwn_rpa::LdaCVwnRpa::boxed()),
        GgaXPbe => Ok(gga_x_pbe::GgaXPbe::boxed()),
        GgaXB88 => Ok(gga_x_b88::GgaXB88::boxed()),
        GgaCPbe => Ok(gga_c_pbe::GgaCPbe::boxed()),
        GgaCLyp => Ok(gga_c_lyp::GgaCLyp::boxed()),
        HybGgaXcPbeh => hybrids::pbeh(),
        HybGgaXcB3lyp5 => hybrids::b3lyp5(),
        HybGgaXcB3lyp => hybrids::b3lyp(),
    }
}
