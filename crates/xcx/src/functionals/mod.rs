// Copyright (c) 2026 Jiekang Tian and the xcx authors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete functional registry.
//!
//! The public contract (types + metadata, see `docs/api-convention.md`) is
//! frozen before any functional math is written. Each functional file is
//! tagged with its provenance, which also determines its license (see
//! `NOTICE`): `Provenance: ported-from-libxc` (MPL-2.0) or
//! `Provenance: clean-room` (implemented from the published literature;
//! MIT OR Apache-2.0).

use crate::error::XcError;
use crate::families::XcEval;
use crate::func::FunctionalId;

mod attenuation;
mod gga_c_lyp;
mod gga_c_p86;
mod gga_c_pbe;
mod gga_c_pbe_sol;
mod gga_x_b88;
mod gga_x_pbe;
mod gga_x_pbe_r;
mod gga_x_pbe_sol;
mod gga_x_rpbe;
mod gga_xc_b97;
mod hyb_gga_xc_wb97x_v;
mod hyb_mgga_x_m06_2x;
mod hyb_mgga_xc_pw6b95;
mod hyb_mgga_xc_pwpb95;
mod hyb_mgga_xc_wb97m_2;
mod hyb_mgga_xc_wb97m_v;
mod hybrids;
mod lda_c_pw;
mod lda_c_vwn;
mod lda_c_vwn_3;
mod lda_c_vwn_rpa;
mod lda_x;
mod mgga_c_m06_2x;
mod mgga_c_m06_l;
mod mgga_c_r2scan;
mod mgga_c_tpss;
mod mgga_x_m06_l;
mod mgga_x_r2scan;
mod mgga_x_tpss;
mod mgga_xc_b97mv;

/// Build the boxed evaluator for a functional id. Every [`FunctionalId`]
/// variant is matched explicitly (no catch-all): adding a future variant
/// deliberately fails to compile here until
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
        GgaXPbeR => Ok(gga_x_pbe_r::GgaXPbeR::boxed()),
        GgaXPbeSol => Ok(gga_x_pbe_sol::GgaXPbeSol::boxed()),
        GgaXRpbe => Ok(gga_x_rpbe::GgaXRpbe::boxed()),
        GgaCPbeSol => Ok(gga_c_pbe_sol::GgaCPbeSol::boxed()),
        MggaXTpss => Ok(mgga_x_tpss::MggaXTpss::boxed()),
        MggaCTpss => Ok(mgga_c_tpss::MggaCTpss::boxed()),
        MggaXR2scan => Ok(mgga_x_r2scan::MggaXR2scan::boxed()),
        MggaCR2scan => Ok(mgga_c_r2scan::MggaCR2scan::boxed()),
        MggaXM06L => Ok(mgga_x_m06_l::MggaXM06L::boxed()),
        MggaCM06L => Ok(mgga_c_m06_l::MggaCM06L::boxed()),
        HybGgaXcPbeh => hybrids::pbeh(),
        HybGgaXcB3lyp5 => hybrids::b3lyp5(),
        HybGgaXcB3lyp => hybrids::b3lyp(),
        HybMggaXM062x => Ok(hyb_mgga_x_m06_2x::HybMggaXM062x::boxed()),
        MggaCM062x => Ok(mgga_c_m06_2x::MggaCM062x::boxed()),
        HybMggaXcPw6b95 => hyb_mgga_xc_pw6b95::pw6b95(),
        MggaXcB97mV => Ok(mgga_xc_b97mv::MggaXcB97mV::boxed()),
        HybGgaXcWb97xV => Ok(hyb_gga_xc_wb97x_v::HybGgaXcWb97xV::boxed()),
        HybMggaXcWb97mV => Ok(hyb_mgga_xc_wb97m_v::HybMggaXcWb97mV::boxed()),
        GgaCP86 => Ok(gga_c_p86::GgaCP86::boxed()),
        HybGgaXcB2plyp => hybrids::b2plyp(),
        HybGgaXcRevdsdPbep86D4 => hybrids::revdsd_pbep86_d4(),
        HybMggaXcPwpb95 => hyb_mgga_xc_pwpb95::pwpb95(),
        HybMggaXcWb97m2 => Ok(hyb_mgga_xc_wb97m_2::HybMggaXcWb97m2::boxed()),
        GgaXcB973c => Ok(gga_xc_b97::b97_3c()),
        HybGgaXcPbeh3c => hybrids::pbeh_3c(),
    }
}

/// Parameterized PBE exchange (κ, μ) — the evaluator behind the public
/// [`crate::Functional::pbe_x`] constructor (shared PBE-x code path).
pub(crate) fn pbe_x_param(kappa: f64, mu: f64) -> Box<dyn XcEval> {
    gga_x_pbe::GgaXPbeParam::boxed(kappa, mu)
}

/// Parameterized PBE correlation (β) — the evaluator behind the public
/// [`crate::Functional::pbe_c`] constructor (shared PBE-c code path).
pub(crate) fn pbe_c_param(beta: f64) -> Box<dyn XcEval> {
    gga_c_pbe::GgaCPbeParam::boxed(beta)
}

/// B97 power series with caller-supplied coefficients — the evaluator behind
/// the public [`crate::Functional::b97_xc`] constructor.
pub(crate) fn b97_series(c_x: &[f64], c_ss: &[f64], c_os: &[f64]) -> Box<dyn XcEval> {
    gga_xc_b97::b97_series(c_x, c_ss, c_os)
}
