// Copyright (c) 2026 Jiekang Tian and the xcx authors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Functional identity, metadata, and the public `Functional` handle.

use crate::error::XcError;
use crate::families::XcEval;
use crate::io::{XcInput, XcResult};

/// Spin treatment of the density.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Spin {
    /// Spin-unpolarized (closed shell): a single density channel.
    Unpolarized,
    /// Spin-polarized (open shell): separate α/β channels.
    Polarized,
}

impl Spin {
    /// Number of spin channels (1 or 2).
    pub fn channels(self) -> usize {
        match self {
            Spin::Unpolarized => 1,
            Spin::Polarized => 2,
        }
    }
}

/// Functional family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Family {
    /// Local density approximation.
    Lda,
    /// Generalized gradient approximation.
    Gga,
    /// Meta-GGA.
    Mgga,
    /// Hybrid GGA (includes exact exchange).
    HybGga,
    /// Hybrid meta-GGA.
    HybMgga,
}

/// Whether a functional models exchange, correlation, both, or kinetic energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    /// Exchange only.
    Exchange,
    /// Correlation only.
    Correlation,
    /// Combined exchange–correlation.
    ExchangeCorrelation,
    /// Kinetic energy functional.
    Kinetic,
}

/// A functional identifier. Numeric values equal libxc's for interoperability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FunctionalId {
    /// Slater exchange (libxc 1).
    LdaX,
    /// Perdew–Wang 1992 (PW92) correlation (libxc 12). Uniform-gas limit of PBE-C.
    LdaCPw,
    /// Vosko–Wilk–Nusair correlation, parametrization V / "VWN5" (libxc 7).
    LdaCVwn,
    /// Vosko–Wilk–Nusair correlation, parametrization III / "VWN3" (libxc 30).
    LdaCVwn3,
    /// Vosko–Wilk–Nusair correlation, RPA parametrization "VWN5_RPA" (libxc 8).
    /// libxc's B3LYP (402) mixes this in — distinct from VWN3 (30) and VWN5 (7).
    LdaCVwnRpa,
    /// Perdew–Burke–Ernzerhof exchange (libxc 101).
    GgaXPbe,
    /// Becke 88 exchange (libxc 106).
    GgaXB88,
    /// Perdew–Burke–Ernzerhof correlation (libxc 130).
    GgaCPbe,
    /// Lee–Yang–Parr correlation (libxc 131).
    GgaCLyp,
    /// Revised PBE exchange (revPBE), Zhang & Yang (libxc 102). PBE-x with κ = 1.245.
    GgaXPbeR,
    /// PBE-for-solids exchange (PBEsol), Perdew et al. 2008 (libxc 116). PBE-x with
    /// μ = 10/81.
    GgaXPbeSol,
    /// RPBE exchange, Hammer–Hansen–Nørskov (libxc 117). PBE constants, *exponential*
    /// enhancement.
    GgaXRpbe,
    /// PBE-for-solids correlation (PBEsol), Perdew et al. 2008 (libxc 133). PBE-c
    /// with β = 0.046.
    GgaCPbeSol,
    /// Tao–Perdew–Staroverov–Scuseria exchange — meta-GGA (libxc 202).
    MggaXTpss,
    /// Tao–Perdew–Staroverov–Scuseria correlation — meta-GGA (libxc 231).
    MggaCTpss,
    /// Re-regularized SCAN exchange (r2SCAN) — meta-GGA (libxc 497).
    MggaXR2scan,
    /// Re-regularized SCAN correlation (r2SCAN) — meta-GGA (libxc 498).
    MggaCR2scan,
    /// Minnesota M06-L exchange — meta-GGA (libxc 203).
    MggaXM06L,
    /// Minnesota M06-L correlation — meta-GGA (libxc 233).
    MggaCM06L,
    /// B3LYP, VWN_RPA convention (libxc 402). Mixes `lda_c_vwn_rpa` (8) — *not*
    /// VWN3 (that is libxc's separate `b3lyp3`/394) and *not* VWN5 (`b3lyp5`/475).
    HybGgaXcB3lyp,
    /// PBE0 / PBEH (libxc 406).
    HybGgaXcPbeh,
    /// B3LYP with VWN5 instead of RPA (libxc 475).
    HybGgaXcB3lyp5,
    /// Minnesota M06-2X hybrid exchange — hybrid meta-GGA (libxc 450), 54% EXX.
    /// The M05 functional form (PBE-x enhancement × kinetic series; no VS98
    /// part), Zhao & Truhlar, Theor. Chem. Acc. 120, 215 (2008).
    HybMggaXM062x,
    /// Minnesota M06-2X correlation — meta-GGA (libxc 236). Same functional form
    /// as `mgga_c_m06_l`, different parameter set.
    MggaCM062x,
    /// PW6B95 — hybrid meta-GGA XC (libxc 451), 28% EXX. `0.72·mPW91-x(PW6
    /// params) + B95-c(PW6 params)`, Zhao & Truhlar, J. Phys. Chem. A 109, 5656
    /// (2005).
    HybMggaXcPw6b95,
    /// B97M-V — meta-GGA XC with VV10 nonlocal correlation (libxc 254).
    /// Mardirossian & Head-Gordon, J. Chem. Phys. 142, 074111 (2015). xcx
    /// evaluates the semilocal part; the host adds VV10 from
    /// [`Vv10Params`] `{ b: 6.0, c: 0.01 }`.
    MggaXcB97mV,
    /// ωB97X-V — range-separated hybrid GGA XC with VV10 (libxc 466).
    /// Mardirossian & Head-Gordon, Phys. Chem. Chem. Phys. 16, 9904 (2014).
    /// CAM (xcx convention): ω = 0.30, α = 0.167, β = 0.833; VV10 b = 6.0,
    /// C = 0.01. xcx evaluates the SR-attenuated semilocal part only.
    HybGgaXcWb97xV,
    /// ωB97M-V — range-separated hybrid meta-GGA XC with VV10 (libxc 531).
    /// Mardirossian & Head-Gordon, J. Chem. Phys. 144, 214110 (2016).
    /// CAM (xcx convention): ω = 0.30, α = 0.15, β = 0.85; VV10 b = 6.0,
    /// C = 0.01. xcx evaluates the SR-attenuated semilocal part only.
    HybMggaXcWb97mV,
    /// Perdew 86 correlation (libxc 132): PZ81 LDA correlation + gradient term.
    GgaCP86,
    /// B2PLYP — the original Grimme double hybrid (Grimme, J. Chem. Phys. 124,
    /// 034108 (2006)). Semilocal part `0.47·B88-x + 0.73·LYP-c`; EXX 0.53; PT2
    /// `c_os = c_ss = 0.27`. **Not in libxc** (no libxc release ships double
    /// hybrids); xcx-private id 100001 (see [`FunctionalId::as_u32`]).
    HybGgaXcB2plyp,
    /// revDSD-PBEP86-D4 semilocal core (Santra, Sylvetsky & Martin, J. Phys.
    /// Chem. A 123, 5129 (2019)). Semilocal part `0.31·PBE-x + 0.4210·P86-c`;
    /// EXX 0.69; PT2 `c_os = 0.5922, c_ss = 0.0636`. The host adds the D4
    /// dispersion term (param_set "revdsd-pbep86"). **Not in libxc**;
    /// xcx-private id 100002.
    HybGgaXcRevdsdPbep86D4,
    /// PWPB95 — Goerigk & Grimme double hybrid (J. Chem. Theory Comput. 7, 291
    /// (2011)). Semilocal part `0.50·mPW-x(reopt.) + 0.731·B95-c(reopt.)`; EXX
    /// 0.50; PT2 `c_os = 0.269, c_ss = 0` (SOS-PT2). **Not in libxc**;
    /// xcx-private id 100003.
    HybMggaXcPwpb95,
    /// ωB97M(2) — range-separated double hybrid on the ωB97M-V machinery
    /// (Mardirossian & Head-Gordon, J. Chem. Phys. 148, 241736 (2018)).
    /// CAM ω = 0.30, α = 0.62194, β = 0.37806 (paper Table II); PT2
    /// `c_os = c_ss = c_PT2 = 0.34096`; VV10 retained (b = 6.0, C = 0.01),
    /// scaled by the host as `c_VV10 = 1 − c_PT2 = 0.65904` (the paper's
    /// constraint). **Not in libxc**; xcx-private id 100004.
    HybMggaXcWb97m2,
    /// B97-3c xc part (libxc 327, `gga_xc_b97_3c`): the Becke-1997 power
    /// series refit by Brandenburg, Bannwarth, Hansen & Grimme, J. Chem.
    /// Phys. 148, 064104 (2018) (Table I; three terms per series, **no exact
    /// exchange** — a pure GGA). The composite method's D3(BJ)/ATM and SRB
    /// corrections are the host's job; `dispersion()` is `None`.
    GgaXcB973c,
    /// PBEh-3c xc part (Grimme, Brandenburg, Bannwarth & Hansen, J. Chem.
    /// Phys. 143, 054107 (2015)): global hybrid on modified PBE —
    /// `0.58·PBE-x(κ = 1.0245, μ = 10/81) + PBE-c(β = 0.03)` semilocal mix,
    /// EXX 0.42 via metadata. The composite method's gCP and D3 corrections
    /// are the host's job; `dispersion()` is `None`. **Not in libxc 6.1.0**;
    /// xcx-private id 100005.
    HybGgaXcPbeh3c,
}

impl FunctionalId {
    /// All functionals known to this build.
    pub const ALL: &'static [FunctionalId] = {
        use FunctionalId::*;
        &[
            LdaX,
            LdaCPw,
            LdaCVwn,
            LdaCVwn3,
            LdaCVwnRpa,
            GgaXPbe,
            GgaXB88,
            GgaCPbe,
            GgaCLyp,
            GgaXPbeR,
            GgaXPbeSol,
            GgaXRpbe,
            GgaCPbeSol,
            MggaXTpss,
            MggaCTpss,
            MggaXR2scan,
            MggaCR2scan,
            MggaXM06L,
            MggaCM06L,
            HybGgaXcB3lyp,
            HybGgaXcPbeh,
            HybGgaXcB3lyp5,
            HybMggaXM062x,
            MggaCM062x,
            HybMggaXcPw6b95,
            MggaXcB97mV,
            HybGgaXcWb97xV,
            HybMggaXcWb97mV,
            GgaCP86,
            HybGgaXcB2plyp,
            HybGgaXcRevdsdPbep86D4,
            HybMggaXcPwpb95,
            HybMggaXcWb97m2,
            GgaXcB973c,
            HybGgaXcPbeh3c,
        ]
    };

    /// The libxc numeric id — except for functionals **absent from libxc**
    /// (libxc ships no double hybrids), which use the xcx-private namespace
    /// `≥ 100000` (far above libxc's id range, currently < 1000): B2PLYP
    /// (100001), revDSD-PBEP86-D4 (100002), PWPB95 (100003), ωB97M(2) (100004),
    /// PBEh-3c (100005 — libxc 6.1.0 ships B97-3c but not PBEh-3c).
    /// Should libxc ever add them, `from_name` keeps resolving the canonical
    /// names; the numeric values stay xcx-private.
    pub fn as_u32(self) -> u32 {
        use FunctionalId::*;
        match self {
            LdaX => 1,
            LdaCPw => 12,
            LdaCVwn => 7,
            LdaCVwn3 => 30,
            LdaCVwnRpa => 8,
            GgaXPbe => 101,
            GgaXB88 => 106,
            GgaCPbe => 130,
            GgaCLyp => 131,
            GgaXPbeR => 102,
            GgaXPbeSol => 116,
            GgaXRpbe => 117,
            GgaCPbeSol => 133,
            MggaXTpss => 202,
            MggaCTpss => 231,
            MggaXR2scan => 497,
            MggaCR2scan => 498,
            MggaXM06L => 203,
            MggaCM06L => 233,
            HybGgaXcB3lyp => 402,
            HybGgaXcPbeh => 406,
            HybGgaXcB3lyp5 => 475,
            HybMggaXM062x => 450,
            MggaCM062x => 236,
            HybMggaXcPw6b95 => 451,
            MggaXcB97mV => 254,
            HybGgaXcWb97xV => 466,
            HybMggaXcWb97mV => 531,
            GgaCP86 => 132,
            HybGgaXcB2plyp => 100_001,
            HybGgaXcRevdsdPbep86D4 => 100_002,
            HybMggaXcPwpb95 => 100_003,
            HybMggaXcWb97m2 => 100_004,
            GgaXcB973c => 327,
            HybGgaXcPbeh3c => 100_005,
        }
    }

    /// Look up a functional by its libxc numeric id.
    pub fn from_u32(id: u32) -> Option<Self> {
        FunctionalId::ALL.iter().copied().find(|f| f.as_u32() == id)
    }

    /// The canonical lowercase libxc name (e.g. `"gga_x_pbe"`).
    pub fn name(self) -> &'static str {
        use FunctionalId::*;
        match self {
            LdaX => "lda_x",
            LdaCPw => "lda_c_pw",
            LdaCVwn => "lda_c_vwn",
            LdaCVwn3 => "lda_c_vwn_3",
            LdaCVwnRpa => "lda_c_vwn_rpa",
            GgaXPbe => "gga_x_pbe",
            GgaXB88 => "gga_x_b88",
            GgaCPbe => "gga_c_pbe",
            GgaCLyp => "gga_c_lyp",
            GgaXPbeR => "gga_x_pbe_r",
            GgaXPbeSol => "gga_x_pbe_sol",
            GgaXRpbe => "gga_x_rpbe",
            GgaCPbeSol => "gga_c_pbe_sol",
            MggaXTpss => "mgga_x_tpss",
            MggaCTpss => "mgga_c_tpss",
            MggaXR2scan => "mgga_x_r2scan",
            MggaCR2scan => "mgga_c_r2scan",
            MggaXM06L => "mgga_x_m06_l",
            MggaCM06L => "mgga_c_m06_l",
            HybGgaXcB3lyp => "hyb_gga_xc_b3lyp",
            HybGgaXcPbeh => "hyb_gga_xc_pbeh",
            HybGgaXcB3lyp5 => "hyb_gga_xc_b3lyp5",
            HybMggaXM062x => "hyb_mgga_x_m06_2x",
            MggaCM062x => "mgga_c_m06_2x",
            HybMggaXcPw6b95 => "hyb_mgga_xc_pw6b95",
            MggaXcB97mV => "mgga_xc_b97m_v",
            HybGgaXcWb97xV => "hyb_gga_xc_wb97x_v",
            HybMggaXcWb97mV => "hyb_mgga_xc_wb97m_v",
            GgaCP86 => "gga_c_p86",
            HybGgaXcB2plyp => "hyb_gga_xc_b2plyp",
            HybGgaXcRevdsdPbep86D4 => "hyb_gga_xc_revdsd_pbep86_d4",
            HybMggaXcPwpb95 => "hyb_mgga_xc_pwpb95",
            HybMggaXcWb97m2 => "hyb_mgga_xc_wb97m_2",
            GgaXcB973c => "gga_xc_b97_3c",
            HybGgaXcPbeh3c => "hyb_gga_xc_pbeh_3c",
        }
    }

    /// Look up a functional by name. Accepts the canonical libxc name plus a few
    /// common aliases (e.g. `"pbe0"`).
    pub fn from_name(name: &str) -> Option<Self> {
        use FunctionalId::*;
        Some(match name {
            "lda_x" | "slater" => LdaX,
            "lda_c_pw" | "pw92" | "pw" => LdaCPw,
            "lda_c_vwn" | "lda_c_vwn_5" => LdaCVwn,
            "lda_c_vwn_3" => LdaCVwn3,
            "lda_c_vwn_rpa" => LdaCVwnRpa,
            "gga_x_pbe" => GgaXPbe,
            "gga_x_b88" => GgaXB88,
            "gga_c_pbe" => GgaCPbe,
            "gga_c_lyp" => GgaCLyp,
            "gga_x_pbe_r" | "revpbe" => GgaXPbeR,
            "gga_x_pbe_sol" => GgaXPbeSol,
            "gga_x_rpbe" => GgaXRpbe,
            "gga_c_pbe_sol" => GgaCPbeSol,
            "mgga_x_tpss" => MggaXTpss,
            "mgga_c_tpss" => MggaCTpss,
            "mgga_x_r2scan" => MggaXR2scan,
            "mgga_c_r2scan" => MggaCR2scan,
            "mgga_x_m06_l" | "mgga_x_m06l" => MggaXM06L,
            "mgga_c_m06_l" | "mgga_c_m06l" => MggaCM06L,
            "hyb_gga_xc_b3lyp" => HybGgaXcB3lyp,
            "hyb_gga_xc_pbeh" | "hyb_gga_xc_pbe0" | "pbe0" => HybGgaXcPbeh,
            "hyb_gga_xc_b3lyp5" => HybGgaXcB3lyp5,
            "hyb_mgga_x_m06_2x" => HybMggaXM062x,
            "mgga_c_m06_2x" => MggaCM062x,
            "hyb_mgga_xc_pw6b95" | "pw6b95" => HybMggaXcPw6b95,
            "mgga_xc_b97m_v" | "b97m-v" | "b97m_v" => MggaXcB97mV,
            "hyb_gga_xc_wb97x_v" | "wb97x-v" | "wb97x_v" => HybGgaXcWb97xV,
            "hyb_mgga_xc_wb97m_v" | "wb97m-v" | "wb97m_v" => HybMggaXcWb97mV,
            "gga_c_p86" => GgaCP86,
            "hyb_gga_xc_b2plyp" | "b2plyp" => HybGgaXcB2plyp,
            "hyb_gga_xc_revdsd_pbep86_d4" | "revdsd-pbep86-d4" | "revdsd_pbep86_d4" => {
                HybGgaXcRevdsdPbep86D4
            }
            "hyb_mgga_xc_pwpb95" | "pwpb95" => HybMggaXcPwpb95,
            "hyb_mgga_xc_wb97m_2" | "wb97m(2)" | "wb97m-2" | "wb97m_2" => HybMggaXcWb97m2,
            "gga_xc_b97_3c" | "b97-3c" | "b97_3c" => GgaXcB973c,
            "hyb_gga_xc_pbeh_3c" | "pbeh-3c" | "pbeh_3c" | "pbeh3c" => HybGgaXcPbeh3c,
            _ => return None,
        })
    }
}

/// Range-separation (CAM) parameters: the host builds short/long-range exact
/// exchange from these.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CamParams {
    /// Range-separation parameter ω.
    pub omega: f64,
    /// Fraction of full-range exact exchange α.
    pub alpha: f64,
    /// Fraction of long-range exact exchange β.
    pub beta: f64,
}

/// VV10 nonlocal-correlation parameters. `xcx` exposes these but never computes
/// the nonlocal integral.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Vv10Params {
    /// VV10 `b` parameter.
    pub b: f64,
    /// VV10 `C` parameter.
    pub c: f64,
}

/// Mixing information for hybrid functionals.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct HybridInfo {
    /// Global fraction of exact (Hartree–Fock) exchange the host must add.
    pub exx_fraction: f64,
    /// Range-separation parameters, if any.
    pub cam: Option<CamParams>,
    /// VV10 parameters, if any.
    pub vv10: Option<Vv10Params>,
}

/// Static metadata describing a functional. See `docs/api-convention.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FunctionalInfo {
    /// Identifier, or `None` for a user-built linear mix.
    pub id: Option<FunctionalId>,
    /// Human-readable name.
    pub name: &'static str,
    /// Functional family.
    pub family: Family,
    /// Exchange / correlation / both / kinetic.
    pub kind: Kind,
    /// Whether `sigma` is required.
    pub needs_sigma: bool,
    /// Whether `lapl` is required.
    pub needs_lapl: bool,
    /// Whether `tau` is required.
    pub needs_tau: bool,
    /// Total-density threshold below which outputs are exactly zero.
    pub dens_threshold: f64,
    /// Hybrid mixing info, if this is a hybrid.
    pub hybrid: Option<HybridInfo>,
}

/// Rung of "Jacob's ladder" a functional sits on, as the host should treat it
/// (so a hybrid meta-GGA reports [`Rung::Hybrid`], not `MetaGga`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rung {
    /// Local density approximation.
    Lda,
    /// Generalized gradient approximation.
    Gga,
    /// Meta-GGA (τ- and/or Laplacian-dependent).
    MetaGga,
    /// Global hybrid (a single exact-exchange fraction).
    Hybrid,
    /// Range-separated hybrid (CAM parameters present; EXX(r₁₂) = α + β·erf(ω·r₁₂)).
    RangeSeparatedHybrid,
    /// Double hybrid (MP2-like correlation; none registered this round).
    DoubleHybrid,
}

/// Dispersion-correction model a functional is canonically paired with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispersionModel {
    /// Grimme D3 with Becke–Johnson damping.
    D3Bj,
    /// Grimme D4.
    D4,
    /// Vydrov–Van Voorhis nonlocal correlation (parameters in
    /// [`HybridInfo::vv10`]; xcx never evaluates the nonlocal integral).
    Vv10,
    /// No dispersion correction.
    None,
}

/// A recommended dispersion pairing: the model plus the named parameter set the
/// host should look up (e.g. a damping-parameter table keyed by functional).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispersionRec {
    /// Dispersion model.
    pub model: DispersionModel,
    /// Parameter-set key (canonical lowercase functional name).
    pub param_set: &'static str,
}

/// Integration-grid recommendation for the host's numerical quadrature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridRec {
    /// Coarse grid-quality level, `0..=4` (0 = coarsest, 4 = finest). Standard
    /// LDA/GGA/hybrids are level 3; grid-sensitive functionals are level 4.
    pub level: u8,
    /// Whether the functional is known to be unusually grid-sensitive.
    ///
    /// `true` for the Minnesota meta-GGAs (M06-L, M06-2X): their highly
    /// parameterized kinetic-energy enhancements oscillate with the integration
    /// grid, requiring substantially finer grids than e.g. PBE/TPSS for converged
    /// energies — see S. E. Wheeler & K. N. Houk, *J. Chem. Theory Comput.* 2010,
    /// 6, 395, and N. Mardirossian & M. Head-Gordon, *Mol. Phys.* 2017, 115, 2315.
    pub grid_sensitive: bool,
}

/// Double-hybrid PT2 mixing coefficients (none registered this round).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleHybridParams {
    /// Opposite-spin PT2 coefficient.
    pub c_os: f64,
    /// Same-spin PT2 coefficient.
    pub c_ss: f64,
}

impl FunctionalInfo {
    /// The rung of Jacob's ladder, derived from [`Family`] and the hybrid info.
    /// Hybrids with CAM parameters report [`Rung::RangeSeparatedHybrid`]; global
    /// hybrids (nonzero `exx_fraction`) report [`Rung::Hybrid`]; pure functionals
    /// report their family's semilocal rung — including a pure functional whose
    /// `hybrid` record exists only to carry VV10 parameters (B97M-V: `exx = 0`,
    /// no CAM ⇒ [`Rung::MetaGga`]). No registered functional is a double hybrid
    /// this round.
    pub fn rung(&self) -> Rung {
        // Double hybrids first: a registered functional carrying PT2
        // coefficients is `DoubleHybrid` regardless of its CAM/EXX details
        // (ωB97M(2) is range-separated *and* double-hybrid; the rung reports
        // the highest applicable treatment the host must provide).
        if self.double_hybrid().is_some() {
            return Rung::DoubleHybrid;
        }
        if let Some(h) = &self.hybrid {
            if h.cam.is_some() {
                return Rung::RangeSeparatedHybrid;
            }
            if h.exx_fraction != 0.0 || matches!(self.family, Family::HybGga | Family::HybMgga) {
                return Rung::Hybrid;
            }
        }
        match self.family {
            Family::Lda => Rung::Lda,
            Family::Gga => Rung::Gga,
            Family::Mgga => Rung::MetaGga,
            // A Hyb* family without `hybrid` info only occurs for user mixes of
            // pure parts (exx = 0 still carries `hybrid: Some` from `mix`).
            Family::HybGga => Rung::Hybrid,
            Family::HybMgga => Rung::Hybrid,
        }
    }

    /// Recommended dispersion pairing.
    ///
    /// - **VV10** where it is part of the functional's *definition* (B97M-V /
    ///   ωB97X-V / ωB97M-V / ωB97M(2)): the `b`/`C` parameters live in
    ///   [`HybridInfo::vv10`]; `param_set` is the canonical functional name.
    ///   For ωB97M(2) the host additionally scales the VV10 energy by
    ///   `c_VV10 = 1 − c_PT2` (see [`Self::double_hybrid`]).
    /// - **D4** where a published D4(BJ,EEQ-ATM) damping-parameter set exists
    ///   (Caldeweyher et al., *J. Chem. Phys.* 150, 154122 (2019); the
    ///   `dftd4` reference parameter table). `param_set` is the
    ///   **dftd4-convention key** (lowercase functional name, e.g. `"b3lyp"`,
    ///   `"pbe0"`, `"r2scan"`, `"b2plyp"`) that the host resolves against its
    ///   own dftd4 damping-parameter table — xcx ships no dispersion data or
    ///   code (scope fence). Component ids map to the canonical functional
    ///   their name denotes (e.g. `mgga_x_r2scan`/`mgga_c_r2scan` → `"r2scan"`,
    ///   `gga_x_pbe_r` → `"revpbe"`); components whose parent pairing is
    ///   ambiguous (B88, LYP, P86) and functionals without a published D4 set
    ///   (the LDAs, M06-2X) return `None`.
    pub fn dispersion(&self) -> Option<DispersionRec> {
        if self.hybrid.as_ref().is_some_and(|h| h.vv10.is_some()) {
            return Some(DispersionRec {
                model: DispersionModel::Vv10,
                param_set: self.name,
            });
        }
        use FunctionalId::*;
        let d4_key = match self.id? {
            GgaXPbe | GgaCPbe => "pbe",
            GgaXPbeR => "revpbe",
            GgaXRpbe => "rpbe",
            GgaXPbeSol | GgaCPbeSol => "pbesol",
            MggaXTpss | MggaCTpss => "tpss",
            MggaXR2scan | MggaCR2scan => "r2scan",
            MggaXM06L | MggaCM06L => "m06l",
            // B3LYP5 differs from B3LYP only in the VWN flavor; the published
            // dftd4 set is keyed "b3lyp" and applies to both.
            HybGgaXcB3lyp | HybGgaXcB3lyp5 => "b3lyp",
            HybGgaXcPbeh => "pbe0",
            HybMggaXcPw6b95 => "pw6b95",
            HybGgaXcB2plyp => "b2plyp",
            HybGgaXcRevdsdPbep86D4 => "revdsdpbep86",
            HybMggaXcPwpb95 => "pwpb95",
            _ => return None,
        };
        Some(DispersionRec {
            model: DispersionModel::D4,
            param_set: d4_key,
        })
    }

    /// Integration-grid recommendation. Level 3 (`grid_sensitive: false`) for
    /// standard LDA/GGA/meta-GGA/hybrids; level 4 (`grid_sensitive: true`) for
    /// the Minnesota M06 family (M06-L, M06-2X exchange and correlation), whose
    /// grid sensitivity is well documented (Wheeler & Houk, JCTC 2010, 6, 395;
    /// Mardirossian & Head-Gordon, Mol. Phys. 2017, 115, 2315), and for the
    /// B97/Minnesota-class combinatorially-optimized B97M-V / ωB97X-V / ωB97M-V,
    /// whose highly-parameterized inhomogeneity expansions likewise need fine
    /// grids for converged energies (Mardirossian & Head-Gordon, J. Chem. Theory
    /// Comput. 2016, 12, 4303, §"grid sensitivity" for B97M-V; Mol. Phys. 2017,
    /// 115, 2315).
    pub fn grid(&self) -> GridRec {
        let minnesota = matches!(
            self.id,
            Some(FunctionalId::MggaXM06L)
                | Some(FunctionalId::MggaCM06L)
                | Some(FunctionalId::HybMggaXM062x)
                | Some(FunctionalId::MggaCM062x)
                | Some(FunctionalId::MggaXcB97mV)
                | Some(FunctionalId::HybGgaXcWb97xV)
                | Some(FunctionalId::HybMggaXcWb97mV)
                | Some(FunctionalId::HybMggaXcWb97m2)
        );
        if minnesota {
            GridRec {
                level: 4,
                grid_sensitive: true,
            }
        } else {
            GridRec {
                level: 3,
                grid_sensitive: false,
            }
        }
    }

    /// Double-hybrid PT2 parameters: the same-spin / opposite-spin MP2-like
    /// correlation coefficients the host applies to its own PT2 energy (xcx
    /// never evaluates PT2 — scope fence). `Some` only for the registered
    /// double hybrids; values are the published ones (see each id's docs):
    /// B2PLYP `c_os = c_ss = 0.27`; revDSD-PBEP86-D4 `c_os = 0.5922,
    /// c_ss = 0.0636`; PWPB95 `c_os = 0.269, c_ss = 0` (SOS-PT2); ωB97M(2)
    /// `c_os = c_ss = c_PT2 = 0.34096` (a single canonical-MP2 coefficient;
    /// its VV10 partner is scaled by `c_VV10 = 1 − c_PT2`, the paper's
    /// constraint).
    pub fn double_hybrid(&self) -> Option<DoubleHybridParams> {
        match self.id {
            Some(FunctionalId::HybGgaXcB2plyp) => Some(DoubleHybridParams {
                c_os: 0.27,
                c_ss: 0.27,
            }),
            Some(FunctionalId::HybGgaXcRevdsdPbep86D4) => Some(DoubleHybridParams {
                c_os: 0.5922,
                c_ss: 0.0636,
            }),
            Some(FunctionalId::HybMggaXcPwpb95) => Some(DoubleHybridParams {
                c_os: 0.269,
                c_ss: 0.0,
            }),
            Some(FunctionalId::HybMggaXcWb97m2) => Some(DoubleHybridParams {
                c_os: 0.34096,
                c_ss: 0.34096,
            }),
            _ => None,
        }
    }
}

/// A ready-to-evaluate functional bound to a spin treatment.
pub struct Functional {
    spin: Spin,
    eval: Box<dyn XcEval>,
}

impl Functional {
    /// Construct a functional by id for the given spin treatment.
    pub fn new(id: FunctionalId, spin: Spin) -> Result<Self, XcError> {
        let eval = crate::functionals::build(id)?;
        Ok(Self { spin, eval })
    }

    /// Construct a functional by libxc name (or known alias).
    pub fn by_name(name: &str, spin: Spin) -> Result<Self, XcError> {
        let id = FunctionalId::from_name(name).ok_or(XcError::UnknownFunctional)?;
        Self::new(id, spin)
    }

    /// Construct a **parameterized PBE exchange** with asymptotic enhancement
    /// bound `kappa` and gradient coefficient `mu` (the literature s-space μ).
    /// Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).
    ///
    /// Routes through the exact same code path as the named PBE-x family, so
    /// the published parameter sets reproduce them **bitwise**: PBE
    /// (κ = 0.8040, μ = 0.06672455060314922·π²/3), revPBE (κ = 1.245, same μ),
    /// PBEsol (κ = 0.8040, μ = 10/81). PBEh-3c's modified exchange is
    /// `pbe_x(1.0245, 10.0/81.0, …)` (Grimme et al., *J. Chem. Phys.* **143**,
    /// 054107 (2015)).
    pub fn pbe_x(kappa: f64, mu: f64, spin: Spin) -> Functional {
        Functional {
            spin,
            eval: crate::functionals::pbe_x_param(kappa, mu),
        }
    }

    /// Construct a **parameterized PBE correlation** with gradient coefficient
    /// `beta` (γ stays the PBE constant `(1 − ln 2)/π²`, as in every published
    /// β-modified PBE-c). Perdew, Burke & Ernzerhof, *Phys. Rev. Lett.* **77**,
    /// 3865 (1996).
    ///
    /// Routes through the exact same code path as the named PBE-c family:
    /// `pbe_c(0.06672455060314922, …)` is bitwise `gga_c_pbe`,
    /// `pbe_c(0.046, …)` bitwise `gga_c_pbe_sol`. PBEh-3c's modified
    /// correlation is `pbe_c(0.03, …)` (Grimme et al., *J. Chem. Phys.*
    /// **143**, 054107 (2015)).
    pub fn pbe_c(beta: f64, spin: Spin) -> Functional {
        Functional {
            spin,
            eval: crate::functionals::pbe_c_param(beta),
        }
    }

    /// Construct a **B97-type GGA exchange–correlation power series** (Becke,
    /// *J. Chem. Phys.* **107**, 8554 (1997)): each ingredient is its parent
    /// LDA times `g = Σ_k c_k·u^k`, `u = γs²/(1 + γs²)`, with Becke's fixed
    /// γ_x = 0.004, γ_ss = 0.2, γ_os = 0.006 and caller-supplied series
    /// coefficients (any truncation): `c_x` for exchange (on per-channel LDA
    /// exchange), `c_ss` for same-spin and `c_os` for opposite-spin
    /// correlation (on the Stoll split of standard PW92).
    ///
    /// This is the form behind B97/B97-1/B97-2/HCTH/B97-3c; the named
    /// [`FunctionalId::GgaXcB973c`] is exactly this constructor with the
    /// Brandenburg et al. 2018 Table-I coefficients. The series carries **no
    /// exact exchange**; hybrid B97 variants are expressed by the host adding
    /// EXX on top (e.g. the original B97's 0.1943).
    pub fn b97_xc(c_x: &[f64], c_ss: &[f64], c_os: &[f64], spin: Spin) -> Functional {
        Functional {
            spin,
            eval: crate::functionals::b97_series(c_x, c_ss, c_os),
        }
    }

    /// Metadata for this functional.
    pub fn info(&self) -> &FunctionalInfo {
        self.eval.info()
    }

    /// The spin treatment this functional was built for.
    pub fn spin(&self) -> Spin {
        self.spin
    }

    /// Fraction of exact exchange the host must add (0.0 for pure functionals).
    pub fn exx_fraction(&self) -> f64 {
        self.info().hybrid.map_or(0.0, |h| h.exx_fraction)
    }

    /// Evaluate energy per particle and all available first derivatives over
    /// `np` points. Inputs follow the packing in `docs/api-convention.md`.
    pub fn eval(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        self.eval.eval(self.spin, np, input)
    }

    /// Evaluate energy, first derivatives, **and** second derivatives (`fxc`)
    /// over `np` points. Fills the same fields as [`eval`](Self::eval) plus
    /// `v2rho2` / `v2rhosigma` / `v2sigma2` (see `docs/api-convention.md` §3 for
    /// packing). Costs more than [`eval`](Self::eval); call it only when the
    /// second derivatives are needed (e.g. TDDFT / response properties).
    pub fn eval_fxc(&self, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        self.eval.eval_fxc(self.spin, np, input)
    }

    /// Build the linear combination `Σ wᵢ·fᵢ` of functionals, which must share a
    /// spin treatment. This is the only composition `xcx` performs.
    pub fn mix(parts: Vec<(f64, Functional)>) -> Result<Functional, XcError> {
        let spin = parts.first().ok_or(XcError::SpinMismatch)?.1.spin;
        if parts.iter().any(|(_, f)| f.spin != spin) {
            return Err(XcError::SpinMismatch);
        }
        let exx = parts.iter().map(|(w, f)| w * f.exx_fraction()).sum();
        let info = FunctionalInfo {
            id: None,
            name: "mixed",
            family: Family::HybGga,
            kind: Kind::ExchangeCorrelation,
            needs_sigma: parts.iter().any(|(_, f)| f.info().needs_sigma),
            needs_lapl: parts.iter().any(|(_, f)| f.info().needs_lapl),
            needs_tau: parts.iter().any(|(_, f)| f.info().needs_tau),
            dens_threshold: parts
                .iter()
                .map(|(_, f)| f.info().dens_threshold)
                .fold(f64::INFINITY, f64::min),
            hybrid: Some(HybridInfo {
                exx_fraction: exx,
                cam: None,
                vv10: None,
            }),
        };
        let weighted: Vec<(f64, Box<dyn XcEval>)> =
            parts.into_iter().map(|(w, f)| (w, f.eval)).collect();
        Ok(Functional {
            spin,
            eval: mixed_eval(weighted, info),
        })
    }
}

/// Build a boxed linear-combination evaluator from already-boxed components and
/// the metadata to report. The shared mixing engine behind both
/// [`Functional::mix`] (synthetic `info`, `id = None`) and the registered hybrids
/// (their own `id`/`name`/`exx_fraction`). Components keep their own
/// `dens_threshold` (each screens itself at eval time), exactly as libxc's
/// `xc_mix` does.
pub(crate) fn mixed_eval(
    parts: Vec<(f64, Box<dyn XcEval>)>,
    info: FunctionalInfo,
) -> Box<dyn XcEval> {
    Box::new(MixEval { parts, info })
}

/// Evaluator for a linear mix of functionals.
struct MixEval {
    parts: Vec<(f64, Box<dyn XcEval>)>,
    info: FunctionalInfo,
}

impl XcEval for MixEval {
    fn info(&self) -> &FunctionalInfo {
        &self.info
    }

    fn eval(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let mut acc = XcResult::default();
        for (w, part) in &self.parts {
            accumulate(&mut acc, *w, &part.eval(spin, np, input)?);
        }
        Ok(acc)
    }

    fn eval_fxc(&self, spin: Spin, np: usize, input: &XcInput) -> Result<XcResult, XcError> {
        let mut acc = XcResult::default();
        for (w, part) in &self.parts {
            accumulate(&mut acc, *w, &part.eval_fxc(spin, np, input)?);
        }
        Ok(acc)
    }
}

/// Accumulate `acc += w·r` over every output field. fxc fields are accumulated
/// the same linear way as the first derivatives, so a hybrid inherits fxc from
/// its semilocal parts for free (the fields are empty when fxc was not
/// requested, so `eval` and `eval_fxc` share this one helper).
fn accumulate(acc: &mut XcResult, w: f64, r: &XcResult) {
    add_scaled(&mut acc.exc, w, &r.exc);
    add_scaled(&mut acc.vrho, w, &r.vrho);
    add_scaled(&mut acc.vsigma, w, &r.vsigma);
    add_scaled(&mut acc.vtau, w, &r.vtau);
    add_scaled(&mut acc.vlapl, w, &r.vlapl);
    add_scaled(&mut acc.v2rho2, w, &r.v2rho2);
    add_scaled(&mut acc.v2rhosigma, w, &r.v2rhosigma);
    add_scaled(&mut acc.v2sigma2, w, &r.v2sigma2);
    add_scaled(&mut acc.v2rhotau, w, &r.v2rhotau);
    add_scaled(&mut acc.v2sigmatau, w, &r.v2sigmatau);
    add_scaled(&mut acc.v2tau2, w, &r.v2tau2);
}

fn add_scaled(dst: &mut Vec<f64>, w: f64, src: &[f64]) {
    if src.len() > dst.len() {
        dst.resize(src.len(), 0.0);
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d += w * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::XcInput;

    #[test]
    fn id_roundtrips_and_matches_libxc_numbers() {
        for &id in FunctionalId::ALL {
            assert_eq!(FunctionalId::from_u32(id.as_u32()), Some(id));
            assert_eq!(FunctionalId::from_name(id.name()), Some(id));
        }
        assert_eq!(FunctionalId::GgaCPbe.as_u32(), 130);
        assert_eq!(FunctionalId::HybGgaXcB3lyp5.as_u32(), 475);
        assert_eq!(
            FunctionalId::from_name("pbe0"),
            Some(FunctionalId::HybGgaXcPbeh)
        );
    }

    // --- Metadata v2 (rung / dispersion / grid / double_hybrid) ---

    /// Every registered functional's rung must be consistent with its family
    /// (+ hybrid info): CAM carriers report `RangeSeparatedHybrid`, Hyb*
    /// families (and any nonzero exx) report `Hybrid`, pure families their
    /// semilocal rung — including B97M-V, whose `hybrid` record carries VV10
    /// only (exx = 0, no CAM ⇒ `MetaGga`).
    #[test]
    fn rung_consistent_with_family_for_all_ids() {
        for &id in FunctionalId::ALL {
            let f = Functional::new(id, Spin::Unpolarized).unwrap();
            let info = f.info();
            let rung = info.rung();
            let cam = info.hybrid.as_ref().is_some_and(|h| h.cam.is_some());
            let exx = info.hybrid.as_ref().map_or(0.0, |h| h.exx_fraction);
            let want = if info.double_hybrid().is_some() {
                Rung::DoubleHybrid
            } else if cam {
                Rung::RangeSeparatedHybrid
            } else if exx != 0.0 || matches!(info.family, Family::HybGga | Family::HybMgga) {
                Rung::Hybrid
            } else {
                match info.family {
                    Family::Lda => Rung::Lda,
                    Family::Gga => Rung::Gga,
                    _ => Rung::MetaGga,
                }
            };
            assert_eq!(
                rung,
                want,
                "{}: rung {rung:?} vs family-derived {want:?}",
                id.name()
            );
            // a Hyb* family must actually carry hybrid info; the only pure
            // family carrying a hybrid record is the VV10-only B97M-V
            if matches!(info.family, Family::HybGga | Family::HybMgga) {
                assert!(info.hybrid.is_some(), "{}: Hyb* without hybrid", id.name());
            } else if info.hybrid.is_some() {
                assert_eq!(
                    id,
                    FunctionalId::MggaXcB97mV,
                    "{}: unexpected hybrid record on a pure family",
                    id.name()
                );
            }
        }
    }

    /// Minnesota + B97/ωB97-class functionals report fine, grid-sensitive
    /// grids; everything else the standard level-3 recommendation. The
    /// VV10-containing functionals keep their (Vv10) pairing; functionals with
    /// a published D4 set report `D4` with the dftd4-convention key; only the
    /// four registered double hybrids report `double_hybrid()` parameters.
    #[test]
    fn grid_dispersion_double_hybrid_metadata() {
        use FunctionalId::*;
        let fine_grid = [
            MggaXM06L,
            MggaCM06L,
            HybMggaXM062x,
            MggaCM062x,
            MggaXcB97mV,
            HybGgaXcWb97xV,
            HybMggaXcWb97mV,
            HybMggaXcWb97m2,
        ];
        let vv10 = [
            MggaXcB97mV,
            HybGgaXcWb97xV,
            HybMggaXcWb97mV,
            HybMggaXcWb97m2,
        ];
        let d4: &[(FunctionalId, &str)] = &[
            (GgaXPbe, "pbe"),
            (GgaCPbe, "pbe"),
            (GgaXPbeR, "revpbe"),
            (GgaXRpbe, "rpbe"),
            (GgaXPbeSol, "pbesol"),
            (GgaCPbeSol, "pbesol"),
            (MggaXTpss, "tpss"),
            (MggaCTpss, "tpss"),
            (MggaXR2scan, "r2scan"),
            (MggaCR2scan, "r2scan"),
            (MggaXM06L, "m06l"),
            (MggaCM06L, "m06l"),
            (HybGgaXcB3lyp, "b3lyp"),
            (HybGgaXcB3lyp5, "b3lyp"),
            (HybGgaXcPbeh, "pbe0"),
            (HybMggaXcPw6b95, "pw6b95"),
            (HybGgaXcB2plyp, "b2plyp"),
            (HybGgaXcRevdsdPbep86D4, "revdsdpbep86"),
            (HybMggaXcPwpb95, "pwpb95"),
        ];
        let dh = [
            HybGgaXcB2plyp,
            HybGgaXcRevdsdPbep86D4,
            HybMggaXcPwpb95,
            HybMggaXcWb97m2,
        ];
        for &id in FunctionalId::ALL {
            let f = Functional::new(id, Spin::Unpolarized).unwrap();
            let info = f.info();
            let g = info.grid();
            if fine_grid.contains(&id) {
                assert_eq!((g.level, g.grid_sensitive), (4, true), "{}", id.name());
            } else {
                assert_eq!((g.level, g.grid_sensitive), (3, false), "{}", id.name());
            }
            assert!(g.level <= 4);
            if vv10.contains(&id) {
                let d = info.dispersion().unwrap();
                assert_eq!(d.model, DispersionModel::Vv10, "{}", id.name());
                assert_eq!(d.param_set, id.name());
                let p = info.hybrid.unwrap().vv10.unwrap();
                assert_eq!((p.b, p.c), (6.0, 0.01), "{}", id.name());
            } else if let Some(&(_, key)) = d4.iter().find(|&&(d4id, _)| d4id == id) {
                let d = info.dispersion().unwrap();
                assert_eq!(d.model, DispersionModel::D4, "{}", id.name());
                assert_eq!(d.param_set, key, "{}", id.name());
            } else {
                assert_eq!(info.dispersion(), None, "{}", id.name());
            }
            if dh.contains(&id) {
                assert!(info.double_hybrid().is_some(), "{}", id.name());
                assert_eq!(info.rung(), Rung::DoubleHybrid, "{}", id.name());
            } else {
                assert_eq!(info.double_hybrid(), None, "{}", id.name());
            }
        }
    }

    /// Pinned double-hybrid metadata: published EXX + PT2 coefficients, and the
    /// D4 spot checks the acceptance criteria call out (b3lyp / pbe0 / r2scan /
    /// b2plyp → Some(D4, key); lda → None).
    #[test]
    fn double_hybrid_and_d4_spot_checks() {
        let get = |id| Functional::new(id, Spin::Unpolarized).unwrap();
        // B2PLYP: 53% EXX, c_os = c_ss = 0.27 (Grimme 2006).
        let f = get(FunctionalId::HybGgaXcB2plyp);
        assert_eq!(f.exx_fraction(), 0.53);
        let p = f.info().double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.27, 0.27));
        assert_eq!(f.info().rung(), Rung::DoubleHybrid);
        // revDSD-PBEP86-D4: 69% EXX, c_os = 0.5922, c_ss = 0.0636 (Santra 2019).
        let f = get(FunctionalId::HybGgaXcRevdsdPbep86D4);
        assert_eq!(f.exx_fraction(), 0.69);
        let p = f.info().double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.5922, 0.0636));
        // PWPB95: 50% EXX, SOS-PT2 c_os = 0.269, c_ss = 0 (Goerigk & Grimme 2011).
        let f = get(FunctionalId::HybMggaXcPwpb95);
        assert_eq!(f.exx_fraction(), 0.50);
        let p = f.info().double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.269, 0.0));
        // ωB97M(2): RS double hybrid; CAM ω = 0.3, α = c_x = 0.62194,
        // β = 0.37806; PT2 c_os = c_ss = 0.34096; VV10 retained (scaled by
        // the host as 1 − c_PT2). Rung DoubleHybrid despite CAM.
        let f = get(FunctionalId::HybMggaXcWb97m2);
        let cam = f.info().hybrid.unwrap().cam.unwrap();
        assert_eq!(
            (cam.omega, cam.alpha, cam.beta),
            (0.3, 0.62194, 1.0 - 0.62194)
        );
        let p = f.info().double_hybrid().unwrap();
        assert_eq!((p.c_os, p.c_ss), (0.34096, 0.34096));
        assert_eq!(f.info().rung(), Rung::DoubleHybrid);
        // D4 spot checks (dftd4-convention keys).
        let d4 = |id| {
            let f = Functional::new(id, Spin::Unpolarized).unwrap();
            let d = f.info().dispersion().unwrap();
            assert_eq!(d.model, DispersionModel::D4);
            d.param_set
        };
        assert_eq!(d4(FunctionalId::HybGgaXcB3lyp), "b3lyp");
        assert_eq!(d4(FunctionalId::HybGgaXcPbeh), "pbe0");
        assert_eq!(d4(FunctionalId::MggaXR2scan), "r2scan");
        assert_eq!(d4(FunctionalId::HybGgaXcB2plyp), "b2plyp");
        let lda = get(FunctionalId::LdaX);
        assert_eq!(lda.info().dispersion(), None);
    }

    /// Pinned spot checks: PBE0 is a Hybrid with exx 0.25; M06-2X exchange a
    /// Hybrid with exx 0.54; PW6B95 a Hybrid with exx 0.28; PBE a pure Gga rung.
    #[test]
    fn rung_and_exx_spot_checks() {
        let pbe0 = Functional::new(FunctionalId::HybGgaXcPbeh, Spin::Unpolarized).unwrap();
        assert_eq!(pbe0.info().rung(), Rung::Hybrid);
        assert_eq!(pbe0.exx_fraction(), 0.25);
        let m062x = Functional::new(FunctionalId::HybMggaXM062x, Spin::Unpolarized).unwrap();
        assert_eq!(m062x.info().rung(), Rung::Hybrid);
        assert_eq!(m062x.exx_fraction(), 0.54);
        let pw6 = Functional::new(FunctionalId::HybMggaXcPw6b95, Spin::Unpolarized).unwrap();
        assert_eq!(pw6.info().rung(), Rung::Hybrid);
        assert_eq!(pw6.exx_fraction(), 0.28);
        let pbe = Functional::new(FunctionalId::GgaXPbe, Spin::Unpolarized).unwrap();
        assert_eq!(pbe.info().rung(), Rung::Gga);
        // Range-separated/VV10 functionals (CAM in the frozen convention
        // EXX(r₁₂) = α + β·erf(ω·r₁₂); exx_fraction is the global part α).
        let wxv = Functional::new(FunctionalId::HybGgaXcWb97xV, Spin::Unpolarized).unwrap();
        assert_eq!(wxv.info().rung(), Rung::RangeSeparatedHybrid);
        assert_eq!(wxv.exx_fraction(), 0.167);
        let cam = wxv.info().hybrid.unwrap().cam.unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.3, 0.167, 1.0 - 0.167));
        let wmv = Functional::new(FunctionalId::HybMggaXcWb97mV, Spin::Unpolarized).unwrap();
        assert_eq!(wmv.info().rung(), Rung::RangeSeparatedHybrid);
        assert_eq!(wmv.exx_fraction(), 0.15);
        let cam = wmv.info().hybrid.unwrap().cam.unwrap();
        assert_eq!((cam.omega, cam.alpha, cam.beta), (0.3, 0.15, 1.0 - 0.15));
        let b97mv = Functional::new(FunctionalId::MggaXcB97mV, Spin::Unpolarized).unwrap();
        assert_eq!(b97mv.info().rung(), Rung::MetaGga);
        assert_eq!(b97mv.exx_fraction(), 0.0);
    }

    /// The frozen metadata-v2 signatures compile exactly as specified.
    #[test]
    fn metadata_v2_signatures_compile() {
        fn takes(
            info: &FunctionalInfo,
        ) -> (
            Rung,
            Option<DispersionRec>,
            GridRec,
            Option<DoubleHybridParams>,
        ) {
            (
                info.rung(),
                info.dispersion(),
                info.grid(),
                info.double_hybrid(),
            )
        }
        let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        let (r, d, g, dh) = takes(f.info());
        assert_eq!(r, Rung::Lda);
        assert_eq!(d, None);
        assert_eq!(g.level, 3);
        assert_eq!(dh, None);
        // enum variants exist as frozen
        let _ = [
            Rung::Lda,
            Rung::Gga,
            Rung::MetaGga,
            Rung::Hybrid,
            Rung::RangeSeparatedHybrid,
            Rung::DoubleHybrid,
        ];
        let _ = [
            DispersionModel::D3Bj,
            DispersionModel::D4,
            DispersionModel::Vv10,
            DispersionModel::None,
        ];
        let _ = DispersionRec {
            model: DispersionModel::D3Bj,
            param_set: "b3lyp",
        };
        let _ = GridRec {
            level: 0,
            grid_sensitive: false,
        };
        let _ = DoubleHybridParams {
            c_os: 1.0,
            c_ss: 1.0,
        };
    }

    // --- Functional::mix mechanism (the linear-combination engine the hybrids
    // are built on) ---

    /// `mix([(1.0, f)])` must reproduce `f` exactly (no spurious scaling).
    #[test]
    fn mix_single_weight_one_is_identity() {
        let n = [0.7_f64];
        let s = [0.2_f64];
        let plain = Functional::new(FunctionalId::GgaXPbe, Spin::Unpolarized).unwrap();
        let want = plain.eval(1, &XcInput::gga(&n, &s)).unwrap();
        let mixed = Functional::mix(vec![(
            1.0,
            Functional::new(FunctionalId::GgaXPbe, Spin::Unpolarized).unwrap(),
        )])
        .unwrap();
        let got = mixed.eval(1, &XcInput::gga(&n, &s)).unwrap();
        assert_eq!(got.exc, want.exc);
        assert_eq!(got.vrho, want.vrho);
        assert_eq!(got.vsigma, want.vsigma);
    }

    /// Linear accumulation: `0.25·lda_x + 0.75·gga_x_pbe` must equal the weighted
    /// sum of the parts componentwise, and FD-agree on the mixed derivatives.
    /// Also exercises heterogeneous LDA+GGA mixing: lda_x contributes empty
    /// vsigma, so the mix's vsigma comes entirely from the GGA part.
    #[test]
    fn mix_accumulates_linearly_and_matches_fd() {
        let (wa, wb) = (0.25_f64, 0.75_f64);
        let build = || {
            Functional::mix(vec![
                (
                    wa,
                    Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap(),
                ),
                (
                    wb,
                    Functional::new(FunctionalId::GgaXPbe, Spin::Unpolarized).unwrap(),
                ),
            ])
            .unwrap()
        };
        let mixed = build();
        let lda = Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap();
        let pbe = Functional::new(FunctionalId::GgaXPbe, Spin::Unpolarized).unwrap();
        for &(n, s) in &[(0.5_f64, 0.1_f64), (2.0, 0.7), (10.0, 5.0)] {
            let rho = [n];
            let sg = [s];
            let inp = XcInput::gga(&rho, &sg);
            let m = mixed.eval(1, &inp).unwrap();
            let l = lda.eval(1, &XcInput::lda(&rho)).unwrap();
            let p = pbe.eval(1, &inp).unwrap();
            // componentwise weighted sum (lda_x has no vsigma → only pbe contributes)
            assert!((m.exc[0] - (wa * l.exc[0] + wb * p.exc[0])).abs() <= 1e-14 * m.exc[0].abs());
            assert!(
                (m.vrho[0] - (wa * l.vrho[0] + wb * p.vrho[0])).abs() <= 1e-14 * m.vrho[0].abs()
            );
            assert_eq!(m.vsigma.len(), 1);
            assert!((m.vsigma[0] - wb * p.vsigma[0]).abs() <= 1e-14 * m.vsigma[0].abs());
            // FD-check the mixed potentials directly (energy density e = n·exc)
            let edens =
                |n: f64, s: f64| n * mixed.eval(1, &XcInput::gga(&[n], &[s])).unwrap().exc[0];
            let hn = 1e-6 * n;
            let hs = 1e-6 * s;
            let fdn = (edens(n + hn, s) - edens(n - hn, s)) / (2.0 * hn);
            let fds = (edens(n, s + hs) - edens(n, s - hs)) / (2.0 * hs);
            assert!((m.vrho[0] - fdn).abs() <= 1e-6 * m.vrho[0].abs().max(1.0));
            assert!((m.vsigma[0] - fds).abs() <= 1e-6 * m.vsigma[0].abs().max(1.0));
        }
        // a pure-semilocal mix carries no exact exchange
        assert_eq!(mixed.exx_fraction(), 0.0);
    }

    /// Mixing functionals of different spin treatments is rejected.
    #[test]
    fn mix_spin_mismatch_errors() {
        let res = Functional::mix(vec![
            (
                0.5,
                Functional::new(FunctionalId::LdaX, Spin::Unpolarized).unwrap(),
            ),
            (
                0.5,
                Functional::new(FunctionalId::LdaX, Spin::Polarized).unwrap(),
            ),
        ]);
        assert!(matches!(res, Err(XcError::SpinMismatch)));
    }
}
