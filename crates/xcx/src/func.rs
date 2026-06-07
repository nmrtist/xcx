// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
    /// B3LYP, VWN_RPA convention (libxc 402). Mixes `lda_c_vwn_rpa` (8) — *not*
    /// VWN3 (that is libxc's separate `b3lyp3`/394) and *not* VWN5 (`b3lyp5`/475).
    HybGgaXcB3lyp,
    /// PBE0 / PBEH (libxc 406).
    HybGgaXcPbeh,
    /// B3LYP with VWN5 instead of RPA (libxc 475).
    HybGgaXcB3lyp5,
}

impl FunctionalId {
    /// All functionals known to this build (the v0.1 set).
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
            HybGgaXcB3lyp,
            HybGgaXcPbeh,
            HybGgaXcB3lyp5,
        ]
    };

    /// The libxc numeric id.
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
            HybGgaXcB3lyp => 402,
            HybGgaXcPbeh => 406,
            HybGgaXcB3lyp5 => 475,
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
            HybGgaXcB3lyp => "hyb_gga_xc_b3lyp",
            HybGgaXcPbeh => "hyb_gga_xc_pbeh",
            HybGgaXcB3lyp5 => "hyb_gga_xc_b3lyp5",
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
            "hyb_gga_xc_b3lyp" => HybGgaXcB3lyp,
            "hyb_gga_xc_pbeh" | "hyb_gga_xc_pbe0" | "pbe0" => HybGgaXcPbeh,
            "hyb_gga_xc_b3lyp5" => HybGgaXcB3lyp5,
            _ => return None,
        })
    }
}

/// Range-separation (CAM) parameters: the host builds short/long-range exact
/// exchange from these. None of the v0.1 functionals are range-separated.
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
/// the nonlocal integral. None of the v0.1 functionals use VV10.
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
            let r = part.eval(spin, np, input)?;
            add_scaled(&mut acc.exc, *w, &r.exc);
            add_scaled(&mut acc.vrho, *w, &r.vrho);
            add_scaled(&mut acc.vsigma, *w, &r.vsigma);
            add_scaled(&mut acc.vtau, *w, &r.vtau);
            add_scaled(&mut acc.vlapl, *w, &r.vlapl);
        }
        Ok(acc)
    }
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
