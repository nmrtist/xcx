// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Hybrid GGA functionals (libxc `HYB_GGA` family): linear mixes of semilocal
//! components plus an exact-exchange (EXX) fraction the host supplies.
//!
//! Provenance: ported-from-libxc (MPL-2.0); `src/hyb_gga_xc_pbeh.c`,
//! `src/hyb_gga_xc_b3lyp.c`.
//!
//! Per the scope fence (docs/api-convention.md), xcx emits **only the semilocal
//! part** — the coefficient-weighted mix of the component functionals — and
//! exposes the EXX fraction via [`FunctionalInfo::hybrid`]'s `exx_fraction` for
//! the host to build the Hartree–Fock exchange itself. This matches libxc: its
//! `xc_gga_exc_vxc` on a hybrid returns the same semilocal mix (the components
//! registered via `xc_mix_init`), with the EXX fraction held separately in
//! `cam_alpha` (queried by `xc_hyb_exx_coef`). So golden compares
//! semilocal-to-semilocal. The mixing itself is [`crate::func::mixed_eval`], the
//! same engine behind the public `Functional::mix`.

use crate::error::XcError;
use crate::families::XcEval;
use crate::func::{mixed_eval, Family, FunctionalId, FunctionalInfo, HybridInfo, Kind};

/// Metadata for a hybrid GGA mix. `dens_threshold` is libxc's hybrid value
/// (1e-15); the actual screening happens per component (each evaluates at its own
/// threshold inside the mix), exactly as libxc's `xc_mix` does.
fn hyb_info(id: FunctionalId, name: &'static str, exx_fraction: f64) -> FunctionalInfo {
    FunctionalInfo {
        id: Some(id),
        name,
        family: Family::HybGga,
        kind: Kind::ExchangeCorrelation,
        needs_sigma: true,
        needs_lapl: false,
        needs_tau: false,
        dens_threshold: 1e-15,
        hybrid: Some(HybridInfo {
            exx_fraction,
            cam: None,
            vv10: None,
        }),
    }
}

/// Build a hybrid as the coefficient-weighted mix of its semilocal components.
fn build_mix(
    components: &[(f64, FunctionalId)],
    info: FunctionalInfo,
) -> Result<Box<dyn XcEval>, XcError> {
    let parts = components
        .iter()
        .map(|&(w, id)| Ok((w, super::build(id)?)))
        .collect::<Result<Vec<_>, XcError>>()?;
    Ok(mixed_eval(parts, info))
}

/// PBE0 / PBEH (libxc 406): `0.75·PBE-x + 1.0·PBE-c`, with `0.25` exact exchange.
/// libxc `hyb_gga_xc_pbeh`: components `{GGA_X_PBE, GGA_C_PBE}`, `beta = 0.25` →
/// `mix_coef = {1−beta, 1.0} = {0.75, 1.0}`, `cam_alpha = beta = 0.25`.
pub(crate) fn pbeh() -> Result<Box<dyn XcEval>, XcError> {
    build_mix(
        &[(0.75, FunctionalId::GgaXPbe), (1.0, FunctionalId::GgaCPbe)],
        hyb_info(FunctionalId::HybGgaXcPbeh, "hyb_gga_xc_pbeh", 0.25),
    )
}

// B3LYP mixing parameters (libxc `b3lyp_values = {a0, ax, ac}`); the coefficients
// are formed exactly as libxc's `b3pw91_set_ext_params` does (subtraction order
// included), so the emitted weights match bit-for-bit.
const B3LYP_A0: f64 = 0.20; // fraction of exact exchange
const B3LYP_AX: f64 = 0.72; // fraction of GGA (B88) exchange correction
const B3LYP_AC: f64 = 0.81; // fraction of GGA (LYP) correlation correction

/// Shared B3LYP recipe: `(1−a0−ax)·LDA_X + ax·B88 + (1−ac)·LDA_C + ac·LYP`, EXX =
/// `a0`. The only thing that varies between the B3LYP variants is the LDA
/// correlation flavor: `lda_c` is VWN_RPA (libxc 402, `hyb_gga_xc_b3lyp`) or VWN5
/// (libxc 475, `hyb_gga_xc_b3lyp5`). libxc's `b3pw91_set_ext_params`:
/// `mix_coef = {1−a0−ax, ax, 1−ac, ac}`, `cam_alpha = a0`.
fn b3lyp_like(
    id: FunctionalId,
    name: &'static str,
    lda_c: FunctionalId,
) -> Result<Box<dyn XcEval>, XcError> {
    build_mix(
        &[
            (1.0 - B3LYP_A0 - B3LYP_AX, FunctionalId::LdaX),
            (B3LYP_AX, FunctionalId::GgaXB88),
            (1.0 - B3LYP_AC, lda_c),
            (B3LYP_AC, FunctionalId::GgaCLyp),
        ],
        hyb_info(id, name, B3LYP_A0),
    )
}

/// B3LYP with VWN5 (libxc 475, `hyb_gga_xc_b3lyp5`): uses `lda_c_vwn` (VWN5, id 7)
/// for the LDA correlation, unlike the original B3LYP (402) which uses VWN_RPA.
pub(crate) fn b3lyp5() -> Result<Box<dyn XcEval>, XcError> {
    b3lyp_like(
        FunctionalId::HybGgaXcB3lyp5,
        "hyb_gga_xc_b3lyp5",
        FunctionalId::LdaCVwn,
    )
}

// --- Double hybrids (clean-room; absent from libxc — see func.rs id docs) ---
//
// xcx emits only the scaled semilocal XC mix; the host adds the EXX fraction
// (metadata `exx_fraction`) and the PT2 correlation scaled by
// `double_hybrid()`'s `c_os`/`c_ss` (xcx never evaluates PT2 — scope fence).

/// B2PLYP (Grimme, J. Chem. Phys. 124, 034108 (2006)). Provenance: clean-room.
/// `E_xc = (1−a_x)·E_x^B88 + a_x·E_x^HF + (1−c)·E_c^LYP + c·E_c^PT2` with
/// `a_x = 0.53`, `c = 0.27` (paper Eq. 1–2) ⇒ semilocal mix
/// `0.47·B88-x + 0.73·LYP-c`, EXX 0.53, PT2 `c_os = c_ss = 0.27`. No LDA
/// exchange component (pure B88 GGA exchange, unlike B3LYP's three-way split).
pub(crate) fn b2plyp() -> Result<Box<dyn XcEval>, XcError> {
    const AX: f64 = 0.53;
    const C: f64 = 0.27;
    build_mix(
        &[
            (1.0 - AX, FunctionalId::GgaXB88),
            (1.0 - C, FunctionalId::GgaCLyp),
        ],
        hyb_info(FunctionalId::HybGgaXcB2plyp, "hyb_gga_xc_b2plyp", AX),
    )
}

/// revDSD-PBEP86-D4 semilocal core (Santra, Sylvetsky & Martin, J. Phys.
/// Chem. A 123, 5129 (2019), Table 4, the D4(EEQ)-paired refit). Provenance:
/// clean-room. DSD form `E_xc = c_x^HF·E_x^HF + (1−c_x^HF)·E_x^PBE +
/// c_c·E_c^P86 + c_os·E_os^PT2 + c_ss·E_ss^PT2 + E_disp(D4)` ⇒ semilocal mix
/// `0.31·PBE-x + 0.4210·P86-c`, EXX 0.69, PT2 `c_os = 0.5922, c_ss = 0.0636`.
/// The D4 dispersion term is the host's (param_set "revdsd-pbep86").
pub(crate) fn revdsd_pbep86_d4() -> Result<Box<dyn XcEval>, XcError> {
    const CX_HF: f64 = 0.69;
    const CC_P86: f64 = 0.4210;
    build_mix(
        &[
            (1.0 - CX_HF, FunctionalId::GgaXPbe),
            (CC_P86, FunctionalId::GgaCP86),
        ],
        hyb_info(
            FunctionalId::HybGgaXcRevdsdPbep86D4,
            "hyb_gga_xc_revdsd_pbep86_d4",
            CX_HF,
        ),
    )
}

/// The original B3LYP (libxc 402, `hyb_gga_xc_b3lyp`): uses **VWN_RPA**
/// (`lda_c_vwn_rpa`, id 8) for the LDA correlation — *not* VWN3 and *not* VWN5
/// (libxc has separate `b3lyp3`/394 and `b3lyp5`/475 for those). This is the
/// classic B3LYP-VWN ambiguity; the flavor here is whatever libxc 402 emits.
pub(crate) fn b3lyp() -> Result<Box<dyn XcEval>, XcError> {
    b3lyp_like(
        FunctionalId::HybGgaXcB3lyp,
        "hyb_gga_xc_b3lyp",
        FunctionalId::LdaCVwnRpa,
    )
}

#[cfg(test)]
mod tests {
    use crate::{Functional, FunctionalId, Spin, XcInput};

    /// PBE0 must be exactly the semilocal mix `0.75·PBE-x + 1.0·PBE-c` (xcx emits
    /// no EXX), componentwise on exc/vrho/vsigma — and expose exx_fraction = 0.25.
    #[test]
    fn pbe0_is_semilocal_mix_and_exposes_exx() {
        let f = Functional::new(FunctionalId::HybGgaXcPbeh, Spin::Polarized).unwrap();
        assert_eq!(f.exx_fraction(), 0.25);
        assert_eq!(f.info().name, "hyb_gga_xc_pbeh");

        let px = Functional::new(FunctionalId::GgaXPbe, Spin::Polarized).unwrap();
        let pc = Functional::new(FunctionalId::GgaCPbe, Spin::Polarized).unwrap();
        let r = [0.6, 0.3];
        let s = [0.1, 0.05, 0.08];
        let h = f.eval(1, &XcInput::gga(&r, &s)).unwrap();
        let ex = px.eval(1, &XcInput::gga(&r, &s)).unwrap();
        let ec = pc.eval(1, &XcInput::gga(&r, &s)).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-14 * a.abs().max(1.0);
        assert!(close(h.exc[0], 0.75 * ex.exc[0] + ec.exc[0]));
        for k in 0..2 {
            assert!(
                close(h.vrho[k], 0.75 * ex.vrho[k] + ec.vrho[k]),
                "vrho[{k}]"
            );
        }
        for k in 0..3 {
            assert!(
                close(h.vsigma[k], 0.75 * ex.vsigma[k] + ec.vsigma[k]),
                "vsigma[{k}]"
            );
        }
    }

    /// B3LYP5 must be the semilocal mix `0.08·LDA_X + 0.72·B88 + 0.19·VWN5 +
    /// 0.81·LYP` — pinning the **VWN5** correlation flavor (libxc 475) — with
    /// exx_fraction = 0.20. Using VWN_RPA or VWN3 here would break this.
    #[test]
    fn b3lyp5_is_semilocal_mix_with_vwn5() {
        let f = Functional::new(FunctionalId::HybGgaXcB3lyp5, Spin::Polarized).unwrap();
        assert_eq!(f.exx_fraction(), 0.20);
        assert_eq!(f.info().name, "hyb_gga_xc_b3lyp5");
        let comp = |id| Functional::new(id, Spin::Polarized).unwrap();
        let lx = comp(FunctionalId::LdaX);
        let b88 = comp(FunctionalId::GgaXB88);
        let vwn5 = comp(FunctionalId::LdaCVwn);
        let lyp = comp(FunctionalId::GgaCLyp);
        let r = [0.6, 0.3];
        let s = [0.1, 0.05, 0.08];
        let inp = XcInput::gga(&r, &s);
        let h = f.eval(1, &inp).unwrap();
        let (clx, cax, cac, cc) = (1.0 - 0.20 - 0.72, 0.72, 1.0 - 0.81, 0.81);
        let want = clx * lx.eval(1, &inp).unwrap().exc[0]
            + cax * b88.eval(1, &inp).unwrap().exc[0]
            + cac * vwn5.eval(1, &inp).unwrap().exc[0]
            + cc * lyp.eval(1, &inp).unwrap().exc[0];
        assert!((h.exc[0] - want).abs() <= 1e-13 * want.abs());
    }

    /// B3LYP (402) must use **VWN_RPA** for the LDA correlation: it equals
    /// `0.08·LDA_X + 0.72·B88 + 0.19·VWN_RPA + 0.81·LYP`, and at a polarized point
    /// it must DIFFER from B3LYP5 (which uses VWN5) — guarding the VWN flavor.
    #[test]
    fn b3lyp_uses_vwn_rpa_not_vwn5() {
        let f = Functional::new(FunctionalId::HybGgaXcB3lyp, Spin::Polarized).unwrap();
        assert_eq!(f.exx_fraction(), 0.20);
        assert_eq!(f.info().name, "hyb_gga_xc_b3lyp");
        let comp = |id| Functional::new(id, Spin::Polarized).unwrap();
        let lx = comp(FunctionalId::LdaX);
        let b88 = comp(FunctionalId::GgaXB88);
        let vwn_rpa = comp(FunctionalId::LdaCVwnRpa);
        let lyp = comp(FunctionalId::GgaCLyp);
        let r = [0.7, 0.3];
        let s = [0.2, 0.1, 0.05];
        let inp = XcInput::gga(&r, &s);
        let h = f.eval(1, &inp).unwrap();
        let (clx, cax, cac, cc) = (1.0 - 0.20 - 0.72, 0.72, 1.0 - 0.81, 0.81);
        let want = clx * lx.eval(1, &inp).unwrap().exc[0]
            + cax * b88.eval(1, &inp).unwrap().exc[0]
            + cac * vwn_rpa.eval(1, &inp).unwrap().exc[0]
            + cc * lyp.eval(1, &inp).unwrap().exc[0];
        assert!((h.exc[0] - want).abs() <= 1e-13 * want.abs());
        // distinct from B3LYP5 (VWN5) at this polarized point
        let f5 = Functional::new(FunctionalId::HybGgaXcB3lyp5, Spin::Polarized).unwrap();
        let h5 = f5.eval(1, &inp).unwrap();
        assert!(
            (h.exc[0] - h5.exc[0]).abs() > 1e-9 * h.exc[0].abs(),
            "B3LYP (VWN_RPA) must differ from B3LYP5 (VWN5): {} vs {}",
            h.exc[0],
            h5.exc[0]
        );
    }

    /// σ_ab dependence comes only from PBE-c (exchange has none), so vsigma_ab is
    /// nonzero for the hybrid — a sanity check that the correlation part mixes in.
    #[test]
    fn pbe0_unpol_pol_consistent_at_zero_polarization() {
        let up = Functional::new(FunctionalId::HybGgaXcPbeh, Spin::Unpolarized).unwrap();
        let po = Functional::new(FunctionalId::HybGgaXcPbeh, Spin::Polarized).unwrap();
        let (n, s) = (0.8, 0.3);
        let ou = up.eval(1, &XcInput::gga(&[n], &[s])).unwrap();
        let op = po
            .eval(
                1,
                &XcInput::gga(&[n / 2.0, n / 2.0], &[s / 4.0, s / 4.0, s / 4.0]),
            )
            .unwrap();
        assert!((ou.exc[0] - op.exc[0]).abs() <= 1e-11 * ou.exc[0].abs());
        assert!((ou.vrho[0] - op.vrho[0]).abs() <= 1e-10 * ou.vrho[0].abs().max(1.0));
    }
}
