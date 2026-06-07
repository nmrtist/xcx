// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Physical / mathematical constants for the reduced-variable layer.
//! Provenance: ported-from-libxc (MPL-2.0), values from `maple/util.mpl`.
//!
//! Each literal is asserted against its exact closed form in the unit test below,
//! so the constants are both fast (compile-time) and provably correct.

/// `(3/(4π))^(1/3)`: Wigner–Seitz radius prefactor, `rs = RS_FACTOR / n^(1/3)`.
pub(crate) const RS_FACTOR: f64 = 0.620_350_490_899_400_1;

/// `X_FACTOR_C = (3/8)(3/π)^(1/3) 4^(2/3)`. 3D LDA exchange uses `−X_FACTOR_C`.
pub(crate) const X_FACTOR_C: f64 = 0.930_525_736_349_1;

/// 3D LDA exchange energy prefactor, `−X_FACTOR_C`.
pub(crate) const LDA_X_FACTOR: f64 = -X_FACTOR_C;

/// `2^(−4/3)`, the per-spin-channel scaling factor in `lda_x_spin`.
pub(crate) const TWO_POW_M4_3: f64 = 0.396_850_262_992_049_84;

/// `2^(4/3) − 2`, the denominator of the spin-scaling function `f(ζ)`.
pub(crate) const FZETA_DENOM: f64 = 0.519_842_099_789_746_4;

/// `f''(0) = 4/(9·(2^(1/3) − 1))`, the VWN spin-stiffness denominator. This is
/// the **exact** value (≈1.709920934…); note it is distinct from PW92's rounded
/// `1.709921` literal (`lda_c_pw::FZ20`) — the two functionals differ here.
pub(crate) const FPP_VWN: f64 = 1.709_920_934_161_365_3;

/// `X2S = 1/(2·(6π²)^(1/3))`: converts the reduced gradient `x = |∇n|/n^(4/3)`
/// to the PBE/B88 dimensionless gradient `s = X2S·x` (util.mpl `X2S`).
pub(crate) const X2S: f64 = 0.128_278_243_853_042_2;

/// `4·2^(1/3)`: the denominator prefactor of the PBE correlation reduced gradient
/// `t = x_t/(4·2^(1/3)·φ(ζ)·√rs)` (util.mpl `tt`).
pub(crate) const FOUR_CBRT2: f64 = 5.039_684_199_579_493;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-15 * b.abs().max(1.0)
    }

    #[test]
    fn constants_match_exact() {
        assert!(close(RS_FACTOR, (3.0 / (4.0 * PI)).cbrt()), "RS_FACTOR");
        let xfc = 3.0 / 8.0 * (3.0 / PI).powf(1.0 / 3.0) * 4.0_f64.powf(2.0 / 3.0);
        assert!(close(X_FACTOR_C, xfc), "X_FACTOR_C {X_FACTOR_C} vs {xfc}");
        assert!(
            close(TWO_POW_M4_3, 2.0_f64.powf(-4.0 / 3.0)),
            "TWO_POW_M4_3"
        );
        assert!(
            close(FZETA_DENOM, 2.0_f64.powf(4.0 / 3.0) - 2.0),
            "FZETA_DENOM"
        );
        assert!(
            close(FPP_VWN, 4.0 / (9.0 * (2.0_f64.cbrt() - 1.0))),
            "FPP_VWN"
        );
        assert!(
            close(X2S, 1.0 / (2.0 * (6.0 * PI * PI).cbrt())),
            "X2S {X2S} vs {}",
            1.0 / (2.0 * (6.0 * PI * PI).cbrt())
        );
        assert!(
            close(FOUR_CBRT2, 4.0 * 2.0_f64.cbrt()),
            "FOUR_CBRT2 {FOUR_CBRT2} vs {}",
            4.0 * 2.0_f64.cbrt()
        );
    }
}
