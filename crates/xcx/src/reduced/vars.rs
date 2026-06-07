// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Map from raw spin densities / gradients to libxc reduced variables, plus the
//! spin-scaling helpers shared by exchange/correlation energies.
//! Provenance: ported-from-libxc (MPL-2.0), `maple/util.mpl`.
//!
//! Spin factors are taken as the cancellation-free `opz = 1 + z = 2·n_a/n` (and
//! `omz = 1 − z = 2·n_b/n`) rather than reconstructed as `1 ± z` from `z`, which
//! would lose precision near full spin polarization (z → ±1).

use num_dual::DualNum;

use super::consts::{FOUR_CBRT2, FZETA_DENOM, LDA_X_FACTOR, RS_FACTOR, TWO_POW_M4_3};

/// Wigner–Seitz radius `rs = (3/(4π n))^(1/3) = RS_FACTOR / n^(1/3)`.
pub(crate) fn rs_from_n<N: DualNum<f64> + Copy>(n: N) -> N {
    N::from(RS_FACTOR) / n.cbrt()
}

/// Relative spin polarization `z = (n_a − n_b)/(n_a + n_b)`.
///
/// Provided for completeness with the libxc reduced-variable vocabulary; the
/// family harnesses derive ζ via the cancellation-free `opz`/`omz` helpers
/// rather than this direct difference form, so this is currently unused.
#[allow(dead_code)]
pub(crate) fn zeta<N: DualNum<f64> + Copy>(na: N, nb: N) -> N {
    (na - nb) / (na + nb)
}

/// Reduced density gradient `x = |∇n| / n^(4/3) = sqrt(sigma) / n^(4/3)`.
pub(crate) fn reduced_grad<N: DualNum<f64> + Copy>(sigma: N, n: N) -> N {
    sigma.sqrt() / n.powf(4.0 / 3.0)
}

/// Squared reduced density gradient `x² = σ / n^(8/3) = (|∇n|/n^(4/3))²`. The
/// sqrt-free companion to [`reduced_grad`]; use it for the *total* gradient,
/// which can be **exactly zero** (the σ_ab clamp drives σ_tot → 0 when the spin
/// gradients cancel). `√σ`'s derivative diverges at σ = 0 — and squaring it back
/// gives `0·∞ = NaN` under forward-AD — whereas `σ/n^(8/3)` stays smooth.
pub(crate) fn reduced_grad_sq<N: DualNum<f64> + Copy>(sigma: N, n: N) -> N {
    sigma / n.powf(8.0 / 3.0)
}

/// `(1 + z)^p` with libxc's clamp, taking the cancellation-free `opz = 1 + z`.
/// When `opz ≤ zeta_threshold`, returns `zeta_threshold^p` (keeps the value and
/// its derivative defined at full polarization). Provenance: util.mpl `opz_pow_n`.
pub(crate) fn opz_pow<N: DualNum<f64> + Copy>(opz: N, p: f64, zeta_threshold: f64) -> N {
    if opz.re() <= zeta_threshold {
        N::from(zeta_threshold.powf(p))
    } else {
        opz.powf(p)
    }
}

/// libxc per-channel density screen: `true` when the (floored) spin density is at
/// or below `dens_threshold`. Uses the spin density directly — matching libxc's
/// `my_rho[spin] ≤ dens_threshold` — so borderline decisions at full
/// polarization agree exactly (reconstructing `n_spin` from `rs`/`opz` would
/// round across the threshold). The predicate is on the real part, so forward-AD
/// follows the chosen branch. Provenance: util.mpl `screen_dens` / `n_spin`.
pub(crate) fn screen_dens<N: DualNum<f64> + Copy>(n_spin: N, dens_threshold: f64) -> bool {
    n_spin.re() <= dens_threshold
}

/// 3D LDA exchange energy per particle for one spin channel, given the
/// cancellation-free `opz = 1 + z`:
/// `LDA_X_FACTOR · opz^(4/3) · 2^(-4/3) · n^(1/3)`, with `n^(1/3) = RS_FACTOR/rs`.
/// Provenance: util.mpl `lda_x_spin`.
pub(crate) fn lda_x_spin<N: DualNum<f64> + Copy>(rs: N, opz: N, zeta_threshold: f64) -> N {
    opz_pow(opz, 4.0 / 3.0, zeta_threshold)
        * N::from(LDA_X_FACTOR * TWO_POW_M4_3)
        * (N::from(RS_FACTOR) / rs)
}

/// `(1+z)^p − 1`, libxc's exact form `expm1(p·log1p(z))`, clamped to
/// `zeta_threshold^p − 1` when `1 + z ≤ zeta_threshold`. The `−1` is built in (so
/// `f(0)=0` is exact), but `log1p(z)` is taken on `z` itself — matching libxc's
/// arithmetic, including its cancellation near `z = −1`. This is required for
/// correlation `f(ζ)` to agree with libxc bit-for-bit; exchange uses the
/// cancellation-free `opz` form above (which is what libxc's exchange does).
/// Provenance: util.mpl `opz_pow_n_m1`.
pub(crate) fn opz_pow_m1<N: DualNum<f64> + Copy>(z: N, p: f64, zeta_threshold: f64) -> N {
    if 1.0 + z.re() <= zeta_threshold {
        N::from(zeta_threshold.powf(p) - 1.0)
    } else {
        (z.ln_1p() * N::from(p)).exp_m1()
    }
}

/// PW92/VWN spin-scaling function
/// `f(z) = [(1+z)^(4/3) + (1−z)^(4/3) − 2] / (2^(4/3) − 2)`, via the `_m1`
/// primitive (so `f(0)=0` exactly) in libxc's exact form. Provenance: util.mpl
/// `f_zeta`.
pub(crate) fn f_zeta<N: DualNum<f64> + Copy>(z: N, zeta_threshold: f64) -> N {
    (opz_pow_m1(z, 4.0 / 3.0, zeta_threshold) + opz_pow_m1(-z, 4.0 / 3.0, zeta_threshold))
        / N::from(FZETA_DENOM)
}

/// `1 − z²` in libxc's factored form `(1−z)(1+z)` (util.mpl `one_minus_z_pow_n`
/// for n = 2). Takes the **raw** `z` (correlation's z-based convention, see
/// docs/api-convention.md §8, divergence A); the `(1−z)(1+z)` factorisation keeps both
/// boundaries cancellation-free. **Unclamped** — distinct from the
/// `zeta_threshold`-clamped `(1±z)²` you get from [`opz_pow`]. Used by LYP
/// correlation. Provenance: util.mpl `one_minus_z_pow_n`.
pub(crate) fn one_minus_z_pow2<N: DualNum<f64> + Copy>(z: N) -> N {
    (N::from(1.0) - z) * (N::from(1.0) + z)
}

/// `1 − z⁴` in libxc's cancellation-free factored form `(1−z)(1+z)(1+z²)`
/// (util.mpl `one_minus_z_pow_n` for n = 4). Like `f_zeta`, this takes the **raw**
/// `z` rather than the cancellation-free `opz`/`omz` — correlation must reproduce
/// libxc's z-based arithmetic to match it (see docs/api-convention.md §8,
/// divergence A). The factor vanishes as `z → ±1`, so the choice is numerically
/// immaterial there.
/// Provenance: util.mpl `one_minus_z_pow_n`.
pub(crate) fn one_minus_z_pow4<N: DualNum<f64> + Copy>(z: N) -> N {
    (N::from(1.0) - z) * (N::from(1.0) + z) * (N::from(1.0) + z * z)
}

/// PBE/PW91 spin-scaling `φ(ζ) = [(1+ζ)^(2/3) + (1−ζ)^(2/3)]/2`, built from the
/// cancellation-free `opz_pow_m1` primitive so `φ(0)=1` is exact and `ζ → ±1` is
/// clamped at `zeta_threshold` (`(1±ζ)^(2/3)` stays defined). z-based, the
/// correlation convention. Provenance: util.mpl `mphi`.
pub(crate) fn mphi<N: DualNum<f64> + Copy>(z: N, zeta_threshold: f64) -> N {
    N::from(1.0)
        + (opz_pow_m1(z, 2.0 / 3.0, zeta_threshold) + opz_pow_m1(-z, 2.0 / 3.0, zeta_threshold))
            / N::from(2.0)
}

/// Squared PBE correlation reduced gradient `t² = x_t²/((4·2^(1/3))²·φ(ζ)²·rs)`
/// (util.mpl `tt`, squared), taking the squared total reduced gradient `xt2`
/// (`reduced_grad_sq`). Working in `t²` (not `t`) keeps forward-AD finite when the
/// total gradient vanishes: PBE-C's `H` depends on `t` only through `t²`/`t⁴`, so
/// the maple's intermediate `√σ_tot` (whose derivative blows up at σ_tot = 0,
/// reachable via the σ_ab clamp) is unnecessary. Takes `φ(ζ)` precomputed (PBE-C
/// reuses it in the `A`/`H` terms). Provenance: util.mpl `tt`.
pub(crate) fn tt_sq<N: DualNum<f64> + Copy>(rs: N, xt2: N, phi: N) -> N {
    xt2 / (N::from(FOUR_CBRT2 * FOUR_CBRT2) * phi * phi * rs)
}
