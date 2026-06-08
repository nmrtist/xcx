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

use super::consts::{FOUR_CBRT2, FZETA_DENOM, K_FACTOR_C, LDA_X_FACTOR, RS_FACTOR, TWO_POW_M4_3};

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

/// Squared reduced density gradient `x² = σ / n^(8/3) = (|∇n|/n^(4/3))²`,
/// computed **directly from σ and n — never via `√σ`**. This is the only reduced
/// gradient the GGA harness seeds into a differentiated energy, for **both** the
/// total gradient `x_t²` and the per-spin gradients `x_{s0}²`/`x_{s1}²`
/// (`σ_aa/n_a^(8/3)`, `σ_bb/n_b^(8/3)`).
///
/// Provenance / rationale (sqrt-free): `√σ`'s derivatives diverge as `σ → 0`
/// (`d√σ ∝ σ^(-1/2)`, `d²√σ ∝ σ^(-3/2)`). For the **total** gradient the σ_ab
/// clamp can drive `σ_tot` to **exactly 0**, where `d√σ = ∞` and squaring back
/// gives `0·∞ = NaN` in `vxc`. For the **per-spin** gradients σ is floored `> 0`,
/// so `vxc` is finite — but the *second* derivative still loses the finite-limit
/// cancellation in f64 as σ approaches the floor (`fxc`'s `v2sigma2` error grows
/// `~ ε·σ^(-3/2)`, reaching ~1e21 at the floor; the crossover past the 1e-10
/// golden tol is `σ ≲ 1e-6`). Working in the squared form `σ/n^(8/3)` (linear in
/// σ, smooth at 0) avoids both: enhancements consume `x²` directly, reintroducing
/// `√` only where a genuine magnitude is needed and only far from 0 (B88's
/// `x·asinh x`, switched to a power series in `x²` near 0). Provenance:
/// ported-from-libxc (MPL-2.0), `maple/util.mpl` (the reduced gradient); the
/// sqrt-free organization is xcx's, for AD-safety (see `docs/api-convention.md` §8).
pub(crate) fn reduced_grad_sq<N: DualNum<f64> + Copy>(sigma: N, n: N) -> N {
    sigma / n.powf(8.0 / 3.0)
}

/// Per-spin dimensionless kinetic-energy density `t_σ = τ_σ / n_σ^(5/3)`,
/// computed **directly from τ and n — never via any √**. This is the meta-GGA
/// analogue of [`reduced_grad_sq`]: τ enters the differentiated path linearly,
/// so forward-AD's first and second derivatives stay finite as τ → 0 (τ is
/// floored `> 0` by the harness, but keeping it sqrt-free avoids reintroducing the
/// `σ → 0`-style cancellation at second order). Provenance: ported-from-libxc
/// (MPL-2.0), `maple/util.mpl` (`t_total` per-spin term `τ_σ/n_σ^(5/3)`).
pub(crate) fn reduced_tau<N: DualNum<f64> + Copy>(tau: N, n: N) -> N {
    tau / n.powf(5.0 / 3.0)
}

/// The dimensionless meta-GGA iso-orbital indicator
/// `α = (τ_σ − τ_W,σ)/τ_unif,σ = (t_σ − x_σ²/8)/K_FACTOR_C`, taking the **squared**
/// reduced gradient `x_σ²` (sqrt-free) and the reduced kinetic-energy density
/// `t_σ` ([`reduced_tau`]). This is libxc's `tpss_alpha` (util.mpl `K_FACTOR_C`),
/// the central switch variable of the SCAN/TPSS family.
///
/// **AD-hazard (CLAUDE.md §3, the τ-ratio class).** `α` has 0/0 structure as the
/// density → uniform: at `α ≈ 0` (`τ → τ_W`, the single-orbital / von Weizsäcker
/// edge) and at `σ → 0` (`τ_W → 0`); the SCAN-family switch functions `f(α)` have
/// near-singular derivatives at `α ≈ 1`. The form here is cancellation-free in
/// the *energy* — `t_σ − x_σ²/8` is a plain difference of two finite reduced
/// variables — and consumers must keep `f(α)` AD-safe across the `α ≈ 1` seam
/// (factor out, don't subtract close-to-equal terms). Densify golden+fuzz around
/// `α = 0`, `α = 1`, `σ → 0`, and the `τ = τ_W` edge.
#[allow(dead_code)]
pub(crate) fn mgga_alpha<N: DualNum<f64> + Copy>(t: N, xs_sq: N) -> N {
    (t - xs_sq / N::from(8.0)) / N::from(K_FACTOR_C)
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

/// `x^(5/3)` matching libxc's C `pow`: `0` at `x ≤ 0` (where `pow(0, 5/3) = 0`
/// and the symbolic derivative `(5/3)·pow(0, 2/3) = 0`), else the real `powf`. The
/// guard is essential at full polarization: when the floored minority density is
/// lost in the f64 sum `n_a + n_b`, a base of exactly `0` reaches `^(5/3)`, where
/// num-dual's `powf(0, 5/3)` would yield `NaN` whereas the true `0·∞ → 0` weight
/// (and libxc's `pow`) give `0`. Shared by [`t_total`] (the meta-GGA spin combiner,
/// used by TPSS-c and r2SCAN-c). Provenance: util.mpl `t_total` (`pow` semantics).
pub(crate) fn pow_5_3<N: DualNum<f64> + Copy>(x: N) -> N {
    if x.re() <= 0.0 {
        N::from(0.0)
    } else {
        x.powf(5.0 / 3.0)
    }
}

/// libxc `t_total(z, a, b) = a·((1+z)/2)^(5/3) + b·((1−z)/2)^(5/3)`: the spin
/// combiner that turns per-spin reduced quantities into the total. The bases
/// `(1±z)/2 = n_{a,b}/n ∈ [0, 1]`; the `^(5/3)` is taken via [`pow_5_3`] so the
/// `z = ±1` (single-channel) edge is finite. With `a = b = 1` it returns the
/// uniform-gas KE spin factor `((1+z)/2)^(5/3) + ((1−z)/2)^(5/3)`. Shared by the
/// meta-GGA correlations (TPSS-c, r2SCAN-c). Provenance: util.mpl `t_total`.
pub(crate) fn t_total<N: DualNum<f64> + Copy>(z: N, a: N, b: N) -> N {
    let wa = (N::from(1.0) + z) / N::from(2.0);
    let wb = (N::from(1.0) - z) / N::from(2.0);
    a * pow_5_3(wa) + b * pow_5_3(wb)
}

/// `1 − z^12` in libxc's cancellation-free factored form
/// `(1−z)(1+z)(1 + z² + z⁴ + z⁶ + z⁸ + z^10)` (util.mpl `one_minus_z_pow_n` for
/// the even case `n = 12`). Takes the **raw** `z` (correlation convention); the
/// `(1−z)(1+z)` factor keeps both `z = ±1` boundaries cancellation-free, and the
/// factor vanishes there. Used by r2SCAN-c's `scan_Gc`. Provenance: util.mpl
/// `one_minus_z_pow_n`.
pub(crate) fn one_minus_z_pow12<N: DualNum<f64> + Copy>(z: N) -> N {
    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;
    let z8 = z4 * z4;
    let z10 = z8 * z2;
    (N::from(1.0) - z) * (N::from(1.0) + z) * (N::from(1.0) + z2 + z4 + z6 + z8 + z10)
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
