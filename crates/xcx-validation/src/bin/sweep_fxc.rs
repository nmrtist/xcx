// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Map the small-σ `fxc` blast radius: sweep σ downward and compare every
//! second-derivative block (`v2rho2`, `v2rhosigma`, `v2sigma2`) of `xcx` against
//! pinned libxc, flagging the first σ at which the relative error crosses the
//! 1e-10 golden tolerance.
//!
//! This is a measurement tool (libxc-ffi only), not a test. It exists to settle
//! — by FFI, not inference — *which* GGA/hybrid functionals and *which* blocks
//! lose accuracy as σ → 0, and at what σ the crossover happens. The resulting
//! divergence record (B88 `v2sigma2` at exact σ = 0) is in
//! `docs/api-convention.md` §8.
//!
//! ```text
//! $env:XCX_LIBXC_DLL = "<env>\Library\bin\xc.dll"
//! cargo run -p xcx-validation --features libxc-ffi --bin sweep_fxc
//! ```

#[cfg(feature = "libxc-ffi")]
fn main() {
    use xcx::{Functional, Spin, XcInput};
    use xcx_validation::ffi::Libxc;
    use xcx_validation::{rel_close, ATOL, RTOL};

    let xc = Libxc::load();
    let (vmaj, vmin, vmic) = xc.version();
    eprintln!("libxc version {vmaj}.{vmin}.{vmic}");

    // σ sweep: a decade ladder down to the deep-small regime, plus exact 0.
    let sigmas: Vec<f64> = {
        let mut v: Vec<f64> = (1..=12).map(|k| 10f64.powi(-k)).collect();
        v.push(0.0);
        v
    };

    // GGA + hybrid functionals (LDA has no σ). gga_c_pbe / gga_c_lyp are included
    // to settle whether the correlation per-spin gradient is affected too.
    let functionals = [
        "gga_x_pbe",
        "gga_x_b88",
        "gga_c_pbe",
        "gga_c_lyp",
        "hyb_gga_xc_pbeh",
        "hyb_gga_xc_b3lyp5",
        "hyb_gga_xc_b3lyp",
    ];

    // Displayed relative error, with the golden ATOL as a denominator floor so
    // *structural* zeros (e.g. a pure-exchange cross-spin Hessian entry, which is
    // 0 in xcx but a ~1e-18 rounding value in libxc) are not reported as 100%
    // error — those pass the golden predicate by its absolute floor. Only genuine
    // (significant-magnitude) disagreement shows up here.
    let rel = |a: f64, b: f64| -> f64 {
        let denom = a.abs().max(b.abs());
        if denom <= ATOL {
            0.0
        } else {
            (a - b).abs() / denom
        }
    };
    let block_max = |got: &[f64], want: &[f64]| -> f64 {
        got.iter()
            .zip(want)
            .map(|(&g, &w)| rel(g, w))
            .fold(0.0, f64::max)
    };
    // Authoritative "does this block fail golden": the exact golden predicate
    // (|Δ| ≤ rtol·max + atol) applied per component. Drives the crossover flag,
    // so it agrees bit-for-bit with what the golden suite would reject.
    let block_fails = |got: &[f64], want: &[f64]| -> bool {
        got.iter()
            .zip(want)
            .any(|(&g, &w)| !rel_close(g, w, RTOL, ATOL))
    };

    // One swept configuration: a label, the spin, and a closure producing the
    // (rho, sigma) packed inputs for a given swept σ value.
    struct Config {
        label: &'static str,
        spin: Spin,
        // (rho, sigma) builder
        build: fn(f64) -> (Vec<f64>, Vec<f64>),
    }
    let configs: &[Config] = &[
        Config {
            label: "unpol n=0.3",
            spin: Spin::Unpolarized,
            build: |s| (vec![0.3], vec![s]),
        },
        Config {
            label: "unpol n=1.0",
            spin: Spin::Unpolarized,
            build: |s| (vec![1.0], vec![s]),
        },
        Config {
            label: "unpol n=10",
            spin: Spin::Unpolarized,
            build: |s| (vec![10.0], vec![s]),
        },
        // Polarized pair, σ swept on both per-spin channels (σ_ab = 0): this is
        // the configuration that drives the per-spin reduced gradient → 0.
        Config {
            label: "pol (0.6,0.4) saa=sbb=s sab=0",
            spin: Spin::Polarized,
            build: |s| (vec![0.6, 0.4], vec![s, 0.0, s]),
        },
        Config {
            label: "pol (2.0,1.0) saa=sbb=s sab=0",
            spin: Spin::Polarized,
            build: |s| (vec![2.0, 1.0], vec![s, 0.0, s]),
        },
    ];

    for name in functionals {
        let id = xc.number(name);
        assert!(id > 0, "libxc does not know `{name}`");
        println!("\n================ {name} ================");
        for cfg in configs {
            let nspin = match cfg.spin {
                Spin::Polarized => 2,
                _ => 1,
            };
            let f = Functional::by_name(name, cfg.spin).unwrap();
            println!(
                "\n-- {} [{}] --",
                cfg.label,
                if nspin == 2 { "pol" } else { "unpol" }
            );
            println!(
                "{:>10}  {:>12}  {:>12}  {:>12}   {:>14} {:>14}",
                "sigma", "v2rho2", "v2rhosigma", "v2sigma2", "xcx v2sig2[0]", "libxc v2sig2[0]"
            );
            // crossover σ per block (first σ, scanning high→low, with rel > 1e-10)
            let mut cross = [f64::NAN; 3];
            for &s in &sigmas {
                let (rho, sigma) = (cfg.build)(s);
                let np = 1;
                let out = f.eval_fxc(np, &XcInput::gga(&rho, &sigma)).unwrap();
                let (lr2, lrs, ls2) = xc.gga_fxc(id, nspin, np, &rho, &sigma);
                let e = [
                    block_max(&out.v2rho2, &lr2),
                    block_max(&out.v2rhosigma, &lrs),
                    block_max(&out.v2sigma2, &ls2),
                ];
                let fails = [
                    block_fails(&out.v2rho2, &lr2),
                    block_fails(&out.v2rhosigma, &lrs),
                    block_fails(&out.v2sigma2, &ls2),
                ];
                for b in 0..3 {
                    if cross[b].is_nan() && fails[b] {
                        cross[b] = s;
                    }
                }
                println!(
                    "{:>10.0e}  {:>12.3e}  {:>12.3e}  {:>12.3e}   {:>14.5e} {:>14.5e}",
                    s, e[0], e[1], e[2], out.v2sigma2[0], ls2[0]
                );
            }
            let fmt = |x: f64| {
                if x.is_nan() {
                    "none".to_string()
                } else {
                    format!("{x:.0e}")
                }
            };
            println!(
                "   first σ over 1e-10:  v2rho2={}  v2rhosigma={}  v2sigma2={}",
                fmt(cross[0]),
                fmt(cross[1]),
                fmt(cross[2]),
            );
        }
    }

    // ----------------------------------------------------------------------
    // meta-GGA small-σ sweep: the same crossover question for the τ-dependent
    // functionals, across ALL SIX fxc blocks (v2rho2, v2rhosigma, v2sigma2,
    // v2rhotau, v2sigmatau, v2tau2). τ is held fixed and physical (well above
    // τ_W so the FHC clamp σ ← min(σ, 8nτ) never fires across the swept σ), so
    // this isolates whether any small-σ band is (a) the TPSS-x enhancement's
    // √(½(9/25 z² + p²)) factoring [TPSS-specific, harmless to r2SCAN] or (b) a
    // HARNESS path (reduced_grad_sq / reduced_tau / mgga_alpha) that r2SCAN would
    // inherit. The σ_ab = 0 polarized configs drive both per-spin reduced
    // gradients → 0 simultaneously (the harness-path stress).
    let mgga_functionals = [
        "mgga_x_tpss",
        "mgga_c_tpss",
        "mgga_x_r2scan",
        "mgga_c_r2scan",
        "mgga_x_m06_l",
        "mgga_c_m06_l",
    ];
    // Packed (rho, sigma, tau) inputs a swept-σ config produces.
    type MggaSweepInputs = (Vec<f64>, Vec<f64>, Vec<f64>);
    struct MggaConfig {
        label: &'static str,
        spin: Spin,
        // builder: swept σ -> (rho, sigma, tau)
        build: fn(f64) -> MggaSweepInputs,
    }
    let mgga_configs: &[MggaConfig] = &[
        MggaConfig {
            label: "unpol n=0.3 τ=0.2",
            spin: Spin::Unpolarized,
            build: |s| (vec![0.3], vec![s], vec![0.2]),
        },
        MggaConfig {
            label: "unpol n=1.0 τ=0.8",
            spin: Spin::Unpolarized,
            build: |s| (vec![1.0], vec![s], vec![0.8]),
        },
        MggaConfig {
            label: "unpol n=10 τ=30",
            spin: Spin::Unpolarized,
            build: |s| (vec![10.0], vec![s], vec![30.0]),
        },
        MggaConfig {
            label: "pol (0.6,0.4) saa=sbb=s sab=0 τ=(0.5,0.3)",
            spin: Spin::Polarized,
            build: |s| (vec![0.6, 0.4], vec![s, 0.0, s], vec![0.5, 0.3]),
        },
        // The session-notes TPSS-x reference point (n=[1.0,0.7], τ=[0.8,0.6]).
        MggaConfig {
            label: "pol (1.0,0.7) saa=sbb=s sab=0 τ=(0.8,0.6)",
            spin: Spin::Polarized,
            build: |s| (vec![1.0, 0.7], vec![s, 0.0, s], vec![0.8, 0.6]),
        },
    ];
    let block_names = [
        "v2rho2",
        "v2rhosigma",
        "v2sigma2",
        "v2rhotau",
        "v2sigmatau",
        "v2tau2",
    ];
    for name in mgga_functionals {
        let id = xc.number(name);
        assert!(id > 0, "libxc does not know `{name}`");
        println!("\n================ {name} ================");
        for cfg in mgga_configs {
            let nspin = match cfg.spin {
                Spin::Polarized => 2,
                _ => 1,
            };
            let f = Functional::by_name(name, cfg.spin).unwrap();
            println!(
                "\n-- {} [{}] --",
                cfg.label,
                if nspin == 2 { "pol" } else { "unpol" }
            );
            println!(
                "{:>10}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}",
                "sigma", "v2rho2", "v2rhosig", "v2sigma2", "v2rhotau", "v2sigtau", "v2tau2"
            );
            let mut cross = [f64::NAN; 6];
            for &s in &sigmas {
                let (rho, sigma, tau) = (cfg.build)(s);
                let np = 1;
                let lapl = vec![0.0; rho.len()];
                let out = f
                    .eval_fxc(np, &XcInput::gga(&rho, &sigma).with_tau(&tau))
                    .unwrap();
                let (lr2, lrs, ls2, lrt, lst, ltt) =
                    xc.mgga_fxc(id, nspin, np, &rho, &sigma, &lapl, &tau);
                let got = [
                    &out.v2rho2[..],
                    &out.v2rhosigma[..],
                    &out.v2sigma2[..],
                    &out.v2rhotau[..],
                    &out.v2sigmatau[..],
                    &out.v2tau2[..],
                ];
                let want = [&lr2[..], &lrs[..], &ls2[..], &lrt[..], &lst[..], &ltt[..]];
                let mut e = [0.0; 6];
                for b in 0..6 {
                    e[b] = block_max(got[b], want[b]);
                    if cross[b].is_nan() && block_fails(got[b], want[b]) {
                        cross[b] = s;
                    }
                }
                println!(
                    "{:>10.0e}  {:>11.3e}  {:>11.3e}  {:>11.3e}  {:>11.3e}  {:>11.3e}  {:>11.3e}",
                    s, e[0], e[1], e[2], e[3], e[4], e[5]
                );
            }
            let fmt = |x: f64| {
                if x.is_nan() {
                    "none".to_string()
                } else {
                    format!("{x:.0e}")
                }
            };
            print!("   first σ over 1e-10: ");
            for b in 0..6 {
                print!(" {}={}", block_names[b], fmt(cross[b]));
            }
            println!();
        }
    }

    // ----------------------------------------------------------------------
    // meta-GGA correlation low-density vxc crossover: at what total density does
    // the analytic-vs-AD derivative divergence (class #1) cross 1e-10? The §2
    // anchor is a central FD of *libxc's* own energy density: whichever of
    // {xcx vsigma, libxc vsigma} the FD agrees with is the accurate side.
    let ld_funcs = ["mgga_c_r2scan", "mgga_c_tpss", "mgga_c_m06_l"];
    let lapl1 = [0.0_f64];
    for name in ld_funcs {
        let id = xc.number(name);
        println!("\n========= {name} low-density vxc =========");
        println!(
            "{:>8} {:>13} {:>13} {:>13} {:>11} {:>11} {:>11}",
            "n", "xcx vsigma", "libxc vsigma", "FD(libxc e)", "rel x-l", "rel x-fd", "rel l-fd"
        );
        let f = xcx::Functional::by_name(name, Spin::Unpolarized).unwrap();
        for k in 2..=14 {
            let n = 10f64.powi(-k);
            // physical-ish: reduced gradient s ~ 0.5 (σ = (2·X2S·s)²·n^(8/3)/4-ish);
            // just use σ scaled so x_t² is O(1), and τ ~ τ_unif (α ~ 1).
            let sigma = n.powf(8.0 / 3.0); // x_t² = 1
            let tau = 4.557_799_872_345_596 * n.powf(5.0 / 3.0); // τ_unif (α≈1)
            let out = f
                .eval(1, &XcInput::gga(&[n], &[sigma]).with_tau(&[tau]))
                .unwrap();
            let (_zk, _vr, vsig, _vl, _vt) =
                xc.mgga_exc_vxc(id, 1, 1, &[n], &[sigma], &lapl1, &[tau]);
            // central FD of libxc energy density e = n·zk wrt σ
            let h = 1e-6 * sigma;
            let ep = {
                let (zk, ..) = xc.mgga_exc_vxc(id, 1, 1, &[n], &[sigma + h], &lapl1, &[tau]);
                n * zk[0]
            };
            let em = {
                let (zk, ..) = xc.mgga_exc_vxc(id, 1, 1, &[n], &[sigma - h], &lapl1, &[tau]);
                n * zk[0]
            };
            let fd = (ep - em) / (2.0 * h);
            let relr = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
            println!(
                "{:>8.0e} {:>13.5e} {:>13.5e} {:>13.5e} {:>11.2e} {:>11.2e} {:>11.2e}",
                n,
                out.vsigma[0],
                vsig[0],
                fd,
                relr(out.vsigma[0], vsig[0]),
                relr(out.vsigma[0], fd),
                relr(vsig[0], fd),
            );
        }
    }
}

#[cfg(not(feature = "libxc-ffi"))]
fn main() {
    eprintln!(
        "sweep_fxc requires `--features libxc-ffi` and a libxc shared library \
         (set XCX_LIBXC_DLL or CONDA_PREFIX). See crates/xcx-validation/README.md."
    );
}
