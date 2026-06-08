// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Runtime FFI to an external conda-forge libxc, via `libloading`. Gated behind
//! the `libxc-ffi` feature; used only to (re)generate golden snapshots.
//!
//! Only a handful of stable libxc C entry points are declared by hand, so no
//! bindgen / libclang is required. No libxc source is vendored or published.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;

use libloading::{Library, Symbol};

type FnAlloc = unsafe extern "C" fn() -> *mut c_void;
type FnInit = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
type FnEnd = unsafe extern "C" fn(*mut c_void);
type FnFree = unsafe extern "C" fn(*mut c_void);
type FnLda = unsafe extern "C" fn(*const c_void, usize, *const f64, *mut f64, *mut f64);
type FnGga = unsafe extern "C" fn(
    *const c_void,
    usize,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
);
type FnNumber = unsafe extern "C" fn(*const c_char) -> c_int;
type FnVersion = unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int);
// mGGA entry points (lapl + tau in, lapl + tau derivatives out):
//   xc_mgga_exc_vxc(p, np, rho, sigma, lapl, tau, zk, vrho, vsigma, vlapl, vtau)
type FnMgga = unsafe extern "C" fn(
    *const c_void,
    usize,
    *const f64,
    *const f64,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
);
//   xc_mgga_fxc(p, np, rho, sigma, lapl, tau, v2rho2, v2rhosigma, v2rholapl,
//               v2rhotau, v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau,
//               v2tau2)
type FnMggaFxc = unsafe extern "C" fn(
    *const c_void,
    usize,
    *const f64,
    *const f64,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
);
// fxc-only entry points: xc_lda_fxc(p, np, rho, v2rho2),
// xc_gga_fxc(p, np, rho, sigma, v2rho2, v2rhosigma, v2sigma2).
type FnLdaFxc = unsafe extern "C" fn(*const c_void, usize, *const f64, *mut f64);
type FnGgaFxc = unsafe extern "C" fn(
    *const c_void,
    usize,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
);

/// A loaded libxc shared library.
pub struct Libxc {
    lib: Library,
}

/// Resolve the libxc shared library: `XCX_LIBXC_DLL` if set, else from
/// `CONDA_PREFIX` (Windows `Library\bin\xc.dll`, or Unix `lib/libxc.{so,dylib}`).
fn dll_path() -> PathBuf {
    if let Ok(p) = std::env::var("XCX_LIBXC_DLL") {
        return PathBuf::from(p);
    }
    let prefix = std::env::var("CONDA_PREFIX")
        .expect("set XCX_LIBXC_DLL to the libxc shared library, or CONDA_PREFIX to a conda env");
    let win = PathBuf::from(&prefix)
        .join("Library")
        .join("bin")
        .join("xc.dll");
    if win.exists() {
        return win;
    }
    for name in ["libxc.so", "libxc.dylib"] {
        let p = PathBuf::from(&prefix).join("lib").join(name);
        if p.exists() {
            return p;
        }
    }
    win
}

impl Libxc {
    /// Load libxc from the resolved path (panics with a clear message on failure).
    pub fn load() -> Self {
        let path = dll_path();
        let lib = unsafe { Library::new(&path) }
            .unwrap_or_else(|e| panic!("failed to load libxc at {}: {e}", path.display()));
        Self { lib }
    }

    fn sym<T>(&self, name: &[u8]) -> Symbol<'_, T> {
        unsafe { self.lib.get(name) }.unwrap_or_else(|e| {
            panic!(
                "missing libxc symbol {}: {e}",
                String::from_utf8_lossy(name)
            )
        })
    }

    /// libxc (major, minor, micro) version.
    pub fn version(&self) -> (i32, i32, i32) {
        let f: Symbol<FnVersion> = self.sym(b"xc_version\0");
        let (mut a, mut b, mut c) = (0, 0, 0);
        unsafe { f(&mut a, &mut b, &mut c) };
        (a, b, c)
    }

    /// Numeric id for a functional name, or a negative value if unknown.
    pub fn number(&self, name: &str) -> i32 {
        let c = CString::new(name).unwrap();
        let f: Symbol<FnNumber> = self.sym(b"xc_functional_get_number\0");
        unsafe { f(c.as_ptr()) }
    }

    /// Evaluate an LDA functional, returning `(zk, vrho)`.
    pub fn lda_exc_vxc(&self, id: i32, nspin: i32, np: usize, rho: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnLda> = self.sym(b"xc_lda_exc_vxc\0");
        let ns = nspin as usize;
        assert_eq!(rho.len(), np * ns, "rho length");
        let mut zk = vec![0.0; np];
        let mut vrho = vec![0.0; np * ns];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(p, np, rho.as_ptr(), zk.as_mut_ptr(), vrho.as_mut_ptr());
            end(p);
            free(p);
        }
        (zk, vrho)
    }

    /// Evaluate a GGA functional, returning `(zk, vrho, vsigma)`.
    pub fn gga_exc_vxc(
        &self,
        id: i32,
        nspin: i32,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnGga> = self.sym(b"xc_gga_exc_vxc\0");
        let ns = nspin as usize;
        let nsig = 2 * ns - 1;
        assert_eq!(rho.len(), np * ns, "rho length");
        assert_eq!(sigma.len(), np * nsig, "sigma length");
        let mut zk = vec![0.0; np];
        let mut vrho = vec![0.0; np * ns];
        let mut vsigma = vec![0.0; np * nsig];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(
                p,
                np,
                rho.as_ptr(),
                sigma.as_ptr(),
                zk.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
            );
            end(p);
            free(p);
        }
        (zk, vrho, vsigma)
    }

    /// Evaluate an LDA functional's second derivative, returning `v2rho2`
    /// (length `np` unpolarized, `3*np` polarized — `[aa, ab, bb]`).
    pub fn lda_fxc(&self, id: i32, nspin: i32, np: usize, rho: &[f64]) -> Vec<f64> {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnLdaFxc> = self.sym(b"xc_lda_fxc\0");
        let ns = nspin as usize;
        let n2 = ns * (ns + 1) / 2; // 1 (unpol) or 3 (pol)
        assert_eq!(rho.len(), np * ns, "rho length");
        let mut v2rho2 = vec![0.0; np * n2];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(p, np, rho.as_ptr(), v2rho2.as_mut_ptr());
            end(p);
            free(p);
        }
        v2rho2
    }

    /// Evaluate a GGA functional's second derivatives, returning
    /// `(v2rho2, v2rhosigma, v2sigma2)`. Polarized lengths are `3*np`, `6*np`,
    /// `6*np`; unpolarized all `np` — libxc's xc.h packing.
    pub fn gga_fxc(
        &self,
        id: i32,
        nspin: i32,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnGgaFxc> = self.sym(b"xc_gga_fxc\0");
        let ns = nspin as usize;
        let nsig = 2 * ns - 1;
        let n_rr = ns * (ns + 1) / 2; // v2rho2:     1 or 3
        let n_rs = ns * nsig; // v2rhosigma: 1 or 6
        let n_ss = nsig * (nsig + 1) / 2; // v2sigma2:   1 or 6
        assert_eq!(rho.len(), np * ns, "rho length");
        assert_eq!(sigma.len(), np * nsig, "sigma length");
        let mut v2rho2 = vec![0.0; np * n_rr];
        let mut v2rhosigma = vec![0.0; np * n_rs];
        let mut v2sigma2 = vec![0.0; np * n_ss];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(
                p,
                np,
                rho.as_ptr(),
                sigma.as_ptr(),
                v2rho2.as_mut_ptr(),
                v2rhosigma.as_mut_ptr(),
                v2sigma2.as_mut_ptr(),
            );
            end(p);
            free(p);
        }
        (v2rho2, v2rhosigma, v2sigma2)
    }

    /// Evaluate a meta-GGA functional's energy + first derivatives, returning
    /// `(zk, vrho, vsigma, vlapl, vtau)`. Polarized lengths: vrho/vlapl/vtau `2*np`,
    /// vsigma `3*np`. A zeroed `lapl` array is supplied even for `needs_lapl =
    /// false` functionals (libxc still dereferences the pointer).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn mgga_exc_vxc(
        &self,
        id: i32,
        nspin: i32,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        lapl: &[f64],
        tau: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnMgga> = self.sym(b"xc_mgga_exc_vxc\0");
        let ns = nspin as usize;
        let nsig = 2 * ns - 1;
        assert_eq!(rho.len(), np * ns, "rho length");
        assert_eq!(sigma.len(), np * nsig, "sigma length");
        assert_eq!(lapl.len(), np * ns, "lapl length");
        assert_eq!(tau.len(), np * ns, "tau length");
        let mut zk = vec![0.0; np];
        let mut vrho = vec![0.0; np * ns];
        let mut vsigma = vec![0.0; np * nsig];
        let mut vlapl = vec![0.0; np * ns];
        let mut vtau = vec![0.0; np * ns];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(
                p,
                np,
                rho.as_ptr(),
                sigma.as_ptr(),
                lapl.as_ptr(),
                tau.as_ptr(),
                zk.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
                vlapl.as_mut_ptr(),
                vtau.as_mut_ptr(),
            );
            end(p);
            free(p);
        }
        (zk, vrho, vsigma, vlapl, vtau)
    }

    /// Evaluate a meta-GGA functional's second derivatives, returning the six
    /// non-Laplacian blocks `(v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau,
    /// v2tau2)` — the ones xcx produces for `needs_lapl = false` functionals.
    /// libxc's xc.h packing (polarized lengths): v2rho2 `3*np`, v2rhosigma `6*np`,
    /// v2sigma2 `6*np`, v2rhotau `4*np`, v2sigmatau `6*np`, v2tau2 `3*np`. The
    /// Laplacian output blocks are allocated (libxc requires non-NULL pointers) but
    /// discarded.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn mgga_fxc(
        &self,
        id: i32,
        nspin: i32,
        np: usize,
        rho: &[f64],
        sigma: &[f64],
        lapl: &[f64],
        tau: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let alloc: Symbol<FnAlloc> = self.sym(b"xc_func_alloc\0");
        let init: Symbol<FnInit> = self.sym(b"xc_func_init\0");
        let end: Symbol<FnEnd> = self.sym(b"xc_func_end\0");
        let free: Symbol<FnFree> = self.sym(b"xc_func_free\0");
        let eval: Symbol<FnMggaFxc> = self.sym(b"xc_mgga_fxc\0");
        let ns = nspin as usize;
        let nsig = 2 * ns - 1;
        let n_rr = ns * (ns + 1) / 2; // v2rho2:     1 or 3
        let n_rs = ns * nsig; // v2rhosigma: 1 or 6
        let n_rl = ns * ns; // v2rholapl:  1 or 4
        let n_rt = ns * ns; // v2rhotau:   1 or 4
        let n_ss = nsig * (nsig + 1) / 2; // v2sigma2:   1 or 6
        let n_sl = nsig * ns; // v2sigmalapl:1 or 6
        let n_st = nsig * ns; // v2sigmatau: 1 or 6
        let n_ll = ns * (ns + 1) / 2; // v2lapl2:    1 or 3
        let n_lt = ns * ns; // v2lapltau:  1 or 4
        let n_tt = ns * (ns + 1) / 2; // v2tau2:     1 or 3
        assert_eq!(rho.len(), np * ns, "rho length");
        assert_eq!(sigma.len(), np * nsig, "sigma length");
        assert_eq!(lapl.len(), np * ns, "lapl length");
        assert_eq!(tau.len(), np * ns, "tau length");
        let mut v2rho2 = vec![0.0; np * n_rr];
        let mut v2rhosigma = vec![0.0; np * n_rs];
        let mut v2rholapl = vec![0.0; np * n_rl];
        let mut v2rhotau = vec![0.0; np * n_rt];
        let mut v2sigma2 = vec![0.0; np * n_ss];
        let mut v2sigmalapl = vec![0.0; np * n_sl];
        let mut v2sigmatau = vec![0.0; np * n_st];
        let mut v2lapl2 = vec![0.0; np * n_ll];
        let mut v2lapltau = vec![0.0; np * n_lt];
        let mut v2tau2 = vec![0.0; np * n_tt];
        unsafe {
            let p = alloc();
            assert!(!p.is_null(), "xc_func_alloc returned null");
            assert_eq!(init(p, id, nspin), 0, "xc_func_init({id},{nspin}) failed");
            eval(
                p,
                np,
                rho.as_ptr(),
                sigma.as_ptr(),
                lapl.as_ptr(),
                tau.as_ptr(),
                v2rho2.as_mut_ptr(),
                v2rhosigma.as_mut_ptr(),
                v2rholapl.as_mut_ptr(),
                v2rhotau.as_mut_ptr(),
                v2sigma2.as_mut_ptr(),
                v2sigmalapl.as_mut_ptr(),
                v2sigmatau.as_mut_ptr(),
                v2lapl2.as_mut_ptr(),
                v2lapltau.as_mut_ptr(),
                v2tau2.as_mut_ptr(),
            );
            end(p);
            free(p);
        }
        (v2rho2, v2rhosigma, v2sigma2, v2rhotau, v2sigmatau, v2tau2)
    }
}
