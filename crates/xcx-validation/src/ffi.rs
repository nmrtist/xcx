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
}
