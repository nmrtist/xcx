#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Generate SCF reference data for the xcx end-to-end gate (v0.1.0 DoD §6).

Runs PySCF (in WSL/Linux) as the *host*: converge a small molecule with B3LYP
forced to **libxc 402** (HYB_GGA_XC_B3LYP, VWN_RPA — the exact flavor xcx's
`hyb_gga_xc_b3lyp` implements), then dump, per integration-grid point, the
inputs xcx's public API consumes (weight, rho, sigma) plus PySCF's own semilocal
energy density, and the converged scalars (total energy, semilocal E_xc, EXX
fraction, versions).

PySCF is the host/grid generator ONLY. The XC *truth* stays on the pinned libxc
6.1.0 (via xcx-validation's FFI) on the Rust side — PySCF bundles its own libxc
(recorded here), so the Rust step quantifies any 6.1.0-vs-bundled delta rather
than trusting version-stability.

Packing matches xcx/api-convention §3 (point-major):
  RKS (unpolarized): rho=[n], sigma=[sigma]
  UKS (polarized):   rho=[n_a,n_b], sigma=[s_aa,s_ab,s_bb]

Usage (WSL):
  python3 gen_scf_ref.py <repo>/crates/xcx-validation/testdata/scf/_fullgrid
"""

import json
import sys

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import libxc

XC = "HYB_GGA_XC_B3LYP"  # -> libxc 402 (VWN_RPA); matches xcx hyb_gga_xc_b3lyp
GRID_LEVEL = 3


def versions():
    return {
        "pyscf_version": pyscf.__version__,
        "libxc_version": ".".join(str(x) for x in _libxc_version()),
        "xc": XC,
        "libxc_id": int(libxc.parse_xc(XC)[1][0][0]),
        "hybrid_coeff": float(libxc.hybrid_coeff(XC)),
    }


def _libxc_version():
    v = libxc.libxc_version()
    if isinstance(v, str):
        return tuple(int(x) for x in v.split("."))
    return tuple(v)


def grid_rho_sigma(ni, mol, ao, dm):
    """Return (density[N], grad[3,N]) for one spin channel on the grid."""
    r = ni.eval_rho(mol, ao, dm, xctype="GGA")
    return r[0], r[1:4]


def gen_closed_shell(name, atom, basis, outdir):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    e_tot = mf.kernel()
    assert mf.converged, f"{name}: SCF did not converge"
    dm = mf.make_rdm1()

    ni = mf._numint
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = ni.eval_ao(mol, coords, deriv=1)

    n, grad = grid_rho_sigma(ni, mol, ao, dm)
    sigma = (grad * grad).sum(axis=0)

    exc, vxc = libxc.eval_xc(XC, np.vstack([n, grad]), spin=0, deriv=1)[:2]
    vrho = vxc[0]
    vsigma = vxc[1]

    # SEMILOCAL E_xc only (nr_rks excsum = grid integral of n*eps, no EXX), and a
    # quadrature cross-check on the same grid. scf_summary['exc'] is the WITH-EXX
    # value for a hybrid (semilocal + 0.2*E_HFX); record both + the implied EXX.
    exc_sl = float(ni.nr_rks(mol, mf.grids, XC, dm)[1])
    exc_sl_quad = float(np.sum(weights * n * exc))
    exc_with_exx = float(mf.scf_summary["exc"])

    rec = {
        "case": name,
        "spin": "unpolarized",
        "molecule": atom,
        "basis": basis,
        "grid_level": GRID_LEVEL,
        **versions(),
        "n_grid": int(n.size),
        "e_tot": float(e_tot),
        "exc_sl_semilocal": exc_sl,
        "exc_sl_quad_pyscf": exc_sl_quad,
        "exc_with_exx": exc_with_exx,
        "exx_energy": exc_with_exx - exc_sl,
        "weights": weights.tolist(),
        # point-major: [n_0, n_1, ...]
        "rho": n.tolist(),
        "sigma": sigma.tolist(),
        "exc_pyscf": exc.tolist(),
        "vrho_pyscf": np.asarray(vrho).tolist(),
        "vsigma_pyscf": np.asarray(vsigma).tolist(),
    }
    _write(outdir, name, rec)
    return rec


def gen_open_shell(name, atom, basis, spin, outdir):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    e_tot = mf.kernel()
    assert mf.converged, f"{name}: SCF did not converge"
    dm = mf.make_rdm1()  # (2, nao, nao)

    ni = mf._numint
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = ni.eval_ao(mol, coords, deriv=1)

    na, ga = grid_rho_sigma(ni, mol, ao, dm[0])
    nb, gb = grid_rho_sigma(ni, mol, ao, dm[1])
    s_aa = (ga * ga).sum(axis=0)
    s_ab = (ga * gb).sum(axis=0)
    s_bb = (gb * gb).sum(axis=0)

    exc, vxc = libxc.eval_xc(
        XC, (np.vstack([na, ga]), np.vstack([nb, gb])), spin=1, deriv=1
    )[:2]
    vrho = np.asarray(vxc[0])  # (N, 2)
    vsigma = np.asarray(vxc[1])  # (N, 3)

    ntot = na + nb
    exc_sl = float(ni.nr_uks(mol, mf.grids, XC, dm)[1])
    exc_sl_quad = float(np.sum(weights * ntot * exc))
    exc_with_exx = float(mf.scf_summary["exc"])

    # point-major packing: rho=[na,nb,...], sigma=[saa,sab,sbb,...],
    # vrho=[va,vb,...], vsigma=[vaa,vab,vbb,...]
    rho_pm = np.empty(2 * na.size)
    rho_pm[0::2] = na
    rho_pm[1::2] = nb
    sig_pm = np.empty(3 * na.size)
    sig_pm[0::3] = s_aa
    sig_pm[1::3] = s_ab
    sig_pm[2::3] = s_bb
    vrho_pm = vrho.reshape(-1)  # already (N,2) row-major -> [va0,vb0,va1,vb1,...]
    vsig_pm = vsigma.reshape(-1)  # (N,3) -> [vaa0,vab0,vbb0,...]

    rec = {
        "case": name,
        "spin": "polarized",
        "molecule": atom,
        "basis": basis,
        "spin_2s": spin,
        "grid_level": GRID_LEVEL,
        **versions(),
        "n_grid": int(na.size),
        "e_tot": float(e_tot),
        "exc_sl_semilocal": exc_sl,
        "exc_sl_quad_pyscf": exc_sl_quad,
        "exc_with_exx": exc_with_exx,
        "exx_energy": exc_with_exx - exc_sl,
        "weights": weights.tolist(),
        "rho": rho_pm.tolist(),
        "sigma": sig_pm.tolist(),
        "exc_pyscf": exc.tolist(),
        "vrho_pyscf": vrho_pm.tolist(),
        "vsigma_pyscf": vsig_pm.tolist(),
    }
    _write(outdir, name, rec)
    return rec


def _write(outdir, name, rec):
    import os

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.fullgrid.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    print(
        f"{name}: n_grid={rec['n_grid']} E_tot={rec['e_tot']:.10f} "
        f"E_xc^sl(nr_rks)={rec['exc_sl_semilocal']:.10f} "
        f"quad={rec['exc_sl_quad_pyscf']:.10f} "
        f"|Δquad|={abs(rec['exc_sl_semilocal'] - rec['exc_sl_quad_pyscf']):.2e} "
        f"| EXX={rec['exx_energy']:.10f} -> {path}"
    )


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    print("versions:", versions())
    # closed-shell: exercises the unpolarized path (single sigma)
    gen_closed_shell(
        "h2o_b3lyp",
        "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692",
        "cc-pvdz",
        outdir,
    )
    # open-shell doublet: exercises polarized sigma_aa/ab/bb packing + both vrho
    gen_open_shell(
        "oh_b3lyp",
        "O 0.0000 0.0000 0.0000; H 0.0000 0.0000 0.9697",
        "cc-pvdz",
        1,
        outdir,
    )


if __name__ == "__main__":
    main()
