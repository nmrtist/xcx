# xcx

Exchange–correlation (XC) functionals for density-functional theory (DFT) in
pure Rust, built around automatic differentiation — no C dependency.

Given a density (and, per functional, its gradient / kinetic energy density),
`xcx` returns the XC energy per particle, its first and second derivatives
(`vxc` and `fxc`), plus metadata (family, requirements, exact-exchange
fraction, range-separation, VV10, and PT2 parameters). Each functional is one
scalar energy expression; all derivatives come from forward-mode automatic
differentiation, so they are correct by construction.

`xcx` keeps [libxc](https://libxc.gitlab.io/) ids and conventions for drop-in
interoperability (golden-verified to ≤ 1e-10 where the two overlap), and goes
beyond it: a first-class **double-hybrid** family (B2PLYP, revDSD-PBEP86-D4,
PWPB95, ωB97M(2)) with structured PT2/CAM metadata that libxc does not ship.

`xcx` maps `(rho, sigma, tau[, lapl]) → energy density + derivatives + metadata
+ linear mixing` and nothing else: no grids, AO evaluation, SCF, or dispersion.

See [`docs/api-convention.md`](https://github.com/nmrtist/xcx/blob/main/docs/api-convention.md)
for the frozen data/ABI contract and the full scope fence. Licensed per file:
original xcx code under **MIT OR Apache-2.0**; code derived from libxc under
**MPL-2.0** (see `NOTICE`).

```rust,ignore
use xcx::{Functional, FunctionalId, Spin, XcInput};

let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized)?;
let rho = [0.1_f64, 0.2, 0.3];
let out = f.eval(rho.len(), &XcInput::lda(&rho))?;
// out.exc[i] = energy per particle; out.vrho[i] = ∂(n·ε)/∂n
# Ok::<(), xcx::XcError>(())
```
