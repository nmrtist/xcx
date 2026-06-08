# xcx

Exchange–correlation (XC) functionals for density-functional theory (DFT) — a
pure-Rust, libxc-compatible reimplementation with no C dependency.

Given a density (and, per functional, its gradient / kinetic energy density),
`xcx` returns the XC energy per particle, its first and second derivatives
(`vxc` and `fxc`), plus metadata (family, requirements, exact-exchange fraction,
range-separation and VV10 parameters). Each functional is one scalar energy
expression; derivatives come from forward-mode automatic differentiation, so
they are correct by construction.

`xcx` maps `(rho, sigma, tau[, lapl]) → energy density + derivatives + metadata
+ linear mixing` and nothing else: no grids, AO evaluation, SCF, or dispersion.

See [`docs/api-convention.md`](https://github.com/nmrtist/xcx/blob/main/docs/api-convention.md)
for the frozen data/ABI contract and the full scope fence. Licensed under
**MPL-2.0** (matching upstream libxc).

```rust,ignore
use xcx::{Functional, FunctionalId, Spin, XcInput};

let f = Functional::new(FunctionalId::LdaX, Spin::Unpolarized)?;
let rho = [0.1_f64, 0.2, 0.3];
let out = f.eval(rho.len(), &XcInput::lda(&rho))?;
// out.exc[i] = energy per particle; out.vrho[i] = ∂(n·ε)/∂n
# Ok::<(), xcx::XcError>(())
```
