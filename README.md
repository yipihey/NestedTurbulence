# NestedTurbulence

**Adaptively subcycled nested-grid ideal MHD in Julia on KernelAbstractions, with constrained
transport or GLM divergence cleaning.**

Report and laboratory: **https://yipihey.github.io/NestedTurbulence/**

The numerical core is the Julia package `NestedMHD` in [`core/`](core/):

- the physics of [GLMMHDTurb](https://github.com/yipihey/GLMMHDTurb): PLM (MonCen) + 3-D MUSCL-Hancock
  + HLL/HLLD/LLF, `@fastmath` branchless limiter, compile-time Riemann selection, generic `Float32`/`Float64`;
- two ∇·B treatments per hierarchy: Dedner GLM (`mode=:glm`), or constrained transport with
  Gardiner–Stone upwinded edge EMFs (`mode=:ct`). The CT fix is from
  [Solaceonx/GLMMHDTurb@sweep-fixes](https://github.com/Solaceonx/GLMMHDTurb/tree/sweep-fixes);
- static ratio-2 nested patches advanced in Berger–Oliger order with **per-level CFL time steps**
  (coarse levels take far fewer steps than their children), **second-order time-interpolated ghost
  zones**, Berger–Colella refluxing, and CT EMF correction plus face restriction, so ∇·B stays at
  rounding across levels.

```julia
using Pkg; Pkg.activate("core"); Pkg.instantiate()
using NestedMHD
H = nested_central(32, 3; n=32, mode=:ct, T=Float32)        # backend=CUDABackend() for a GPU (untested)
init!(H, (x,y,z) -> (ρ, vx, vy, vz, p, Bx, By, Bz); vecpot = (x,y,z) -> (Ax, Ay, Az))
evolve!(H, 0.5); level_summary(H); totals(H); max_divb(H)
```

## Layout (the [quadrant](https://github.com/yipihey/quadrant) report format)

```
core/          Julia package NestedMHD: src/, test/runtests.jl, bench/run_experiments.jl
chapters/      the book: methods, implementation, validation, cost, a nested Orszag–Tang run
app/           the laboratory (plain ES modules) and its datasets app/data/*.json
_extensions/   the quadrant shortcodes, extended to resolve Julia definitions and @testsets
scripts/       build info, brand → CSS, provenance link check (Julia-aware), local server
```

```sh
make test     # Julia validation suite
make data     # regenerate app/data/*.json (CPU; NAMES="ghost convergence" for a subset)
make check    # provenance links resolve
make site     # quarto render → _site/
```

The publish workflow tests the core, checks the links, renders the book and deploys it to GitHub
Pages on every push to `main`.

## Status

Validated on CPU only (see the Status appendix of the report). No GPU backend has been compiled or
run yet. Refinement is static, with one child per level.

MIT License.
