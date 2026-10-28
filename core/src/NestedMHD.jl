"""
    NestedMHD

Nested-grid, adaptively subcycled ideal MHD on KernelAbstractions (CPU / CUDA / Metal / ROCm),
built from the GLMMHDTurb physics. Two ∇·B treatments selected per hierarchy:

  * `mode = :glm` — Dedner GLM cleaning, 9 cell-centred variables (GLMMHDTurb's default scheme);
  * `mode = :ct`  — constrained transport with face-staggered B and Gardiner & Stone (2005)
                    upwinded edge EMFs (the CT fix from Solaceonx/GLMMHDTurb@sweep-fixes).

Both use PLM (MonCen) + 3-D MUSCL-Hancock + HLL / HLLD / LLF, single-stage unsplit.

Nesting: static ratio-2 patches, Berger-Oliger order (parent first), per-level CFL steps (coarse
levels take far fewer steps than their children), second-order time-interpolated ghost zones,
Berger-Colella refluxing, CT EMF correction + face restriction (∇·B at rounding across levels),
and covered-region skipping (a parent does no work deep inside its child's footprint).
"""
module NestedMHD

using KernelAbstractions
const KA = KernelAbstractions
using Printf

export Hierarchy, nested_central, init!, evolve!, totals, max_divb, level_summary

include("physics.jl")
include("kernels.jl")
include("interlevel.jl")
include("hierarchy.jl")
include("setup.jl")

end # module
