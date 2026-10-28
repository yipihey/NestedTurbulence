# kernels.jl — per-patch KernelAbstractions kernels (portable: CPU / CUDA / Metal / ROCm).
#
# Index conventions (patch-local, 1-based, g ghost cells per side, interior g+1 .. g+n):
#   U[i,j,k,v]          cell (i,j,k). GLM: v = 1:9 conserved. CT: v = 1:5 conserved, 6:8 cell-centred B
#                       (derived from the faces by `bcc_kernel!`).
#   bx[i,j,k]           Bx on the LOWER x-face of cell (i,j,k); size (nx+2g+1, ny+2g, nz+2g). by, bz likewise.
#   F[i,j,k,c,d]        flux component c through the LOWER d-face of cell (i,j,k).
#   ex[i,j,k]           Ex on the x-edge (i, j-½, k-½); ey on (i-½, j, k-½); ez on (i-½, j-½, k).
#   rec[i,j,k,:]        1:9 MUSCL-Hancock half-step primitive, 10:18/19:27/28:36 half-slopes x/y/z,
#                       37:39 (CT only) cell-centred reference EMF for the GS05 upwinding.
#
# Covered-region skipping (the nested-cost minimiser): a patch with a child carries the local box
# [clo, chi] of cells its child covers. Those cells are overwritten by restriction at the end of
# the parent step, so the parent does not update cells deeper than MU inside the box, and does not
# compute fluxes / EMFs / reconstructions deeper than the margins that the remaining updates need:
#   update cells & faces : not deep(MU)        EMF  : not deep(MU+1)
#   flux                 : not deep(MU+2)      recon: not deep(MU+3)
# (shifting a cell by one in one direction increases the depth by at most one).
# MU = 2 keeps the covered rim (2 coarse cells) evolving on the coarse grid, so the child's
# time-interpolated ghosts — which use coarse cells a-1..b+1 and their slope neighbours — see a
# coarse field advanced to t^{n+1}, not a stale restricted value.

const MU = 2
const MEMF = MU + 1
const MFLX = MU + 2
const MREC = MU + 3

@inline function deep(i::Int, j::Int, k::Int, clo::NTuple{3,Int}, chi::NTuple{3,Int}, m::Int)
    (clo[1]+m <= i <= chi[1]-m) & (clo[2]+m <= j <= chi[2]-m) & (clo[3]+m <= k <= chi[3]-m)
end

@inline loadq(U, i, j, k, ::Val{9}) = ntuple(v -> @inbounds(U[i,j,k,v]), 9)
@inline loadq(U, i, j, k, ::Val{8}) = @inbounds (U[i,j,k,1], U[i,j,k,2], U[i,j,k,3], U[i,j,k,4],
                                                 U[i,j,k,5], U[i,j,k,6], U[i,j,k,7], U[i,j,k,8], zero(eltype(U)))
@inline rget(rec, i, j, k, off) = ntuple(v -> @inbounds(rec[i,j,k,off+v]), 9)
@inline iflo(f::NTuple{9,T}, m::NTuple{9,T}, smallr::T, pfl::T) where {T} = (f[1] < smallr || f[5] < pfl) ? m : f

# ---------------------------------------------------------------------------------------------
# Ghost filling
# ---------------------------------------------------------------------------------------------

# periodic root: every ghost cell copies its wrapped interior cell (all components)
@kernel function periodic_cells_kernel!(U, g::Int, n::NTuple{3,Int}, nvs::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        gi = (i <= g) | (i > g+n[1]); gj = (j <= g) | (j > g+n[2]); gk = (k <= g) | (k > g+n[3])
        if gi | gj | gk
            si = g + mod(i-g-1, n[1]) + 1; sj = g + mod(j-g-1, n[2]) + 1; sk = g + mod(k-g-1, n[3]) + 1
            for v in 1:nvs
                U[i,j,k,v] = U[si,sj,sk,v]
            end
        end
    end
end

# periodic root faces: owned faces are g+1 .. g+n along the normal (face g+n+1 ≡ face g+1)
@kernel function periodic_faces_kernel!(b, g::Int, n::NTuple{3,Int}, d::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        oi = (g+1 <= i <= g+n[1]); oj = (g+1 <= j <= g+n[2]); ok = (g+1 <= k <= g+n[3])
        if !(oi & oj & ok)
            si = g + mod(i-g-1, n[1]) + 1; sj = g + mod(j-g-1, n[2]) + 1; sk = g + mod(k-g-1, n[3]) + 1
            b[i,j,k] = b[si,sj,sk]
        end
    end
end

# cell-centred B from the faces (CT), whole patch
@kernel function bcc_kernel!(U, @Const(bx), @Const(by), @Const(bz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        h = eltype(U)(0.5)
        U[i,j,k,6] = h*(bx[i,j,k] + bx[i+1,j,k])
        U[i,j,k,7] = h*(by[i,j,k] + by[i,j+1,k])
        U[i,j,k,8] = h*(bz[i,j,k] + bz[i,j,k+1])
    end
end

# ---------------------------------------------------------------------------------------------
# Integrator: recon -> flux -> (EMF) -> update. Each reconstruction computed once, each face once.
# ---------------------------------------------------------------------------------------------

@kernel function recon_kernel!(rec, @Const(U), o::NTuple{3,Int}, clo::NTuple{3,Int}, chi::NTuple{3,Int},
                               gamma::T, smallr::T, pfl::T, dtdx::T, nv::Val, ::Val{CT}) where {T,CT}
    I = @index(Global, NTuple)
    i = I[1] + o[1]; j = I[2] + o[2]; k = I[3] + o[3]
    @inbounds if !deep(i, j, k, clo, chi, MREC)
        m0  = cons2prim(loadq(U, i, j, k, nv), gamma, smallr, pfl)
        sx = prim_slope(cons2prim(loadq(U, i-1, j, k, nv), gamma, smallr, pfl), m0,
                        cons2prim(loadq(U, i+1, j, k, nv), gamma, smallr, pfl))
        sy = prim_slope(cons2prim(loadq(U, i, j-1, k, nv), gamma, smallr, pfl), m0,
                        cons2prim(loadq(U, i, j+1, k, nv), gamma, smallr, pfl))
        sz = prim_slope(cons2prim(loadq(U, i, j, k-1, nv), gamma, smallr, pfl), m0,
                        cons2prim(loadq(U, i, j, k+1, nv), gamma, smallr, pfl))
        mh = cons2prim(hancock(m0, sx, sy, sz, dtdx, gamma), gamma, smallr, pfl)
        if mh[1] <= smallr || mh[5] <= pfl
            mh = m0
        end
        for v in 1:9
            rec[i,j,k,v] = mh[v]; rec[i,j,k,9+v] = sx[v]; rec[i,j,k,18+v] = sy[v]; rec[i,j,k,27+v] = sz[v]
        end
        if CT   # GS05 reference EMF: predictor half-step velocity, start-of-step field (sweep-fixes 7256715)
            vx = mh[2]; vy = mh[3]; vz = mh[4]; bx = m0[6]; by = m0[7]; bz = m0[8]
            rec[i,j,k,37] = vz*by - vy*bz
            rec[i,j,k,38] = vx*bz - vz*bx
            rec[i,j,k,39] = vy*bx - vx*by
        end
    end
end

# fluxes through the lower d-faces; normal index range [g+1, g+n+1], transverse [g, g+n+1]
@kernel function flux_kernel!(F, @Const(rec), @Const(bx), @Const(by), @Const(bz),
                              o::NTuple{3,Int}, g::Int, n::NTuple{3,Int}, clo::NTuple{3,Int}, chi::NTuple{3,Int},
                              gamma::T, ch::T, smallr::T, pfl::T, rv::Val, ::Val{CT}, ::Val{NF}) where {T,CT,NF}
    I = @index(Global, NTuple)
    i = I[1] + o[1]; j = I[2] + o[2]; k = I[3] + o[3]
    @inbounds if !deep(i, j, k, clo, chi, MFLX)
        mh = rget(rec, i, j, k, 0)
        if i >= g+1
            ml = rget(rec, i-1, j, k, 0)
            L = iflo(padd(ml, rget(rec, i-1, j, k, 9), one(T)), ml, smallr, pfl)
            R = iflo(padd(mh, rget(rec, i, j, k, 9), -one(T)), mh, smallr, pfl)
            f = riemann_dir(rv, Val(CT), L, R, 1, gamma, ch, smallr, pfl, CT ? bx[i,j,k] : zero(T))
            for c in 1:NF; F[i,j,k,c,1] = f[c]; end
        end
        if j >= g+1
            ml = rget(rec, i, j-1, k, 0)
            L = iflo(padd(ml, rget(rec, i, j-1, k, 18), one(T)), ml, smallr, pfl)
            R = iflo(padd(mh, rget(rec, i, j, k, 18), -one(T)), mh, smallr, pfl)
            f = riemann_dir(rv, Val(CT), L, R, 2, gamma, ch, smallr, pfl, CT ? by[i,j,k] : zero(T))
            for c in 1:NF; F[i,j,k,c,2] = f[c]; end
        end
        if k >= g+1
            ml = rget(rec, i, j, k-1, 0)
            L = iflo(padd(ml, rget(rec, i, j, k-1, 27), one(T)), ml, smallr, pfl)
            R = iflo(padd(mh, rget(rec, i, j, k, 27), -one(T)), mh, smallr, pfl)
            f = riemann_dir(rv, Val(CT), L, R, 3, gamma, ch, smallr, pfl, CT ? bz[i,j,k] : zero(T))
            for c in 1:NF; F[i,j,k,c,3] = f[c]; end
        end
    end
end

# GS05 edge EMFs on edge index range [g+1, g+n+1] (along-edge index ≤ g+n)
@kernel function emf_kernel!(ex, ey, ez, @Const(F), @Const(rec), o::NTuple{3,Int}, g::Int, n::NTuple{3,Int},
                             clo::NTuple{3,Int}, chi::NTuple{3,Int})
    I = @index(Global, NTuple)
    i = I[1] + o[1]; j = I[2] + o[2]; k = I[3] + o[3]
    @inbounds if !deep(i, j, k, clo, chi, MEMF)
        if i <= g+n[1]   # ex at (i, j-½, k-½): a = y, b = z
            ex[i,j,k] = gs05(-F[i,j,k,8,2], -F[i,j,k-1,8,2], F[i,j,k,7,3], F[i,j-1,k,7,3],
                             F[i,j,k,1,2], F[i,j,k-1,1,2], F[i,j,k,1,3], F[i,j-1,k,1,3],
                             rec[i,j,k,37], rec[i,j-1,k,37], rec[i,j,k-1,37], rec[i,j-1,k-1,37])
        end
        if j <= g+n[2]   # ey at (i-½, j, k-½): a = z, b = x
            ey[i,j,k] = gs05(-F[i,j,k,6,3], -F[i-1,j,k,6,3], F[i,j,k,8,1], F[i,j,k-1,8,1],
                             F[i,j,k,1,3], F[i-1,j,k,1,3], F[i,j,k,1,1], F[i,j,k-1,1,1],
                             rec[i,j,k,38], rec[i,j,k-1,38], rec[i-1,j,k,38], rec[i-1,j,k-1,38])
        end
        if k <= g+n[3]   # ez at (i-½, j-½, k): a = x, b = y
            ez[i,j,k] = gs05(-F[i,j,k,7,1], -F[i,j-1,k,7,1], F[i,j,k,6,2], F[i-1,j,k,6,2],
                             F[i,j,k,1,1], F[i,j-1,k,1,1], F[i,j,k,1,2], F[i-1,j,k,1,2],
                             rec[i,j,k,39], rec[i-1,j,k,39], rec[i,j-1,k,39], rec[i-1,j-1,k,39])
        end
    end
end

# conservative cell update (in place: all fluxes were computed first). GLM: ψ damping + floor.
@kernel function update_cells_kernel!(U, @Const(F), o::NTuple{3,Int}, clo::NTuple{3,Int}, chi::NTuple{3,Int},
                                      dtdx::T, glm_fac::T, smallr::T, pfl::T, gamma::T, ::Val{CT}) where {T,CT}
    I = @index(Global, NTuple)
    i = I[1] + o[1]; j = I[2] + o[2]; k = I[3] + o[3]
    @inbounds if !deep(i, j, k, clo, chi, MU)
        nvu = CT ? 5 : 9
        r = ntuple(Val(9)) do v
            v <= nvu ? U[i,j,k,v] + dtdx*((F[i,j,k,v,1] - F[i+1,j,k,v,1]) + (F[i,j,k,v,2] - F[i,j+1,k,v,2]) +
                                           (F[i,j,k,v,3] - F[i,j,k+1,v,3])) : zero(T)
        end
        if r[1] < smallr       # positivity floor (GLMMHDTurb): static floored cell, B kept
            bxc = CT ? U[i,j,k,6] : r[6]; byc = CT ? U[i,j,k,7] : r[7]; bzc = CT ? U[i,j,k,8] : r[8]
            r = (smallr, zero(T), zero(T), zero(T), pfl/(gamma-one(T)) + T(0.5)*(bxc*bxc+byc*byc+bzc*bzc),
                 r[6], r[7], r[8], r[9])
        end
        for v in 1:nvu; U[i,j,k,v] = r[v]; end
        if !CT; U[i,j,k,9] = r[9]*glm_fac; end
    end
end

# CT face update: owned faces normal-index g+1 .. g+nfo (nfo = n periodic root, n+1 child)
@kernel function update_faces_kernel!(bx, by, bz, @Const(ex), @Const(ey), @Const(ez), o::NTuple{3,Int}, g::Int,
                                      n::NTuple{3,Int}, nfo::NTuple{3,Int}, clo::NTuple{3,Int}, chi::NTuple{3,Int}, dtdx::T) where {T}
    I = @index(Global, NTuple)
    i = I[1] + o[1]; j = I[2] + o[2]; k = I[3] + o[3]
    @inbounds if !deep(i, j, k, clo, chi, MU)
        inx = i <= g+n[1]; iny = j <= g+n[2]; inz = k <= g+n[3]
        if (i <= g+nfo[1]) & iny & inz
            bx[i,j,k] -= dtdx*((ez[i,j+1,k] - ez[i,j,k]) - (ey[i,j,k+1] - ey[i,j,k]))
        end
        if inx & (j <= g+nfo[2]) & inz
            by[i,j,k] -= dtdx*((ex[i,j,k+1] - ex[i,j,k]) - (ez[i+1,j,k] - ez[i,j,k]))
        end
        if inx & iny & (k <= g+nfo[3])
            bz[i,j,k] -= dtdx*((ey[i+1,j,k] - ey[i,j,k]) - (ex[i,j+1,k] - ex[i,j,k]))
        end
    end
end

# per-cell signal speed for the CFL (0 on ghosts and on cells the parent does not update)
@kernel function sig_kernel!(sig, @Const(U), g::Int, n::NTuple{3,Int}, clo::NTuple{3,Int}, chi::NTuple{3,Int},
                             gamma::T, smallr::T, pfl::T, nv::Val) where {T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        interior = (g < i <= g+n[1]) & (g < j <= g+n[2]) & (g < k <= g+n[3])
        if interior && !deep(i, j, k, clo, chi, MU)
            sig[i,j,k] = signal_speed(cons2prim(loadq(U, i, j, k, nv), gamma, smallr, pfl), gamma)
        else
            sig[i,j,k] = zero(T)
        end
    end
end
