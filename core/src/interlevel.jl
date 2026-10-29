# interlevel.jl — kernels coupling a parent patch and its (ratio-2) child patch.
#
# Geometry. A child with first global cell lo_f (odd) and n_f (even) cells per dim covers the
# parent's global cells a = (lo_f+1)/2 .. b = a + n_f/2 - 1, i.e. parent-local [clo, chi].
# Fine global cell G_f sits in coarse global cell fld(G_f+1, 2): odd G_f is the lower half.
#
# Second-order time interpolation (Berger-Oliger order). The parent is advanced first. Before its
# step it copies the cells [clo-2, chi+2] (and faces [clo-2, chi+3]) into a snapshot (t_p0); after
# its step the same cells hold t_p1. A child step starting at t has its ghosts prolonged from
#     q(θ) = (1-θ)·snapshot + θ·current,   θ = (t - t_p0)/(t_p1 - t_p0),
# i.e. ghosts are evaluated at the start of the child step; the MUSCL-Hancock predictor then
# advances ghosts and interior alike to the half step. (Setting θ ≡ 0 reproduces the first-order
# "stale coarse" boundary of the Rust cascade driver, kept as an option for comparison.)
#
# Conservation. Berger-Colella: the child accumulates Σ dt_f·⟨F_f⟩ on the coarse faces of its
# boundary; the parent replaces its dt_c·F_c there. CT: the child accumulates Σ dt_f·⟨E_f⟩ on the
# coarse edges of its boundary surface; the parent corrects every face outside the footprint that
# touches such an edge with the curl of (Σ dt_f⟨E_f⟩ - dt_c·E_c). Faces on and inside the footprint
# are restricted (area averages of fine faces); by Stokes' theorem the restricted boundary face
# equals the EMF-corrected one, so ∇·B stays at rounding in every coarse cell.

const SM = 2   # snapshot margin: prolongation reads coarse cells/faces a-1 .. b+1 and their ±1 slope neighbours

@kernel function copybox4_kernel!(dst, @Const(src), o::NTuple{3,Int}, nv::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds for v in 1:nv
        dst[i,j,k,v] = src[o[1]+i, o[2]+j, o[3]+k, v]
    end
end
@kernel function copybox3_kernel!(dst, @Const(src), o::NTuple{3,Int})
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i,j,k] = src[o[1]+i, o[2]+j, o[3]+k]
end

@inline _gf(lo, idx, g) = lo + idx - g - 1                 # local -> global cell index
@inline _cg(Gf) = fld(Gf + 1, 2)                          # fine global -> coarse global cell

# --- prolongation of child ghost faces (CT). Normal: the coarse face, or the mean of the two coarse
#     faces of the coarse cell for a fine mid-plane face; transverse: MonCen-limited linear in the
#     coarse face field (second order). Ghost faces only feed the reconstruction: every OWNED fine
#     face is advanced by the curl of edge EMFs, so ∇·B of the evolved cells stays at rounding
#     whatever the ghosts hold. (Piecewise-constant transverse prolongation — exactly solenoidal in
#     the ghosts — measured first order in space and masked the time interpolation: bench/ghost_accuracy.jl.)
# Periodic-span dimensions. A child dimension that spans the whole box (lo = 1, n = cells across the
# box) is periodic: no coarse-fine boundary in it. `per[d]` marks such dimensions; local indices are
# wrapped into the interior there (_pw), both for the child's own ghosts and for the parent cells read.
@inline _pw(i, g, n, per) = per ? g + mod(i - g - 1, n) + 1 : i

@kernel function prolong_faces_kernel!(bf, @Const(Sb), @Const(Cb), D::Val{d}, lo_f::NTuple{3,Int}, g::Int,
                                       n_f::NTuple{3,Int}, lo_c::NTuple{3,Int}, gc::Int, so::NTuple{3,Int}, θ::T,
                                       per::NTuple{3,Bool}, n_c::NTuple{3,Int}) where {d,T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        idx = (i, j, k)
        owned = ntuple(Val(3)) do e
            (e == d && !per[e]) ? (g+1 <= idx[e] <= g+n_f[e]+1) : (g+1 <= idx[e] <= g+n_f[e])
        end
        if !(owned[1] & owned[2] & owned[3])
            w = ntuple(e -> _pw(idx[e], g, n_f[e], per[e]), Val(3))          # wrap periodic dims
            ow = ntuple(Val(3)) do e
                (e == d && !per[e]) ? (g+1 <= w[e] <= g+n_f[e]+1) : (g+1 <= w[e] <= g+n_f[e])
            end
            if ow[1] & ow[2] & ow[3]
                bf[i,j,k] = bf[w[1], w[2], w[3]]                              # ghost only in periodic dims
            else
                Gf = ntuple(e -> _gf(lo_f[e], w[e], g), Val(3))
                ci = ntuple(Val(3)) do e
                    Gc = e == d ? (isodd(Gf[e]) ? (Gf[e]+1) ÷ 2 : Gf[e] ÷ 2) : _cg(Gf[e])
                    Gc - lo_c[e] + gc + 1                                     # parent-local
                end
                mid = iseven(Gf[d])
                a = (one(T) - θ); h = T(0.5); q4 = T(0.25)
                t1 = d == 1 ? 2 : 1; t2 = d == 3 ? 2 : 3
                o1 = ifelse(isodd(Gf[t1]), -q4, q4); o2 = ifelse(isodd(Gf[t2]), -q4, q4)
                @inline function bc(c1, c2, c3)
                    c1 = _pw(c1, gc, n_c[1], per[1]); c2 = _pw(c2, gc, n_c[2], per[2]); c3 = _pw(c3, gc, n_c[3], per[3])
                    a*Sb[c1-so[1], c2-so[2], c3-so[3]] + θ*Cb[c1, c2, c3]
                end
                # coarse face value + MonCen-limited transverse slopes at the fine sub-face position
                @inline function face(c)
                    v0 = bc(c[1], c[2], c[3])
                    u1 = ntuple(e -> e == t1 ? 1 : 0, Val(3)); u2 = ntuple(e -> e == t2 ? 1 : 0, Val(3))
                    s1 = moncen(v0 - bc(c[1]-u1[1], c[2]-u1[2], c[3]-u1[3]), bc(c[1]+u1[1], c[2]+u1[2], c[3]+u1[3]) - v0)
                    s2 = moncen(v0 - bc(c[1]-u2[1], c[2]-u2[2], c[3]-u2[3]), bc(c[1]+u2[1], c[2]+u2[2], c[3]+u2[3]) - v0)
                    v0 + o1*s1 + o2*s2
                end
                c0 = face(ci)
                if mid
                    c1 = face(ntuple(e -> ci[e] + (e == d ? 1 : 0), Val(3)))
                    bf[i,j,k] = h*(c0 + c1)
                else
                    bf[i,j,k] = c0
                end
            end
        end
    end
end

# --- prolongation of child ghost cells: θ-blended coarse cell + MonCen-limited linear slopes at
#     ±¼ coarse cell. GLM: all 9 conserved. CT: ρ, ρv linear; the non-magnetic energy E - ½|B_c|²
#     linear, then + ½|B_f|² from the (already prolonged) fine faces, so ghost pressures are not
#     polluted by the face/cell B mismatch.
@kernel function prolong_cells_kernel!(Uf, @Const(Sc), @Const(Cc), @Const(bx), @Const(by), @Const(bz),
                                       lo_f::NTuple{3,Int}, g::Int, n_f::NTuple{3,Int}, lo_c::NTuple{3,Int},
                                       gc::Int, so::NTuple{3,Int}, θ::T, ::Val{CT}, per::NTuple{3,Bool},
                                       n_c::NTuple{3,Int}, nvs::Int) where {T,CT}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ghost = (i <= g) | (i > g+n_f[1]) | (j <= g) | (j > g+n_f[2]) | (k <= g) | (k > g+n_f[3])
        if ghost
            iw = _pw(i, g, n_f[1], per[1]); jw = _pw(j, g, n_f[2], per[2]); kw = _pw(k, g, n_f[3], per[3])
            if (g < iw <= g+n_f[1]) & (g < jw <= g+n_f[2]) & (g < kw <= g+n_f[3])
                for v in 1:nvs; Uf[i,j,k,v] = Uf[iw,jw,kw,v]; end            # ghost only in periodic dims
            else
                Gfx = _gf(lo_f[1], iw, g); Gfy = _gf(lo_f[2], jw, g); Gfz = _gf(lo_f[3], kw, g)
                ic = _cg(Gfx) - lo_c[1] + gc + 1; jc = _cg(Gfy) - lo_c[2] + gc + 1; kc = _cg(Gfz) - lo_c[3] + gc + 1
                q4 = T(0.25)
                ox = ifelse(isodd(Gfx), -q4, q4); oy = ifelse(isodd(Gfy), -q4, q4); oz = ifelse(isodd(Gfz), -q4, q4)
                a = one(T) - θ
                @inline function qc(di, dj, dk, v)
                    c1 = _pw(ic+di, gc, n_c[1], per[1]); c2 = _pw(jc+dj, gc, n_c[2], per[2]); c3 = _pw(kc+dk, gc, n_c[3], per[3])
                    a*Sc[c1-so[1], c2-so[2], c3-so[3], v] + θ*Cc[c1, c2, c3, v]
                end
                @inline function qv(di, dj, dk, v)
                    if CT && v == 5
                        b6 = qc(di,dj,dk,6); b7 = qc(di,dj,dk,7); b8 = qc(di,dj,dk,8)
                        return qc(di,dj,dk,5) - T(0.5)*(b6*b6 + b7*b7 + b8*b8)
                    end
                    return qc(di, dj, dk, v)
                end
                nvp = CT ? 5 : 9
                for v in 1:nvp
                    q0 = qv(0,0,0,v)
                    sx = moncen(q0 - qv(-1,0,0,v), qv(1,0,0,v) - q0)
                    sy = moncen(q0 - qv(0,-1,0,v), qv(0,1,0,v) - q0)
                    sz = moncen(q0 - qv(0,0,-1,v), qv(0,0,1,v) - q0)
                    Uf[i,j,k,v] = q0 + ox*sx + oy*sy + oz*sz
                end
                if CT
                    h = T(0.5)
                    bxc = h*(bx[i,j,k] + bx[i+1,j,k]); byc = h*(by[i,j,k] + by[i,j+1,k]); bzc = h*(bz[i,j,k] + bz[i,j,k+1])
                    Uf[i,j,k,5] += h*(bxc*bxc + byc*byc + bzc*bzc)
                end
            end
        end
    end
end

# --- restriction of the covered parent cells (volume average) and faces (area average)
@kernel function restrict_cells_kernel!(Uc, @Const(Uf), clo::NTuple{3,Int}, g::Int, nvr::Int)
    I, J, K = @index(Global, NTuple)
    @inbounds begin
        ic = clo[1]+I-1; jc = clo[2]+J-1; kc = clo[3]+K-1
        fi = g + 2I - 1; fj = g + 2J - 1; fk = g + 2K - 1
        T = eltype(Uc)
        for v in 1:nvr
            s = (Uf[fi,fj,fk,v] + Uf[fi+1,fj,fk,v]) + (Uf[fi,fj+1,fk,v] + Uf[fi+1,fj+1,fk,v]) +
                (Uf[fi,fj,fk+1,v] + Uf[fi+1,fj,fk+1,v]) + (Uf[fi,fj+1,fk+1,v] + Uf[fi+1,fj+1,fk+1,v])
            Uc[ic,jc,kc,v] = T(0.125)*s
        end
    end
end
@kernel function restrict_faces_kernel!(bc, @Const(bf), D::Val{d}, clo::NTuple{3,Int}, g::Int) where {d}
    I = @index(Global, NTuple)
    @inbounds begin
        ci = ntuple(e -> clo[e] + I[e] - 1, Val(3))
        f0 = ntuple(e -> g + 2I[e] - 1, Val(3))
        t1 = d == 1 ? 2 : 1; t2 = d == 3 ? 2 : 3
        e1 = ntuple(e -> e == t1 ? 1 : 0, Val(3)); e2 = ntuple(e -> e == t2 ? 1 : 0, Val(3))
        T = eltype(bc)
        s = (bf[f0[1], f0[2], f0[3]] + bf[f0[1]+e1[1], f0[2]+e1[2], f0[3]+e1[3]]) +
            (bf[f0[1]+e2[1], f0[2]+e2[2], f0[3]+e2[3]] + bf[f0[1]+e1[1]+e2[1], f0[2]+e1[2]+e2[2], f0[3]+e1[3]+e2[3]])
        bc[ci[1], ci[2], ci[3]] = T(0.25)*s
    end
end

# --- child: accumulate Σ dt·⟨F⟩ on the coarse faces of its boundary (FA[side, t1, t2, c]),
#     side 1 = lower face of fine cell g+1, side 2 = lower face of fine cell g+n+1 (upper boundary)
@kernel function accum_flux_kernel!(FA, @Const(F), D::Val{d}, g::Int, n_f::NTuple{3,Int}, dt::T, nf::Int) where {d,T}
    A, B, s = @index(Global, NTuple)
    @inbounds begin
        t1 = d == 1 ? 2 : 1; t2 = d == 3 ? 2 : 3
        fn = s == 1 ? g + 1 : g + n_f[d] + 1
        f1 = g + 2A - 1; f2 = g + 2B - 1
        q = T(0.25)*dt
        for c in 1:nf
            acc = zero(T)
            for b2 in 0:1, b1 in 0:1
                idx = ntuple(e -> e == d ? fn : (e == t1 ? f1 + b1 : f2 + b2), Val(3))
                acc += F[idx[1], idx[2], idx[3], c, d]
            end
            FA[s, A, B, c] += q*acc
        end
    end
end

# --- parent: Berger-Colella reflux of the cells just outside the footprint
@kernel function reflux_kernel!(Uc, @Const(Fc), @Const(FA), D::Val{d}, clo::NTuple{3,Int}, chi::NTuple{3,Int},
                                dtc::T, idx_c::T, nvr::Int) where {d,T}
    A, B, s = @index(Global, NTuple)
    @inbounds begin
        t1 = d == 1 ? 2 : 1; t2 = d == 3 ? 2 : 3
        fn  = s == 1 ? clo[d] : chi[d] + 1             # parent-local index of the face's upper cell
        out = s == 1 ? clo[d] - 1 : chi[d] + 1          # the uncovered cell sharing that face
        sg  = s == 1 ? one(T) : -one(T)
        fi = ntuple(e -> e == d ? fn  : (e == t1 ? clo[t1] + A - 1 : clo[t2] + B - 1), Val(3))
        oi = ntuple(e -> e == d ? out : (e == t1 ? clo[t1] + A - 1 : clo[t2] + B - 1), Val(3))
        for c in 1:nvr
            Uc[oi[1], oi[2], oi[3], c] += sg*idx_c*(dtc*Fc[fi[1], fi[2], fi[3], c, d] - FA[s, A, B, c])
        end
    end
end

# --- child (CT): accumulate Σ dt·⟨E⟩ on coarse surface edges, box arrays at coarse resolution
# a surface edge lies on a face of the footprint box normal to a NON-periodic dimension
@inline _sf(R, m, per) = !per & ((R == 1) | (R == m + 1))
@inline _surf_ex(I, J, K, nc, per) = _sf(J, nc[2], per[2]) | _sf(K, nc[3], per[3])
@inline _surf_ey(I, J, K, nc, per) = _sf(I, nc[1], per[1]) | _sf(K, nc[3], per[3])
@inline _surf_ez(I, J, K, nc, per) = _sf(I, nc[1], per[1]) | _sf(J, nc[2], per[2])

@kernel function accum_emf_kernel!(EAx, EAy, EAz, @Const(ex), @Const(ey), @Const(ez), g::Int, nc::NTuple{3,Int}, dt::T,
                                   per::NTuple{3,Bool}) where {T}
    I, J, K = @index(Global, NTuple)          # ndrange nc .+ 1
    @inbounds begin
        h = T(0.5)*dt
        fi = g + 2I - 1; fj = g + 2J - 1; fk = g + 2K - 1
        if I <= nc[1] && _surf_ex(I, J, K, nc, per)
            EAx[I,J,K] += h*(ex[fi,fj,fk] + ex[fi+1,fj,fk])
        end
        if J <= nc[2] && _surf_ey(I, J, K, nc, per)
            EAy[I,J,K] += h*(ey[fi,fj,fk] + ey[fi,fj+1,fk])
        end
        if K <= nc[3] && _surf_ez(I, J, K, nc, per)
            EAz[I,J,K] += h*(ez[fi,fj,fk] + ez[fi,fj,fk+1])
        end
    end
end

# --- parent (CT): EMF correction of the faces outside the footprint touching corrected edges.
#     Gather over faces (no atomics). ΔE·dt = Σ dt_f⟨E_f⟩ - dt_c·E_c on surface edges, 0 elsewhere.
@kernel function emf_correct_kernel!(bx, by, bz, @Const(ex), @Const(ey), @Const(ez), @Const(EAx), @Const(EAy), @Const(EAz),
                                     clo::NTuple{3,Int}, nc::NTuple{3,Int}, dtc::T, idx_c::T, per::NTuple{3,Bool}) where {T}
    I0 = @index(Global, NTuple)
    @inbounds begin
        # parent-local face/edge index; relative index R = local - clo + 1 runs over [-1, nc+2]
        # (over [1, nc] in a periodic dimension, where the footprint spans the whole patch)
        i = clo[1] + I0[1] - (per[1] ? 1 : 3); j = clo[2] + I0[2] - (per[2] ? 1 : 3); k = clo[3] + I0[3] - (per[3] ? 1 : 3)
        Ri = i - clo[1] + 1; Rj = j - clo[2] + 1; Rk = k - clo[3] + 1
        @inline inr(R, m) = 1 <= R <= m
        @inline function dEx(a, b, c)   # local edge ex[a,b,c]
            I = a - clo[1] + 1; J = b - clo[2] + 1; K = c - clo[3] + 1
            (inr(I, nc[1]) & inr(J, nc[2]+1) & inr(K, nc[3]+1)) && _surf_ex(I, J, K, nc, per) ?
                EAx[I,J,K] - dtc*ex[a,b,c] : zero(T)
        end
        @inline function dEy(a, b, c)
            I = a - clo[1] + 1; J = b - clo[2] + 1; K = c - clo[3] + 1
            (inr(I, nc[1]+1) & inr(J, nc[2]) & inr(K, nc[3]+1)) && _surf_ey(I, J, K, nc, per) ?
                EAy[I,J,K] - dtc*ey[a,b,c] : zero(T)
        end
        @inline function dEz(a, b, c)
            I = a - clo[1] + 1; J = b - clo[2] + 1; K = c - clo[3] + 1
            (inr(I, nc[1]+1) & inr(J, nc[2]+1) & inr(K, nc[3])) && _surf_ez(I, J, K, nc, per) ?
                EAz[I,J,K] - dtc*ez[a,b,c] : zero(T)
        end
        # footprint faces (restricted later) are skipped
        if !(inr(Ri, nc[1]+1) & inr(Rj, nc[2]) & inr(Rk, nc[3]))
            bx[i,j,k] -= idx_c*((dEz(i,j+1,k) - dEz(i,j,k)) - (dEy(i,j,k+1) - dEy(i,j,k)))
        end
        if !(inr(Ri, nc[1]) & inr(Rj, nc[2]+1) & inr(Rk, nc[3]))
            by[i,j,k] -= idx_c*((dEx(i,j,k+1) - dEx(i,j,k)) - (dEz(i+1,j,k) - dEz(i,j,k)))
        end
        if !(inr(Ri, nc[1]) & inr(Rj, nc[2]) & inr(Rk, nc[3]+1))
            bz[i,j,k] -= idx_c*((dEy(i+1,j,k) - dEy(i,j,k)) - (dEx(i,j+1,k) - dEx(i,j,k)))
        end
    end
end
