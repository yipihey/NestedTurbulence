# hierarchy.jl — static nested patches, adaptive (Enzo-style) subcycling, Berger-Oliger order.

const G = 2   # ghost cells per side (enough for the GS05 EMF on the patch-boundary edges; see kernels.jl)

Base.@kwdef struct Params{T}
    gamma::T        = T(5/3)
    courant::T      = T(0.4)    # on the Σ_d(|v_d|+c_f,d) unsplit criterion (GLMMHDTurb convention)
    smallr::T       = T(1e-6)
    smallc::T       = T(1e-4)
    glm_ch_scale::T = T(0.25)   # c_h = courant·dx/(3·dt_cfl)·scale (sweep-fixes: from the CFL step); 0.25 = GLMMHDTurb.
                                # Scale 1.0 at Courant 0.8 puts ψ at its own CFL limit and goes unstable on nests.
    glm_cp_coef::T  = T(0.18)
    boxlen::T       = T(1)
end
pfloor(p::Params) = p.smallr * p.smallc^2 / p.gamma

mutable struct Patch
    lev::Int
    lo::NTuple{3,Int}      # global index of the first interior cell at this level
    n::NTuple{3,Int}       # interior cells
    Nlev::NTuple{3,Int}    # cells across the box at this level
    dx::Float64
    root::Bool
    per::NTuple{3,Bool}    # periodic dims: the patch spans the whole box there (root: all)
    U; bx; by; bz          # state (faces are 1-element dummies in GLM mode)
    rec; F; ex; ey; ez; sig
    # as parent (set if a child exists)
    clo::NTuple{3,Int}; chi::NTuple{3,Int}; nc::NTuple{3,Int}; cper::NTuple{3,Bool}
    sU; sbx; sby; sbz; so::NTuple{3,Int}
    # as child (accumulators at the parent's resolution)
    FA; EA
end

mutable struct Hierarchy
    patches::Vector{Patch}
    backend
    T::DataType
    mode::Symbol            # :glm or :ct
    riemann::Symbol         # :hll, :hlld, :llf
    p
    t::Vector{Float64}
    tp0::Vector{Float64}; tp1::Vector{Float64}
    nsteps::Vector{Int}
    time_interp::Bool       # 2nd-order θ-blended ghosts (true) or stale coarse ghosts (false)
    adaptive::Bool          # Enzo-style per-level CFL steps (true) or fixed ratio-2 subcycling (false)
    skip_covered::Bool      # parent skips work deep inside its child's footprint (true)
    wall::Vector{Float64}
    sources::Vector{Any}    # driving, probes, … (forcing.jl: apply_sources!, post_sync!, after_step!, finalize!)
end

ct(H::Hierarchy) = H.mode === :ct
nvs(H::Hierarchy) = ct(H) ? 8 : 9      # stored components
nvr(H::Hierarchy) = ct(H) ? 5 : 9      # conserved (updated / refluxed / restricted) components
nfl(H::Hierarchy) = ct(H) ? 8 : 9      # stored flux components
ng(P::Patch) = P.n .+ 2G
nfo(P::Patch) = ntuple(d -> P.per[d] ? P.n[d] : P.n[d] + 1, 3)   # owned faces per dim

_z(be, T, dims...) = KA.zeros(be, T, dims...)

function Patch(be, T, mode, lev, lo, n, Nlev, boxlen, per=(true,true,true))
    nx, ny, nz = n .+ 2G
    ctm = mode === :ct
    U = _z(be, T, nx, ny, nz, ctm ? 8 : 9)
    bx = ctm ? _z(be, T, nx+1, ny, nz) : _z(be, T, 1, 1, 1)
    by = ctm ? _z(be, T, nx, ny+1, nz) : _z(be, T, 1, 1, 1)
    bz = ctm ? _z(be, T, nx, ny, nz+1) : _z(be, T, 1, 1, 1)
    rec = _z(be, T, nx, ny, nz, ctm ? 39 : 36)
    F   = _z(be, T, nx, ny, nz, ctm ? 8 : 9, 3)        # face/edge indices never exceed g+n+1 = nx-1
    ex = ctm ? _z(be, T, nx, ny, nz) : _z(be, T, 1, 1, 1)
    ey = ctm ? _z(be, T, nx, ny, nz) : _z(be, T, 1, 1, 1)
    ez = ctm ? _z(be, T, nx, ny, nz) : _z(be, T, 1, 1, 1)
    sig = _z(be, T, nx, ny, nz)
    dx = Float64(boxlen) / Nlev[1]
    Patch(lev, lo, n, Nlev, dx, lev == 1, per, U, bx, by, bz, rec, F, ex, ey, ez, sig,
          (1,1,1), (0,0,0), (0,0,0), (false,false,false), nothing, nothing, nothing, nothing, (0,0,0), nothing, nothing)
end

"""
    Hierarchy(; N0, levels, mode=:glm, riemann=:hll, T=Float32, backend=KA.CPU(), params...)

`levels` lists the refined patches as `(lo, n)` pairs at levels 2, 3, … (global indices at their
own level, ratio 2 per level). Each must start on an odd global index, have an even size, and sit at
least 2 parent cells inside its parent. The root is the periodic box of `N0` cells per dimension
(an `Int`, or a 3-tuple for an anisotropic box; cells are cubic, dx = boxlen/N0[1]).
"""
function Hierarchy(; N0, levels=Tuple[], mode::Symbol=:glm, riemann::Symbol=:hll, T::DataType=Float32,
                   backend=KA.CPU(), time_interp::Bool=true, adaptive::Bool=true, skip_covered::Bool=true, kw...)
    mode in (:glm, :ct) || error("mode must be :glm or :ct")
    riemann in (:hll, :hlld, :llf) || error("riemann must be :hll, :hlld or :llf")
    p = Params{T}(; (k => T(v) for (k, v) in kw)...)
    N0 = N0 isa Integer ? (N0, N0, N0) : Tuple(N0)
    P1 = Patch(backend, T, mode, 1, (1,1,1), N0, N0, p.boxlen)
    ps = [P1]
    for (l, (lo, n)) in enumerate(levels)
        lev = l + 1
        Nlev = N0 .* 2^(lev-1)
        lo = Tuple(lo); n = Tuple(n)
        all(isodd, lo) || error("level $lev: lo must be odd (aligned to parent cells)")
        all(iseven, n) || error("level $lev: n must be even")
        par = ps[end]
        # a dimension the child spans completely is periodic (no coarse-fine boundary in it)
        per = ntuple(d -> lo[d] == 1 && n[d] == Nlev[d] && par.per[d], 3)
        a = (lo .+ 1) .÷ 2; b = a .+ n .÷ 2 .- 1
        all(d -> per[d] || (a[d] - SM >= par.lo[d] && b[d] + SM <= par.lo[d] + par.n[d] - 1), 1:3) ||
            error("level $lev: patch must sit ≥ $SM parent cells inside its parent (or span the box)")
        par.clo = a .- par.lo .+ G .+ 1
        par.chi = b .- par.lo .+ G .+ 1
        par.nc = n .÷ 2
        par.cper = per
        sn = ntuple(d -> per[d] ? par.nc[d] : par.nc[d] + 2SM, 3)      # snapshot box
        par.so = ntuple(d -> per[d] ? G : par.clo[d] - SM - 1, 3)
        par.sU = _z(backend, T, sn..., mode === :ct ? 8 : 9)
        if mode === :ct
            par.sbx = _z(backend, T, sn[1]+1, sn[2]+1, sn[3]+1)
            par.sby = _z(backend, T, sn[1]+1, sn[2]+1, sn[3]+1)
            par.sbz = _z(backend, T, sn[1]+1, sn[2]+1, sn[3]+1)
        end
        P = Patch(backend, T, mode, lev, lo, n, Nlev, p.boxlen, per)
        nc = n .÷ 2; nf = mode === :ct ? 8 : 9
        P.FA = (_z(backend, T, 2, nc[2], nc[3], nf), _z(backend, T, 2, nc[1], nc[3], nf), _z(backend, T, 2, nc[1], nc[2], nf))
        if mode === :ct
            P.EA = (_z(backend, T, nc[1], nc[2]+1, nc[3]+1), _z(backend, T, nc[1]+1, nc[2], nc[3]+1),
                    _z(backend, T, nc[1]+1, nc[2]+1, nc[3]))
        end
        push!(ps, P)
    end
    L = length(ps)
    Hierarchy(ps, backend, T, mode, riemann, p, zeros(L), zeros(L), zeros(L), zeros(Int, L),
              time_interp, adaptive, skip_covered, zeros(L), Any[])
end

"""
    nested_central(N0, nlev; n=N0, kw...)

`nlev` levels, each refined patch `n³` cells centred in the box (each covers the central n/(2N_parent)
of its parent). Requires n % 4 == 0 and n ≤ N0 (so every level keeps ≥ 2 parent cells of margin).
"""
function nested_central(N0::Int, nlev::Int; n::Int=N0, kw...)
    n % 4 == 0 || error("n must be a multiple of 4")
    levels = Tuple[]
    for lev in 2:nlev
        Nl = N0 * 2^(lev-1)
        lo = (Nl - n) ÷ 2 + 1
        push!(levels, ((lo,lo,lo), (n,n,n)))
    end
    Hierarchy(; N0=N0, levels=levels, kw...)
end

# ---------------------------------------------------------------------------------------------
# single-patch operations
# ---------------------------------------------------------------------------------------------
_sync(H) = KA.synchronize(H.backend)

function bcc!(H, P)
    ct(H) || return
    bcc_kernel!(H.backend)(P.U, P.bx, P.by, P.bz; ndrange=ng(P))
end

function fill_ghosts!(H::Hierarchy, l::Int)
    P = H.patches[l]; be = H.backend
    if P.root
        periodic_cells_kernel!(be)(P.U, G, P.n, nvs(H); ndrange=ng(P))
        if ct(H)
            periodic_faces_kernel!(be)(P.bx, G, P.n, 1; ndrange=size(P.bx))
            periodic_faces_kernel!(be)(P.by, G, P.n, 2; ndrange=size(P.by))
            periodic_faces_kernel!(be)(P.bz, G, P.n, 3; ndrange=size(P.bz))
        end
    else
        Q = H.patches[l-1]
        θ = H.time_interp ? clamp((H.t[l] - H.tp0[l-1]) / (H.tp1[l-1] - H.tp0[l-1]), 0.0, 1.0) : 0.0
        θT = H.T(θ)
        if ct(H)
            prolong_faces_kernel!(be)(P.bx, Q.sbx, Q.bx, Val(1), P.lo, G, P.n, Q.lo, G, Q.so, θT, P.per, Q.n; ndrange=size(P.bx))
            prolong_faces_kernel!(be)(P.by, Q.sby, Q.by, Val(2), P.lo, G, P.n, Q.lo, G, Q.so, θT, P.per, Q.n; ndrange=size(P.by))
            prolong_faces_kernel!(be)(P.bz, Q.sbz, Q.bz, Val(3), P.lo, G, P.n, Q.lo, G, Q.so, θT, P.per, Q.n; ndrange=size(P.bz))
        end
        prolong_cells_kernel!(be)(P.U, Q.sU, Q.U, P.bx, P.by, P.bz, P.lo, G, P.n, Q.lo, G, Q.so, θT, Val(ct(H)),
                                  P.per, Q.n, nvs(H); ndrange=ng(P))
    end
    bcc!(H, P)
end

function kbox(H::Hierarchy, P::Patch)
    H.skip_covered || return ((1,1,1), (0,0,0))
    big = 10^9                      # a periodic dim of the footprint is covered everywhere
    (ntuple(d -> P.cper[d] ? -big : P.clo[d], 3), ntuple(d -> P.cper[d] ? big : P.chi[d], 3))
end

function cfl_dt(H::Hierarchy, P::Patch)
    p = H.p; clo, chi = kbox(H, P)
    sig_kernel!(H.backend)(P.sig, P.U, G, P.n, clo, chi, p.gamma, p.smallr, pfloor(p), Val(nvs(H)); ndrange=ng(P))
    _sync(H)
    smax = Float64(maximum(P.sig))
    return Float64(p.courant) * P.dx / smax
end

function snapshot!(H::Hierarchy, P::Patch)
    be = H.backend
    copybox4_kernel!(be)(P.sU, P.U, P.so, nvs(H); ndrange=size(P.sU)[1:3])
    if ct(H)
        copybox3_kernel!(be)(P.sbx, P.bx, P.so; ndrange=size(P.sbx))
        copybox3_kernel!(be)(P.sby, P.by, P.so; ndrange=size(P.sby))
        copybox3_kernel!(be)(P.sbz, P.bz, P.so; ndrange=size(P.sbz))
    end
end

function step_patch!(H::Hierarchy, P::Patch, dt::Float64, dt_cfl::Float64)
    be = H.backend; p = H.p; T = H.T
    dtdx = T(dt / P.dx); pfl = pfloor(p)
    ch = ct(H) ? one(T) : T(Float64(p.courant) * P.dx / (3dt_cfl) * Float64(p.glm_ch_scale))
    glm_fac = (ct(H) || p.glm_cp_coef <= 0) ? one(T) : T(exp(-Float64(ch) * dt / (Float64(p.glm_cp_coef) * Float64(p.boxlen))))
    o1 = (G-1, G-1, G-1)                       # recon & flux over [g, g+n+1]
    r1 = P.n .+ 2
    clo, chi = kbox(H, P)
    recon_kernel!(be)(P.rec, P.U, o1, clo, chi, p.gamma, p.smallr, pfl, dtdx, Val(nvs(H)), Val(ct(H)); ndrange=r1)
    flux_kernel!(be)(P.F, P.rec, P.bx, P.by, P.bz, o1, G, P.n, clo, chi, p.gamma, ch, p.smallr, pfl,
                     Val(H.riemann), Val(ct(H)), Val(nfl(H)); ndrange=r1)
    if ct(H)
        emf_kernel!(be)(P.ex, P.ey, P.ez, P.F, P.rec, (G, G, G), G, P.n, clo, chi; ndrange=P.n .+ 1)
    end
    update_cells_kernel!(be)(P.U, P.F, (G, G, G), clo, chi, dtdx, glm_fac, p.smallr, pfl, p.gamma, Val(ct(H)); ndrange=P.n)
    if ct(H)
        update_faces_kernel!(be)(P.bx, P.by, P.bz, P.ex, P.ey, P.ez, (G, G, G), G, P.n, nfo(P), clo, chi, dtdx;
                                 ndrange=P.n .+ 1)
        bcc!(H, P)
    end
end

function accumulate!(H::Hierarchy, P::Patch, dt::Float64)
    be = H.backend; T = H.T; nc = P.n .÷ 2
    P.per[1] || accum_flux_kernel!(be)(P.FA[1], P.F, Val(1), G, P.n, T(dt), nfl(H); ndrange=(nc[2], nc[3], 2))
    P.per[2] || accum_flux_kernel!(be)(P.FA[2], P.F, Val(2), G, P.n, T(dt), nfl(H); ndrange=(nc[1], nc[3], 2))
    P.per[3] || accum_flux_kernel!(be)(P.FA[3], P.F, Val(3), G, P.n, T(dt), nfl(H); ndrange=(nc[1], nc[2], 2))
    if ct(H)
        accum_emf_kernel!(be)(P.EA[1], P.EA[2], P.EA[3], P.ex, P.ey, P.ez, G, nc, T(dt), P.per; ndrange=nc .+ 1)
    end
end

emf_range(C::Patch, nc) = ntuple(d -> C.per[d] ? nc[d] : nc[d] + 4, 3)

function zero_accumulators!(C::Patch)
    foreach(a -> fill!(a, 0), C.FA)
    C.EA === nothing || foreach(a -> fill!(a, 0), C.EA)
end

function sync_from_child!(H::Hierarchy, P::Patch, C::Patch, dt::Float64)
    be = H.backend; T = H.T; nc = P.nc; idx = T(1 / P.dx)
    C.per[1] || reflux_kernel!(be)(P.U, P.F, C.FA[1], Val(1), P.clo, P.chi, T(dt), idx, nvr(H); ndrange=(nc[2], nc[3], 2))
    C.per[2] || reflux_kernel!(be)(P.U, P.F, C.FA[2], Val(2), P.clo, P.chi, T(dt), idx, nvr(H); ndrange=(nc[1], nc[3], 2))
    C.per[3] || reflux_kernel!(be)(P.U, P.F, C.FA[3], Val(3), P.clo, P.chi, T(dt), idx, nvr(H); ndrange=(nc[1], nc[2], 2))
    if ct(H)
        emf_correct_kernel!(be)(P.bx, P.by, P.bz, P.ex, P.ey, P.ez, C.EA[1], C.EA[2], C.EA[3], P.clo, nc, T(dt), idx, C.per;
                                ndrange=emf_range(C, nc))
        restrict_faces_kernel!(be)(P.bx, C.bx, Val(1), P.clo, G; ndrange=(nc[1]+1, nc[2], nc[3]))
        restrict_faces_kernel!(be)(P.by, C.by, Val(2), P.clo, G; ndrange=(nc[1], nc[2]+1, nc[3]))
        restrict_faces_kernel!(be)(P.bz, C.bz, Val(3), P.clo, G; ndrange=(nc[1], nc[2], nc[3]+1))
    end
    restrict_cells_kernel!(be)(P.U, C.U, P.clo, G, nvr(H); ndrange=nc)
    bcc!(H, P)
end

# ---------------------------------------------------------------------------------------------
# recursive driver
# ---------------------------------------------------------------------------------------------

"""
    evolve_level!(H, l, t_stop)

Advance level `l` from `H.t[l]` to `t_stop`. Every step of level `l` is followed by the complete
evolution of level `l+1` over that step (Berger-Oliger), then reflux, EMF correction, restriction.
Adaptive mode: the step is the level's own CFL step, evened out so the level lands exactly on
`t_stop` (n = ⌈remaining/dt_cfl⌉, dt = remaining/n) — a level whose CFL step exceeds its parent's
takes one step; coarse levels never pay for the finest level's signal speed.
"""
function evolve_level!(H::Hierarchy, l::Int, t_stop::Float64)
    P = H.patches[l]; L = length(H.patches)
    tol = 1e-12 * max(abs(t_stop), 1.0)
    while H.t[l] < t_stop - tol
        tw = time_ns()
        fill_ghosts!(H, l)
        dtc = cfl_dt(H, P)
        rem = t_stop - H.t[l]
        if H.adaptive || l == 1
            nsub = max(1, ceil(Int, rem / dtc - 1e-9))
            dt = rem / nsub
        else
            dt = (t_stop - H.tp0[l-1]) / 2       # fixed ratio 2
            dt > dtc * (1 + 1e-9) && @warn "level $l: fixed ratio-2 step exceeds its CFL step" dt dtc
        end
        has_child = l < L
        if has_child
            snapshot!(H, P)
            H.tp0[l] = H.t[l]; H.tp1[l] = H.t[l] + dt
        end
        step_patch!(H, P, dt, dtc)      # c_h from the CFL step, not the shortened one (sweep-fixes 176058b)
        for s in H.sources; apply_sources!(s, H, P, H.t[l], dt); end
        l > 1 && accumulate!(H, P, dt)
        _sync(H)
        H.wall[l] += (time_ns() - tw) * 1e-9
        if has_child
            C = H.patches[l+1]
            zero_accumulators!(C)
            evolve_level!(H, l+1, H.t[l] + dt)
            tw = time_ns()
            sync_from_child!(H, P, C, dt)
            for s in H.sources; post_sync!(s, H, P); end
            _sync(H)
            H.wall[l] += (time_ns() - tw) * 1e-9
        end
        H.t[l] += dt
        H.nsteps[l] += 1
        _sync(H)
        for s in H.sources; after_step!(s, H, l); end
    end
    H.t[l] = t_stop
    return H
end

"Advance the whole hierarchy to time `t_end`."
function evolve!(H::Hierarchy, t_end::Real)
    evolve_level!(H, 1, Float64(t_end))
    for s in H.sources; finalize!(s, H); end
    H
end
