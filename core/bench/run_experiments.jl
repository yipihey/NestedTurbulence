# run_experiments.jl — every dataset shown in the book and the laboratory.
#
#   julia -t auto --project=core core/bench/run_experiments.jl [names...]
#
# writes app/data/<name>.json. Names: ghost convergence adaptive skip otnested throughput otref (default: all).
# All runs are CPU (KernelAbstractions CPU backend); the hardware is recorded in each file.

using NestedMHD, Printf
const NM = NestedMHD
const OUT = joinpath(@__DIR__, "..", "..", "app", "data")
mkpath(OUT)

# ---------------------------------------------------------------- minimal JSON writer
js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
js(x::Symbol) = js(String(x))
js(x::Bool) = x ? "true" : "false"
js(x::Integer) = string(x)
js(x::Real) = isfinite(x) ? @sprintf("%.6g", x) : "null"
js(x::Nothing) = "null"
js(x::AbstractVector) = "[" * join((js(v) for v in x), ",") * "]"
js(x::AbstractMatrix) = js([x[:, j] for j in axes(x, 2)])          # column-major: list of columns
js(x::Tuple) = js(collect(x))
js(x::NamedTuple) = js(Dict(pairs(x)))
js(x::AbstractDict) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in x), ",") * "}"
function save(name, obj)
    meta = Dict("generated" => string(Base.Libc.strftime("%Y-%m-%d %H:%M", time())), "threads" => Threads.nthreads(),
                "cpu" => Sys.cpu_info()[1].model, "julia" => string(VERSION))
    open(joinpath(OUT, name * ".json"), "w") do io
        write(io, js(Dict("meta" => meta, "data" => obj)))
    end
    println("wrote ", name, ".json")
end

# ---------------------------------------------------------------- shared problems
const A = 0.1
cp_ic(x, y, z) = (1.0, 0.0, -A*sin(2π*x), -A*cos(2π*x), 0.1, 1.0, A*sin(2π*x), A*cos(2π*x))
cp_A(x, y, z)  = (0.0, A*sin(2π*x)/(2π), y + A*cos(2π*x)/(2π))
const b0 = 1/sqrt(4π)
ot3(x, y, z) = (25/(36π)*(1+0.1*sin(2π*z)), -sin(2π*y), sin(2π*x), 0.3*sin(2π*x), 5/(12π), -b0*sin(2π*y), b0*sin(4π*x), 0.2*b0)
aot3(x, y, z) = (0.0, 0.2*b0*x, b0*(cos(4π*x)/(4π) + cos(2π*y)/(2π)))
ot2(x, y, z) = (25/(36π), -sin(2π*y), sin(2π*x), 0.0, 5/(12π), -b0*sin(2π*y), b0*sin(4π*x), 0.0)
aot2(x, y, z) = (0.0, 0.0, b0*(cos(4π*x)/(4π) + cos(2π*y)/(2π)))
cp_hier(N, mode, ti; courant=0.8) = Hierarchy(N0=(N,8,8), levels=[((N÷2+1,5,5),(N,8,8))], mode=mode, T=Float64,
                                               time_interp=ti, courant=courant)

# ---------------------------------------------------------------- 1. ghost-zone time accuracy
function ghost_err(H, l, t)
    P = H.patches[l]; g = 2; U = Array(P.U); e = 0.0
    for k in 1:P.n[3]+2g, j in 1:P.n[2]+2g, i in 1:P.n[1]+2g
        ((i <= g) | (i > g+P.n[1])) || continue
        x = (P.lo[1] + i - g - 1 - 0.5) * P.dx
        e = max(e, abs(U[i,j,k,7] - A*sin(2π*(x - t))))
    end
    e
end
function evolve_measure!(H, l, t_stop, errs)
    P = H.patches[l]; L = length(H.patches)
    while H.t[l] < t_stop - 1e-12
        NM.fill_ghosts!(H, l)
        l > 1 && push!(errs, ghost_err(H, l, H.t[l]))
        dtc = NM.cfl_dt(H, P); rem = t_stop - H.t[l]
        dt = rem / max(1, ceil(Int, rem/dtc - 1e-9))
        if l < L; NM.snapshot!(H, P); H.tp0[l] = H.t[l]; H.tp1[l] = H.t[l] + dt; end
        NM.step_patch!(H, P, dt, dtc); l > 1 && NM.accumulate!(H, P, dt); NM._sync(H)
        if l < L
            C = H.patches[l+1]; NM.zero_accumulators!(C)
            evolve_measure!(H, l+1, H.t[l] + dt, errs)
            NM.sync_from_child!(H, P, C, dt); NM._sync(H)
        end
        H.t[l] += dt; H.nsteps[l] += 1
    end
    H.t[l] = t_stop
end
function exp_ghost()
    rows = []
    for mode in (:glm, :ct), courant in (0.4, 0.8), ti in (true, false), N in (16, 32, 64, 128)
        H = cp_hier(N, mode, ti; courant=courant); init!(H, cp_ic; vecpot=cp_A)
        errs = Float64[]; evolve_measure!(H, 1, 0.25, errs)
        push!(rows, Dict("mode"=>mode, "courant"=>courant, "ghosts"=>ti ? "interp" : "stale", "N"=>N, "err"=>maximum(errs)))
        @printf("ghost %s c=%.1f %s N=%d  %.3e\n", mode, courant, ti ? "interp" : "stale", N, maximum(errs))
    end
    save("ghost", rows)
end

# ---------------------------------------------------------------- 2. global convergence (analytic)
function l1_by(H, l)
    P = H.patches[l]; g = 2; U = Array(P.U); e = 0.0; c = 0
    for k in g+1:g+P.n[3], j in g+1:g+P.n[2], i in g+1:g+P.n[1]
        x = (P.lo[1] + i - g - 1 - 0.5) * P.dx; e += abs(U[i,j,k,7] - A*sin(2π*x)); c += 1
    end
    e / c
end
function exp_convergence()
    rows = []
    for mode in (:glm, :ct), courant in (0.4, 0.8), kind in ("interp", "stale", "uniform"), N in (16, 32, 64, 128)
        H = kind == "uniform" ? Hierarchy(N0=(N,8,8), mode=mode, T=Float64, courant=courant) : cp_hier(N, mode, kind == "interp"; courant=courant)
        init!(H, cp_ic; vecpot=cp_A); m0 = totals(H).mass
        tw = @elapsed evolve!(H, 1.0)
        r = Dict("mode"=>mode, "courant"=>courant, "kind"=>kind, "N"=>N, "l1_root"=>l1_by(H, 1),
                 "l1_child"=>kind == "uniform" ? nothing : l1_by(H, 2), "steps"=>copy(H.nsteps), "wall"=>tw,
                 "dmass"=>abs(totals(H).mass - m0)/m0, "divb"=>maximum(max_divb(H)))
        push!(rows, r); @printf("conv %s c=%.1f %-7s N=%3d  root %.3e  (%.1fs)\n", mode, courant, kind, N, r["l1_root"], tw)
    end
    save("convergence", rows)
end

# ---------------------------------------------------------------- 3. adaptive vs fixed ratio-2 subcycling
function cold_core(x, y, z)
    r2 = (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2
    ρ = 1 + 99*exp(-r2/0.15^2)                         # pressure equilibrium: c_s,core ≈ c_s,out/10
    (ρ, 0.5, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0)
end
cold_A(x, y, z) = (0.0, 0.0, 0.1*y)
function exp_adaptive()
    rows = []
    for (prob, ic, vp, tend) in (("orszag-tang", ot3, aot3, 0.05), ("cold-core", cold_core, cold_A, 0.05))
        for mode in (:ct, :glm), adaptive in (true, false)
            H = nested_central(16, 5; n=16, mode=mode, T=Float64, adaptive=adaptive)
            init!(H, ic; vecpot=vp); t0 = totals(H)
            tw = @elapsed evolve!(H, tend)
            cells = [prod(P.n) for P in H.patches]
            push!(rows, Dict("problem"=>prob, "mode"=>mode, "adaptive"=>adaptive, "steps"=>copy(H.nsteps),
                             "wall_level"=>copy(H.wall), "wall"=>tw, "cells"=>cells,
                             "cell_updates"=>sum(cells .* H.nsteps), "dmass"=>abs(totals(H).mass - t0.mass)/t0.mass,
                             "divb"=>maximum(max_divb(H))))
            @printf("adaptive %s %s %s steps=%s wall=%.1f\n", prob, mode, adaptive, string(H.nsteps), tw)
        end
    end
    save("adaptive", rows)
end

# ---------------------------------------------------------------- 4. covered-region skipping
function exp_skip()
    rows = []
    for mode in (:ct, :glm), skip in (true, false)
        H = nested_central(32, 3; n=32, mode=mode, T=Float32, skip_covered=skip)
        init!(H, ot3; vecpot=aot3); evolve!(H, 0.002)                 # warm-up / compile
        H = nested_central(32, 3; n=32, mode=mode, T=Float32, skip_covered=skip)
        init!(H, ot3; vecpot=aot3)
        tw = @elapsed evolve!(H, 0.03)
        push!(rows, Dict("mode"=>mode, "skip"=>skip, "steps"=>copy(H.nsteps), "wall"=>tw, "wall_level"=>copy(H.wall)))
        @printf("skip %s %s steps=%s wall=%.2f\n", mode, skip, string(H.nsteps), tw)
    end
    save("skip", rows)
end

# ---------------------------------------------------------------- 5. nested Orszag-Tang: fields + diagnostics
function zslice(H, l, v)
    P = H.patches[l]; g = 2; U = Array(P.U); k = g + P.n[3] ÷ 2 + 1
    if v == "rho"
        return Float32.(U[g+1:g+P.n[1], g+1:g+P.n[2], k, 1])
    elseif v == "pmag"
        return Float32.(0.5 .* (U[g+1:g+P.n[1], g+1:g+P.n[2], k, 6].^2 .+ U[g+1:g+P.n[1], g+1:g+P.n[2], k, 7].^2 .+
                                U[g+1:g+P.n[1], g+1:g+P.n[2], k, 8].^2))
    else # vorticity-like: |v|
        ρ = U[g+1:g+P.n[1], g+1:g+P.n[2], k, 1]
        return Float32.(sqrt.(U[g+1:g+P.n[1], g+1:g+P.n[2], k, 2].^2 .+ U[g+1:g+P.n[1], g+1:g+P.n[2], k, 3].^2) ./ ρ)
    end
end
function exp_otnested()
    out = Dict()
    for mode in (:ct, :glm)
        H = nested_central(32, 3; n=32, mode=mode, T=Float64)
        init!(H, ot3; vecpot=aot3); t0 = totals(H)
        frames = []; series = []
        for (it, t) in enumerate(0.0:0.05:0.5)
            t > 0 && evolve!(H, t)
            tt = totals(H)
            push!(series, Dict("t"=>t, "dmass"=>(tt.mass - t0.mass)/t0.mass, "denergy"=>(tt.energy - t0.energy)/t0.energy,
                               "divb"=>max_divb(H), "steps"=>copy(H.nsteps)))
            push!(frames, Dict("t"=>t, "levels"=>[Dict("lo"=>P.lo, "n"=>P.n, "dx"=>P.dx,
                                                       "rho"=>zslice(H, l, "rho"), "pmag"=>zslice(H, l, "pmag"),
                                                       "speed"=>zslice(H, l, "speed")) for (l, P) in enumerate(H.patches)]))
            @printf("otnested %s t=%.2f steps=%s divB=%s\n", mode, t, string(H.nsteps), string(round.(max_divb(H), sigdigits=2)))
        end
        out[string(mode)] = Dict("frames"=>frames, "series"=>series)
    end
    save("otnested", out)
end

# ---------------------------------------------------------------- 6. CPU throughput (single periodic patch)
function exp_throughput()
    rows = []
    for mode in (:glm, :ct), T in (Float32, Float64), riem in (:hll, :hlld)
        N = 48
        H = Hierarchy(N0=N, mode=mode, T=T, riemann=riem); init!(H, ot3; vecpot=aot3); evolve!(H, 0.003)
        H = Hierarchy(N0=N, mode=mode, T=T, riemann=riem); init!(H, ot3; vecpot=aot3)
        tw = @elapsed evolve!(H, 0.03)
        mcs = N^3 * H.nsteps[1] / tw / 1e6
        push!(rows, Dict("mode"=>mode, "T"=>string(T), "riemann"=>riem, "N"=>N, "steps"=>H.nsteps[1], "wall"=>tw, "mcells"=>mcs))
        @printf("throughput %s %s %s: %.3f Mcell/s\n", mode, T, riem, mcs)
    end
    save("throughput", rows)
end

# ---------------------------------------------------------------- 7. Orszag-Tang reference (GLMMHDTurb defaults)
function exp_otref()
    rows = []
    for (mode, riem) in ((:glm, :hlld), (:glm, :hll), (:ct, :hlld), (:ct, :hll)), N in (64, 128)
        H = Hierarchy(N0=(N, N, 4), mode=mode, T=Float64, riemann=riem, courant=0.7, glm_ch_scale=0.25)
        init!(H, ot2; vecpot=aot2); tw = @elapsed evolve!(H, 0.2)
        U = Array(H.patches[1].U)[3:N+2, 3:N+2, 3:6, :]
        vrms = sqrt(sum(@. (U[:,:,:,2]^2 + U[:,:,:,3]^2 + U[:,:,:,4]^2) / U[:,:,:,1]) / sum(U[:,:,:,1]))
        push!(rows, Dict("mode"=>mode, "riemann"=>riem, "N"=>N, "vrms"=>vrms, "steps"=>H.nsteps[1], "divb"=>max_divb(H)[1], "wall"=>tw))
        @printf("otref %s %s N=%d vrms=%.4f\n", mode, riem, N, vrms)
    end
    save("otref", rows)
end

const ALL = Dict("ghost"=>exp_ghost, "convergence"=>exp_convergence, "adaptive"=>exp_adaptive, "skip"=>exp_skip,
                 "otnested"=>exp_otnested, "throughput"=>exp_throughput, "otref"=>exp_otref)
for name in (isempty(ARGS) ? ["otref", "ghost", "adaptive", "skip", "throughput", "otnested", "convergence"] : ARGS)
    ALL[name]()
end
