# Direct check of the time interpolation: L∞ error of the child's ghost zones against the exact
# travelling CP Alfvén wave, measured at every child ghost fill (i.e. at every child substep start).
# θ-interp ghosts should converge at 2nd order; stale (θ ≡ 0) ghosts at 1st order (lag = dt_f).
using NestedMHD, Printf
const NM = NestedMHD
const A = 0.1
cp_ic(x, y, z) = (1.0, 0.0, -A*sin(2π*x), -A*cos(2π*x), 0.1, 1.0, A*sin(2π*x), A*cos(2π*x))
cp_A(x, y, z)  = (0.0, A*sin(2π*x)/(2π), y + A*cos(2π*x)/(2π))
# exact: By(x,t) = A sin(2π(x - t))   (vA = Bx/√ρ = 1)

function ghost_err(H, l, t)
    P = H.patches[l]; g = 2; U = Array(P.U); e = 0.0
    for k in 1:P.n[3]+2g, j in 1:P.n[2]+2g, i in 1:P.n[1]+2g
        ghost = (i <= g) | (i > g+P.n[1])          # x-ghosts carry the time-dependent wave
        ghost || continue
        x = (P.lo[1] + i - g - 1 - 0.5) * P.dx
        e = max(e, abs(U[i,j,k,7] - A*sin(2π*(x - t))))
    end
    e
end

# evolve_level! with a measurement after every child ghost fill
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

for mode in (:ct, :glm), courant in (0.8,)
    println("== $mode, courant $courant: max ghost-zone error over t ∈ [0, 0.25] ==")
    for ti in (true, false)
        prev = NaN
        for N in (16, 32, 64, 128)
            H = Hierarchy(N0=(N,8,8), levels=[((N÷2+1,5,5),(N,8,8))], mode=mode, T=Float64, time_interp=ti, courant=courant)
            init!(H, cp_ic; vecpot=cp_A)
            errs = Float64[]; evolve_measure!(H, 1, 0.25, errs)
            e = maximum(errs)
            @printf("  N=%4d  %-9s  max|ghost - exact| = %.3e   order %5.2f\n", N, ti ? "θ-interp" : "stale", e, log2(prev/e))
            prev = e
        end
    end
end
