# Circularly polarised Alfvén wave (exact nonlinear solution) along x through a nested patch.
# Quasi-1D box (N, 8, 8); the child covers the central half in x and is entered/left by the wave
# twice per period. L1 error of By after one period, on the root (covered cells = restricted child)
# and on the child. Compares 2nd-order time-interpolated ghosts with stale (θ ≡ 0) ghosts.
using NestedMHD, Printf

const A = 0.1
cp_ic(x, y, z) = (1.0, 0.0, -A*sin(2π*x), -A*cos(2π*x), 0.1, 1.0, A*sin(2π*x), A*cos(2π*x))
cp_A(x, y, z)  = (0.0, A*sin(2π*x)/(2π), y + A*cos(2π*x)/(2π))     # curl = (1, A sin, A cos)

function l1_by(H, l)
    P = H.patches[l]; g = 2
    U = Array(P.U); err = 0.0; cnt = 0
    for k in g+1:g+P.n[3], j in g+1:g+P.n[2], i in g+1:g+P.n[1]
        x = (P.lo[1] + i - g - 1 - 0.5) * P.dx
        err += abs(U[i,j,k,7] - A*sin(2π*x)); cnt += 1
    end
    err / cnt
end

function run(N, mode, ti; nested=true, riemann=:hll)
    levels = nested ? [((N÷2+1, 5, 5), (N, 8, 8))] : Tuple[]
    H = Hierarchy(N0=(N, 8, 8), levels=levels, mode=mode, T=Float64, riemann=riemann, time_interp=ti,
                  courant=0.4, gamma=5/3)
    init!(H, cp_ic; vecpot=cp_A)
    m0 = totals(H).mass
    tw = @elapsed evolve!(H, 1.0)
    e1 = l1_by(H, 1); e2 = nested ? l1_by(H, 2) : NaN
    (e1=e1, e2=e2, steps=copy(H.nsteps), wall=tw, dm=(totals(H).mass-m0)/m0, divb=maximum(max_divb(H)))
end

for mode in (:glm, :ct)
    println("== $mode ==")
    @printf("  %4s  %-10s %11s %11s %6s %6s  %s\n", "N", "ghosts", "L1 root", "L1 child", "p_root", "p_chld", "steps   |Δm/m| max(∇·B)")
    for ti in (true, false)
        prev = nothing
        for N in (16, 32, 64)
            r = run(N, mode, ti)
            pr = prev === nothing ? NaN : log2(prev.e1/r.e1); pc = prev === nothing ? NaN : log2(prev.e2/r.e2)
            @printf("  %4d  %-10s %11.4e %11.4e %6.2f %6.2f  %s %.1e %.1e  (%.0fs)\n", N, ti ? "θ-interp" : "stale",
                    r.e1, r.e2, pr, pc, string(r.steps), abs(r.dm), r.divb, r.wall)
            prev = r
        end
    end
    prev = nothing
    for N in (16, 32, 64)
        r = run(N, mode, true; nested=false)
        pr = prev === nothing ? NaN : log2(prev.e1/r.e1)
        @printf("  %4d  %-10s %11.4e %11s %6.2f %6s  %s\n", N, "uniform", r.e1, "—", pr, "", string(r.steps))
        prev = r
    end
end
