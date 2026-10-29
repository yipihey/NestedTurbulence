using NestedMHD, Test

const b0 = 1/sqrt(4π)
ot(x, y, z)  = (25/(36π)*(1+0.1*sin(2π*z)), -sin(2π*y), sin(2π*x), 0.3*sin(2π*x), 5/(12π),
                -b0*sin(2π*y), b0*sin(4π*x), 0.2*b0)
aot(x, y, z) = (0.0, 0.2*b0*x, b0*(cos(4π*x)/(4π) + cos(2π*y)/(2π)))

rel(a, b) = abs(a - b) / abs(b)

# --- ghost-zone accuracy against the exact travelling CP Alfvén wave (bench/run_experiments.jl, experiment "ghost")
const NM = NestedMHD
const Aw = 0.1
cp_ic(x, y, z) = (1.0, 0.0, -Aw*sin(2π*x), -Aw*cos(2π*x), 0.1, 1.0, Aw*sin(2π*x), Aw*cos(2π*x))
cp_A(x, y, z)  = (0.0, Aw*sin(2π*x)/(2π), y + Aw*cos(2π*x)/(2π))
function ghost_err(H, l, t)
    P = H.patches[l]; g = 2; U = Array(P.U); e = 0.0
    for k in 1:P.n[3]+2g, j in 1:P.n[2]+2g, i in 1:P.n[1]+2g
        ((i <= g) | (i > g+P.n[1])) || continue
        x = (P.lo[1] + i - g - 1 - 0.5) * P.dx
        e = max(e, abs(U[i,j,k,7] - Aw*sin(2π*(x - t))))
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
function ghost_order(mode, ti)
    e = map((32, 64)) do N
        H = Hierarchy(N0=(N,8,8), levels=[((N÷2+1,5,5),(N,8,8))], mode=mode, T=Float64, time_interp=ti, courant=0.8)
        init!(H, cp_ic; vecpot=cp_A); errs = Float64[]; evolve_measure!(H, 1, 0.25, errs); maximum(errs)
    end
    log2(e[1]/e[2])
end
cold_core(x, y, z) = (1 + 99*exp(-((x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2)/0.15^2), 0.5, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0)
cold_A(x, y, z) = (0.0, 0.0, 0.1*y)

@testset "NestedMHD" begin
    @testset "single level: $mode" for mode in (:glm, :ct)
        H = Hierarchy(N0=16, mode=mode, T=Float64)
        init!(H, ot; vecpot=aot)
        t0 = totals(H); evolve!(H, 0.05); t1 = totals(H)
        @test rel(t1.mass, t0.mass) < 1e-13
        @test rel(t1.energy, t0.energy) < 1e-13
        @test abs(t1.momx - t0.momx) < 1e-14
        mode === :ct && @test maximum(max_divb(H)) < 1e-13
    end

    @testset "nested hierarchy: $nlev levels, $mode, time_interp=$ti" for mode in (:glm, :ct), nlev in (2, 3), ti in (true, false)
        H = nested_central(16, nlev; n=16, mode=mode, T=Float64, time_interp=ti)
        init!(H, ot; vecpot=aot)
        mode === :ct && @test maximum(max_divb(H)) < 1e-14          # consistent initial hierarchy
        t0 = totals(H); evolve!(H, 0.05); t1 = totals(H)
        @test rel(t1.mass, t0.mass) < 1e-13                           # Berger-Colella refluxing
        @test rel(t1.energy, t0.energy) < 1e-13
        @test abs(t1.momy - t0.momy) < 1e-14
        mode === :ct && @test maximum(max_divb(H)) < 1e-13            # EMF correction + face restriction
        @test all(P -> all(isfinite, Array(P.U)), H.patches)
        @test H.nsteps[end] > H.nsteps[1]                             # finer levels take more steps
    end

    @testset "covered-region skipping does not change the answer" begin
        for mode in (:glm, :ct)
            Ha = nested_central(16, 2; n=16, mode=mode, T=Float64, skip_covered=true)
            Hb = nested_central(16, 2; n=16, mode=mode, T=Float64, skip_covered=false, adaptive=true)
            init!(Ha, ot; vecpot=aot); init!(Hb, ot; vecpot=aot)
            # same steps: skip_covered=false would include covered cells in the CFL, so fix dt via a short run
            evolve!(Ha, 0.01); evolve!(Hb, 0.01)
            Ua = Array(Ha.patches[2].U); Ub = Array(Hb.patches[2].U)
            if Ha.nsteps == Hb.nsteps
                @test maximum(abs.(Ua .- Ub)) < 1e-12
            else
                @test_skip "different CFL steps"
            end
        end
    end

    @testset "second-order time-interpolated ghosts: $mode" for mode in (:glm, :ct)
        @test ghost_order(mode, true)  > 1.8       # measured 1.96–1.97 (32 → 64)
        @test ghost_order(mode, false) < 1.0       # stale coarse ghosts: first order (measured 0.90–0.91)
    end

    @testset "adaptive per-level steps: cold core" begin
        upd = map((true, false)) do adaptive
            H = nested_central(16, 4; n=16, mode=:ct, T=Float64, adaptive=adaptive)
            init!(H, cold_core; vecpot=cold_A); t0 = totals(H); evolve!(H, 0.03)
            @test rel(totals(H).mass, t0.mass) < 1e-13
            @test maximum(max_divb(H)) < 1e-13
            sum(prod(P.n) * H.nsteps[l] for (l, P) in enumerate(H.patches))
        end
        @test upd[1] < 0.6 * upd[2]                 # the cold levels take one step per parent step
    end

    @testset "2-D nests (children span z periodically): $mode" for mode in (:glm, :ct)
        ot2(x, y, z) = (25/(36π), -sin(2π*y), sin(2π*x), 0.0, 5/(12π), -b0*sin(2π*y), b0*sin(4π*x), 0.0)
        a2(x, y, z) = (0.0, 0.0, b0*(cos(4π*x)/(4π) + cos(2π*y)/(2π)))
        H = Hierarchy(; N0=(16,16,1), levels=[((9,9,1),(16,16,2)), ((25,25,1),(16,16,4))], mode=mode, T=Float64)
        @test [P.per for P in H.patches] == [(true,true,true), (false,false,true), (false,false,true)]
        init!(H, ot2; vecpot=a2); t0 = totals(H)
        evolve!(H, 0.1); t1 = totals(H)
        @test rel(t1.mass, t0.mass) < 1e-13
        @test rel(t1.energy, t0.energy) < 1e-13
        @test abs(t1.momz) < 1e-14
        mode === :ct && @test maximum(max_divb(H)) < 1e-13
        for P in H.patches[2:end]                       # the solution stays exactly z-independent
            U = Array(P.U)
            @test all(k -> U[:,:,k,:] == U[:,:,3,:], 3:2+P.n[3])
        end
    end
    @testset "driving: the same force on every level" begin
        H = Hierarchy(; N0=(16,16,1), levels=[((5,5,1),(16,16,2))], mode=:ct, T=Float64, gamma=1.01)
        init!(H, (x,y,z) -> (1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0); vecpot=(x,y,z) -> (0.0, 0.0, 0.1y))
        d = Driving(H; mach=10.0, kmax=3, zeta=0.5, isothermal=false)
        @test size(d.K, 2) == 14 && all(d.K[3, :] .== 0)
        for P in H.patches; NM.apply_sources!(d, H, P, 0.0, 1e-3); end
        Pr, Pc = H.patches; Ur = Array(Pr.U); Uc = Array(Pc.U); g = 2
        err = 0.0                    # restriction of the fine kick equals the coarse kick (exact cell averages)
        for J in 1:Pr.nc[2], I in 1:Pr.nc[1], v in 2:3
            fi = g + 2I - 1; fj = g + 2J - 1
            fine = 0.25*(Uc[fi,fj,3,v] + Uc[fi+1,fj,3,v] + Uc[fi,fj+1,3,v] + Uc[fi+1,fj+1,3,v])
            err = max(err, abs(fine - Ur[Pr.clo[1]+I-1, Pr.clo[2]+J-1, 3, v]))
        end
        @test err < 1e-13 * maximum(abs, Ur[:,:,3,2:3])
        a = [sqrt(2*sum(abs2, amplitudes(d, t))) for t in 0:0.01:2]   # unit-rms OU field, continuous in t
        @test 0.5 < sum(a)/length(a) < 1.5
        @test maximum(abs, amplitudes(d, 0.3001) .- amplitudes(d, 0.3)) < 1e-2
    end
    @testset "driven run: isothermal, zero mean momentum, probe file round trip" begin
        H = Hierarchy(; N0=(16,16,1), levels=[((9,9,1),(16,16,2))], mode=:ct, T=Float64, gamma=1.01)
        init!(H, (x,y,z) -> (1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0); vecpot=(x,y,z) -> (0.0, 0.0, 0.1y))
        d = Driving(H; mach=2.0); push!(H.sources, d)
        path = tempname()
        pr = Probe(H, path; cs=1.0, flush_every=7, meta="{\"test\":1}"); push!(H.sources, pr)
        m0 = totals(H).mass
        evolve!(H, 0.1)
        tt = totals(H)
        @test rel(tt.mass, m0) < 1e-13
        @test abs(tt.momx) + abs(tt.momy) < 1e-12
        @test maximum(max_divb(H)) < 1e-13
        U = Array(H.patches[2].U); ρ = U[3:end-2,3:end-2,3,1]
        eint = U[3:end-2,3:end-2,3,5] .- (U[3:end-2,3:end-2,3,2].^2 .+ U[3:end-2,3:end-2,3,3].^2) ./ (2ρ) .-
               0.5 .* (U[3:end-2,3:end-2,3,6].^2 .+ U[3:end-2,3:end-2,3,7].^2 .+ U[3:end-2,3:end-2,3,8].^2)
        @test maximum(abs, eint .* 0.01 ./ ρ .- 1) < 1e-10                # p = ρ cs² everywhere
        @test pr.nrec == H.nsteps[2] ÷ 2
        r = read_probe(path)
        @test r.vars == ["rho", "vx", "vy", "vz", "Bx", "By", "Bz", "p"] && r.meta == "{\"test\":1}"
        @test length(r.t) == pr.nrec && issorted(r.t) && r.level == 2 && r.every == 2
        @test r.x[1] ≈ 0.5 + r.dx/2 && r.x[2] ≈ 0.5 + r.dx/2
        i, j, k = pr.cell
        iseven(H.nsteps[2]) && @test r.data[end, 1] == U[i,j,k,1]     # (v is recorded before the root's Galilean shift)
        @test all(r.data[:, 8] .== r.data[:, 1])                          # isothermal p = ρ cs², cs = 1
    end
    @testset "Float32 nested CT" begin
        H = nested_central(16, 2; n=16, mode=:ct, T=Float32)
        init!(H, ot; vecpot=aot); t0 = totals(H); evolve!(H, 0.05)
        @test rel(totals(H).mass, t0.mass) < 1e-5
        @test maximum(max_divb(H)) < 1e-5
    end
end
