# run_mach10.jl — the driven Mach-10 2-D run of the "Ergodicity" chapter.
#
#   julia -t auto --project=core core/bench/run_mach10.jl [N=64] [t_end=1.0] [seed=1] [tag=mach10]
#
# Root N² (periodic), two refined levels of N² cells, each the central half of its parent (finest:
# 4N² effective resolution in the central quarter of the box). Isothermal (cs = 1), ρ0 = 1, uniform
# B_x with β = 2p/B² = 200, CT, HLL. Driven by 14 solenoidal modes 1 ≤ |k| ≤ 3 with Mach-10 feedback.
# t_c = L/(M cs) = 0.1, so t_end = 1 is ten large-scale crossing times.
#
# writes app/data/<tag>_probe.bin (central-cell time series, NTPROBE1 format, see core/src/probe.jl)
#    and app/data/<tag>.json (Mach history, snapshots of log10 ρ on all levels, volume PDFs).

using NestedMHD, Printf
const OUT = joinpath(@__DIR__, "..", "..", "app", "data")
mkpath(OUT)
include(joinpath(@__DIR__, "jsonio.jl"))

N    = parse(Int, get(ARGS, 1, "64"))
tend = parse(Float64, get(ARGS, 2, "1.0"))
seed = parse(Int, get(ARGS, 3, "1"))
tag  = get(ARGS, 4, "mach10")

const MACH = 10.0; const CS = 1.0; const BETA = 200.0
B0 = sqrt(2 * CS^2 / BETA)
lv = [((N÷2+1, N÷2+1, 1), (N, N, 2)), ((3N÷2+1, 3N÷2+1, 1), (N, N, 4))]
H = Hierarchy(; N0=(N, N, 1), levels=lv, mode=:ct, riemann=:hll, T=Float64, gamma=1.01, courant=0.4)
init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, CS^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0 * y))
drv = Driving(H; mach=MACH, cs=CS, kmin=1, kmax=3, zeta=1.0, seed=seed)
push!(H.sources, drv)
probe_path = joinpath(OUT, tag * "_probe.bin")
meta = "{\"problem\":\"mach10-2d\",\"N\":$N,\"levels\":3,\"mach\":$MACH,\"cs\":$CS,\"beta\":$BETA,\"seed\":$seed,\"t_cross\":$(1/MACH)}"
pr = Probe(H, probe_path; every=2, cs=CS, meta=meta)
push!(H.sources, pr)
println("modes: ", size(drv.K, 2), "  probe at ", NestedMHD._cell_x(H.patches[end], pr.cell[1], 1), ", ",
        NestedMHD._cell_x(H.patches[end], pr.cell[2], 2))

# ------------------------------------------------------------------ analysis helpers
const SB = range(-12.0, 6.0; length=91)          # ln ρ bin edges
const VB = range(-60.0, 60.0; length=121)         # v_x bin edges
function hist(x, edges)
    h = zeros(Int, length(edges) - 1); a = first(edges); w = step(edges)
    for v in x
        b = floor(Int, (v - a) / w) + 1
        1 <= b <= length(h) && (h[b] += 1)
    end
    h
end
interior(P) = (G = 2; (G+1:G+P.n[1], G+1:G+P.n[2], G+1))
function level_fields(P)
    U = Array(P.U); r = interior(P)
    ρ = U[r..., 1]; vx = U[r..., 2] ./ ρ; vy = U[r..., 3] ./ ρ
    ρ, vx, vy
end
r3(x) = round(x; digits=3)

snaps = Any[]; frames_t = Float64[]
pdf_t = Float64[]; pdf_fine_s = Vector{Int}[]; pdf_root_s = Vector{Int}[]; pdf_fine_v = Vector{Int}[]; pdf_root_v = Vector{Int}[]
function record!(t)
    lev = Any[]
    for P in H.patches
        ρ, vx, vy = level_fields(P)
        push!(lev, Dict("lo" => P.lo[1:2], "n" => P.n[1:2], "Nlev" => P.Nlev[1:2],
                        "logrho" => vec(r3.(log10.(ρ)))))          # column-major (x fastest)
    end
    push!(snaps, lev); push!(frames_t, t)
end
function pdfs!(t)
    ρf, vxf, _ = level_fields(H.patches[end]); ρr, vxr, _ = level_fields(H.patches[1])
    push!(pdf_t, t)
    push!(pdf_fine_s, hist(log.(ρf), SB)); push!(pdf_root_s, hist(log.(ρr), SB))
    push!(pdf_fine_v, hist(vxf, VB)); push!(pdf_root_v, hist(vxr, VB))
end

# ------------------------------------------------------------------ run
dt_out = 0.05; dt_pdf = 0.01
record!(0.0); pdfs!(0.0)
tw = time(); t = 0.0
while t < tend - 1e-12
    global t = min(t + dt_pdf, tend)
    evolve!(H, t)
    get(ENV, "NT_DEBUG", "") == "1" && (@printf("  t=%.3f steps=%s\n", t, H.nsteps); flush(stdout))
    pdfs!(t)
    abs(t / dt_out - round(t / dt_out)) < 1e-6 && record!(t)
    begin
        @printf("t=%.3f  M_m=%.2f  M_v=%.2f  amp=%.1f  steps=%s  max divB=%.1e  probe=%d  wall=%.0fs\n",
                t, drv.log_mach_m[end], drv.log_mach[end], drv.amp, H.nsteps, maximum(max_divb(H)), pr.nrec, time() - tw)
        flush(stdout)
    end
end
wall = time() - tw

# Mach history thinned to ≤ 4000 points
nl = length(drv.log_t); st = max(1, nl ÷ 4000); ix = 1:st:nl
save(OUT, tag, Dict(
    "params" => Dict("N" => N, "levels" => 3, "mach" => MACH, "cs" => CS, "beta" => BETA, "gamma" => 1.01,
                     "t_end" => tend, "t_cross" => 1 / MACH, "seed" => seed, "modes" => size(drv.K, 2),
                     "kmin" => 1, "kmax" => 3, "zeta" => 1.0, "t_corr" => drv.t_corr, "t_feedback" => drv.t_feedback,
                     "riemann" => "hll", "mode" => "ct", "courant" => 0.4,
                     "probe_x" => collect(NestedMHD._cell_x.(Ref(H.patches[end]), pr.cell, (1, 2, 3))),
                     "probe_dx" => H.patches[end].dx, "probe_every" => pr.every, "probe_file" => tag * "_probe.bin"),
    "steps" => H.nsteps, "wall" => wall, "wall_levels" => H.wall, "divb" => max_divb(H),
    "mach" => Dict("t" => drv.log_t[ix], "vol" => drv.log_mach[ix], "mass" => drv.log_mach_m[ix], "amp" => drv.log_amp[ix]),
    "frames" => Dict("t" => frames_t, "levels" => snaps),
    "pdf" => Dict("s_edges" => collect(SB), "v_edges" => collect(VB), "t" => pdf_t,
                  "fine_s" => pdf_fine_s, "root_s" => pdf_root_s, "fine_v" => pdf_fine_v, "root_v" => pdf_root_v,
                  "fine_cells" => prod(H.patches[end].n[1:2]), "root_cells" => prod(H.patches[1].n[1:2]))))
@printf("done: t=%.3f, %d probe records, wall %.0f s\n", tend, pr.nrec, wall)
