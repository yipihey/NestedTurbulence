# combine_ensemble.jl — the ensemble of driven Mach-10 runs (different driving seeds), condensed for the lab.
#
#   for s in 1 2 3 4 5 6 7 8; do julia -t auto --project=core core/bench/run_mach10.jl 32 1.0 $s ens32_s$s; done
#   julia --project=core core/bench/combine_ensemble.jl 32 8
#
# reads app/data/ens<N>_s<seed>_probe.bin (+ .json) and writes app/data/mach10_ensemble.json: every member's
# probe resampled on a common time grid (s = ln ρ and v_x), its Mach history, and its volume PDFs of s
# averaged over t ≥ 3 t_c. The per-member files are deleted afterwards (they are not needed by the site).

using NestedMHD, Printf
const OUT = joinpath(@__DIR__, "..", "..", "app", "data")
include(joinpath(@__DIR__, "jsonio.jl"))

N = parse(Int, get(ARGS, 1, "32")); M = parse(Int, get(ARGS, 2, "8"))
keep = get(ARGS, 3, "") == "keep"
const DT = 0.002
tg = collect(0.0:DT:1.0)

# linear interpolation of (t, y) onto tg (NaN outside the record)
function resample(t, y)
    out = fill(NaN, length(tg)); j = 1
    for (i, τ) in enumerate(tg)
        (τ < t[1] || τ > t[end]) && continue
        while j < length(t) - 1 && t[j+1] < τ; j += 1; end
        w = (τ - t[j]) / (t[j+1] - t[j]); out[i] = y[j] + w * (y[j+1] - y[j])
    end
    out
end
r4(x) = isnan(x) ? nothing : round(x; digits=4)

# the volume PDFs are read back from each member's JSON (a minimal parse of the arrays we need)
function jarray(txt, key; from=1)
    i = findnext("\"$key\":[", txt, from); i === nothing && error("no $key")
    k = last(i); depth = 0; j = k
    while true
        c = txt[j]; c == '[' && (depth += 1); c == ']' && (depth -= 1)
        depth == 0 && break; j = nextind(txt, j)
    end
    eval(Meta.parse(txt[k:j]))
end

members = Any[]
for s in 1:M
    tag = "ens$(N)_s$s"
    r = read_probe(joinpath(OUT, tag * "_probe.bin"))
    txt = read(joinpath(OUT, tag * ".json"), String)
    p0 = first(findfirst("\"pdf\":{", txt))
    pt = Float64.(jarray(txt, "t"; from=p0)); fine = jarray(txt, "fine_s"; from=p0); root = jarray(txt, "root_s"; from=p0)
    tc = 0.1
    sel = findall(pt .>= 3tc - 1e-9)
    fs = sum(Float64.(fine[k]) for k in sel); rs = sum(Float64.(root[k]) for k in sel)
    push!(members, Dict("seed" => s, "s" => r4.(resample(r.t, log.(r.data[:, 1]))), "vx" => r4.(resample(r.t, r.data[:, 2])),
                        "records" => length(r.t), "fine_s" => fs, "root_s" => rs))
    @printf("member %d: %d probe records\n", s, length(r.t))
end
save(OUT, "mach10_ensemble", Dict("N" => N, "members" => M, "dt" => DT, "t" => tg,
                                 "s_edges" => collect(range(-12.0, 6.0; length=91)), "runs" => members))
if !keep
    for s in 1:M, ext in ("_probe.bin", ".json")
        rm(joinpath(OUT, "ens$(N)_s$s" * ext); force=true)
    end
end
