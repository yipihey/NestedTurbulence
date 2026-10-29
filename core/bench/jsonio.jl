# jsonio.jl — the minimal JSON writer shared by the long-run scripts (same format as run_experiments.jl)
using Printf
js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
js(x::Symbol) = js(String(x))
js(x::Bool) = x ? "true" : "false"
js(x::Integer) = string(x)
js(x::Real) = isfinite(x) ? @sprintf("%.6g", x) : "null"
js(x::Nothing) = "null"
js(x::AbstractVector) = "[" * join((js(v) for v in x), ",") * "]"
js(x::Tuple) = js(collect(x))
js(x::AbstractDict) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in x), ",") * "}"
function save(out, name, obj)
    meta = Dict("generated" => string(Base.Libc.strftime("%Y-%m-%d %H:%M", time())), "threads" => Threads.nthreads(),
                "cpu" => Sys.cpu_info()[1].model, "julia" => string(VERSION))
    open(joinpath(out, name * ".json"), "w") do io
        write(io, js(Dict("meta" => meta, "data" => obj)))
    end
    println("wrote ", name, ".json")
end
