# probe.jl — time series of one cell, streamed to a self-describing binary file.
#
# File layout (little-endian):
#   header   8 bytes  magic "NTPROBE1"
#            UInt32   format version (1)
#            UInt32   nvars (number of variables after t in each record)
#            nvars × 8 bytes  ASCII variable names, NUL-padded
#            Int32    level;   Int32 every (record every `every`-th step of that level)
#            3 × Float64 cell-centre position (x, y, z);  Float64 dx
#            UInt32   length of the JSON metadata string;  that many bytes of UTF-8 JSON
#   records  (1 + nvars) × Float64 each: t, var_1 … var_nvars
# The record count is (filesize - header) / (8(1+nvars)); a file cut short by a crash loses at most
# the unflushed buffer, never the header.

const PROBE_MAGIC = b"NTPROBE1"
const PROBE_VARS = ("rho", "vx", "vy", "vz", "Bx", "By", "Bz", "p")

"""
    Probe(H, path; level=length(H.patches), cell=nothing, every=2, flush_every=512, meta="")

Record the primitive state of one interior cell of `level` every `every`-th step of that level and
append it to `path` in blocks of `flush_every` records. Default cell: the one whose lower corner is
the box centre (one of the 2^dim most central cells). `meta` is a JSON string stored in the header.
"""
mutable struct Probe
    path::String
    level::Int
    cell::NTuple{3,Int}      # patch-local index (ghosts included)
    every::Int
    flush_every::Int
    buf::Vector{Float64}
    nrec::Int
    counter::Int
    cs2::Float64             # >0: isothermal pressure ρcs², else from E
end

function Probe(H, path::AbstractString; level::Int=length(H.patches), cell=nothing, every::Int=2,
               flush_every::Int=512, meta::AbstractString="", cs=nothing)
    P = H.patches[level]
    if cell === nothing
        gidx = ntuple(d -> P.Nlev[d] == 1 ? 1 : P.Nlev[d] ÷ 2 + 1, 3)        # global index at this level
        cell = ntuple(d -> gidx[d] - P.lo[d] + G + 1, 3)
    end
    all(d -> G < cell[d] <= G + P.n[d], 1:3) || error("probe cell $cell is not inside level $level")
    x = ntuple(d -> _cell_x(P, cell[d], d), 3)
    open(path, "w") do io
        write(io, PROBE_MAGIC, UInt32(1), UInt32(length(PROBE_VARS)))
        for v in PROBE_VARS
            b = zeros(UInt8, 8); cu = codeunits(v); b[1:length(cu)] .= cu; write(io, b)
        end
        write(io, Int32(level), Int32(every), Float64(x[1]), Float64(x[2]), Float64(x[3]), Float64(P.dx))
        mb = codeunits(String(meta)); write(io, UInt32(length(mb)), mb)
    end
    Probe(String(path), level, Tuple(cell), every, flush_every, Float64[], 0, 0,
          cs === nothing ? 0.0 : Float64(cs)^2)
end

function after_step!(pr::Probe, H, l)
    l == pr.level || return
    pr.counter += 1
    pr.counter % pr.every == 0 || return
    P = H.patches[l]; i, j, k = pr.cell
    u = Array(@view P.U[i, j, k, 1:8])                 # one cell: a tiny device→host copy
    ρ = Float64(u[1]); vx, vy, vz = u[2]/ρ, u[3]/ρ, u[4]/ρ
    bx, by, bz = Float64(u[6]), Float64(u[7]), Float64(u[8])
    p = pr.cs2 > 0 ? ρ * pr.cs2 :
        (Float64(H.p.gamma) - 1) * (u[5] - 0.5ρ*(vx^2 + vy^2 + vz^2) - 0.5*(bx^2 + by^2 + bz^2))
    append!(pr.buf, (H.t[l], ρ, vx, vy, vz, bx, by, bz, p))
    pr.nrec += 1
    length(pr.buf) >= pr.flush_every * (1 + length(PROBE_VARS)) && flush!(pr)
end

function flush!(pr::Probe)
    isempty(pr.buf) && return
    open(pr.path, "a") do io; write(io, pr.buf); end
    empty!(pr.buf)
end
finalize!(pr::Probe, H) = flush!(pr)

"""
    read_probe(path) -> (; vars, level, every, x, dx, meta, t, data)

`data` is an `nrec × nvars` matrix (columns in the order of `vars`), `t` the record times.
"""
function read_probe(path::AbstractString)
    open(path, "r") do io
        read(io, 8) == PROBE_MAGIC || error("$path is not an NTPROBE1 file")
        ver = read(io, UInt32); ver == 1 || error("unsupported probe format version $ver")
        nv = Int(read(io, UInt32))
        vars = [rstrip(String(read(io, 8)), '\0') for _ in 1:nv]
        level = Int(read(io, Int32)); every = Int(read(io, Int32))
        x = (read(io, Float64), read(io, Float64), read(io, Float64)); dx = read(io, Float64)
        meta = String(read(io, Int(read(io, UInt32))))
        raw = reinterpret(Float64, read(io))
        nr = length(raw) ÷ (1 + nv)
        M = permutedims(reshape(raw[1:nr*(1+nv)], 1 + nv, nr))
        (; vars, level, every, x, dx, meta, t = M[:, 1], data = M[:, 2:end])
    end
end
