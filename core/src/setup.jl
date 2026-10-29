# setup.jl — initial conditions, consistent restriction, diagnostics.

_cell_x(P::Patch, idx, d) = (P.lo[d] + idx - G - 1 - 0.5) * P.dx      # cell centre
_face_x(P::Patch, idx, d) = (P.lo[d] + idx - G - 1 - 1.0) * P.dx      # lower face

"""
    init!(H, ic; vecpot=nothing)

`ic(x,y,z)` returns the primitive state `(ρ, vx, vy, vz, p, Bx, By, Bz)`.

CT: if `vecpot(x,y,z) -> (Ax,Ay,Az)` is given, face B is the discrete curl of A sampled on cell
edges, and the parent faces just outside each child footprint are made consistent with the
restricted (fine) boundary faces by the curl of δA = ⟨A_fine⟩ − A_coarse on the footprint's
surface edges. ∇·B is then zero to rounding in every cell of every level. Without `vecpot`, faces
are sampled at face centres; the parent cells bordering a footprint then carry an O(dx²)
truncation divergence (the scheme preserves it — it never grows — but it is not zero).

All levels are filled, then parents are restricted from their children (finest first), on the host.
"""
function init!(H::Hierarchy, ic; vecpot=nothing)
    T = H.T; γ = Float64(H.p.gamma); cpu = KA.CPU(); L = length(H.patches)
    hU = Vector{Array{T,4}}(undef, L)
    hb = Vector{NTuple{3,Array{T,3}}}(undef, L)
    for (l, P) in enumerate(H.patches)                        # 1. faces, every level
        nx, ny, nz = ng(P)
        ct(H) || continue
        bx = zeros(T, nx+1, ny, nz); by = zeros(T, nx, ny+1, nz); bz = zeros(T, nx, ny, nz+1)
        if vecpot === nothing
            for k in 1:nz, j in 1:ny, i in 1:nx+1
                bx[i,j,k] = ic(_face_x(P,i,1), _cell_x(P,j,2), _cell_x(P,k,3))[6]
            end
            for k in 1:nz, j in 1:ny+1, i in 1:nx
                by[i,j,k] = ic(_cell_x(P,i,1), _face_x(P,j,2), _cell_x(P,k,3))[7]
            end
            for k in 1:nz+1, j in 1:ny, i in 1:nx
                bz[i,j,k] = ic(_cell_x(P,i,1), _cell_x(P,j,2), _face_x(P,k,3))[8]
            end
        else
            dx = P.dx
            Ax(i,j,k) = vecpot(_cell_x(P,i,1), _face_x(P,j,2), _face_x(P,k,3))[1]   # x-edge (i, j-½, k-½)
            Ay(i,j,k) = vecpot(_face_x(P,i,1), _cell_x(P,j,2), _face_x(P,k,3))[2]
            Az(i,j,k) = vecpot(_face_x(P,i,1), _face_x(P,j,2), _cell_x(P,k,3))[3]
            for k in 1:nz, j in 1:ny, i in 1:nx+1
                bx[i,j,k] = ((Az(i,j+1,k) - Az(i,j,k)) - (Ay(i,j,k+1) - Ay(i,j,k))) / dx
            end
            for k in 1:nz, j in 1:ny+1, i in 1:nx
                by[i,j,k] = ((Ax(i,j,k+1) - Ax(i,j,k)) - (Az(i+1,j,k) - Az(i,j,k))) / dx
            end
            for k in 1:nz+1, j in 1:ny, i in 1:nx
                bz[i,j,k] = ((Ay(i+1,j,k) - Ay(i,j,k)) - (Ax(i,j+1,k) - Ax(i,j,k))) / dx
            end
        end
        hb[l] = (bx, by, bz)
    end
    if ct(H)                                                  # 2. consistent faces, finest first
        for l in L-1:-1:1
            P = H.patches[l]; C = H.patches[l+1]; nc = P.nc
            bx, by, bz = hb[l]
            if vecpot !== nothing
                nx, ny, nz = ng(P); q = P.dx/4
                exA = zeros(T, nx, ny, nz); eyA = zeros(T, nx, ny, nz); ezA = zeros(T, nx, ny, nz)
                for k in 1:nz, j in 1:ny, i in 1:nx
                    exA[i,j,k] = vecpot(_cell_x(P,i,1), _face_x(P,j,2), _face_x(P,k,3))[1]
                    eyA[i,j,k] = vecpot(_face_x(P,i,1), _cell_x(P,j,2), _face_x(P,k,3))[2]
                    ezA[i,j,k] = vecpot(_face_x(P,i,1), _face_x(P,j,2), _cell_x(P,k,3))[3]
                end
                # minus the fine sub-edge average on the footprint's edge box (only surface entries are used)
                EAx = zeros(T, nc[1], nc[2]+1, nc[3]+1); EAy = zeros(T, nc[1]+1, nc[2], nc[3]+1); EAz = zeros(T, nc[1]+1, nc[2]+1, nc[3])
                for K in axes(EAx,3), J in axes(EAx,2), I in axes(EAx,1)
                    i, j, k = P.clo .+ (I, J, K) .- 1; x, y, z = _cell_x(P,i,1), _face_x(P,j,2), _face_x(P,k,3)
                    EAx[I,J,K] = -0.5*(vecpot(x-q, y, z)[1] + vecpot(x+q, y, z)[1])
                end
                for K in axes(EAy,3), J in axes(EAy,2), I in axes(EAy,1)
                    i, j, k = P.clo .+ (I, J, K) .- 1; x, y, z = _face_x(P,i,1), _cell_x(P,j,2), _face_x(P,k,3)
                    EAy[I,J,K] = -0.5*(vecpot(x, y-q, z)[2] + vecpot(x, y+q, z)[2])
                end
                for K in axes(EAz,3), J in axes(EAz,2), I in axes(EAz,1)
                    i, j, k = P.clo .+ (I, J, K) .- 1; x, y, z = _face_x(P,i,1), _face_x(P,j,2), _cell_x(P,k,3)
                    EAz[I,J,K] = -0.5*(vecpot(x, y, z-q)[3] + vecpot(x, y, z+q)[3])
                end
                # emf_correct with dtc = -1:  B += curl(⟨A_f⟩ - A_c)/dx on the faces outside the footprint
                emf_correct_kernel!(cpu)(bx, by, bz, exA, eyA, ezA, EAx, EAy, EAz, P.clo, nc, -one(T), T(1/P.dx), P.cper;
                                         ndrange=emf_range(H.patches[l+1], nc))
            end
            cbx, cby, cbz = hb[l+1]
            restrict_faces_kernel!(cpu)(bx, cbx, Val(1), P.clo, G; ndrange=(nc[1]+1, nc[2], nc[3]))
            restrict_faces_kernel!(cpu)(by, cby, Val(2), P.clo, G; ndrange=(nc[1], nc[2]+1, nc[3]))
            restrict_faces_kernel!(cpu)(bz, cbz, Val(3), P.clo, G; ndrange=(nc[1], nc[2], nc[3]+1))
            KA.synchronize(cpu)
        end
    end
    for (l, P) in enumerate(H.patches)                        # 3. cells (E from the final face B)
        nx, ny, nz = ng(P)
        U = zeros(T, nx, ny, nz, nvs(H))
        for k in 1:nz, j in 1:ny, i in 1:nx
            q = ic(_cell_x(P,i,1), _cell_x(P,j,2), _cell_x(P,k,3))
            ρ, vx, vy, vz, pr = q[1], q[2], q[3], q[4], q[5]
            if ct(H)
                bx, by, bz = hb[l]
                Bx = 0.5*(Float64(bx[i,j,k]) + bx[i+1,j,k]); By = 0.5*(Float64(by[i,j,k]) + by[i,j+1,k])
                Bz = 0.5*(Float64(bz[i,j,k]) + bz[i,j,k+1])
            else
                Bx, By, Bz = q[6], q[7], q[8]
            end
            E = pr/(γ-1) + 0.5ρ*(vx^2+vy^2+vz^2) + 0.5*(Bx^2+By^2+Bz^2)
            vals = (ρ, ρ*vx, ρ*vy, ρ*vz, E, Bx, By, Bz, 0.0)
            for v in 1:nvs(H); U[i,j,k,v] = T(vals[v]); end
        end
        hU[l] = U
    end
    for l in L-1:-1:1                                         # 4. restrict conserved cells
        P = H.patches[l]
        restrict_cells_kernel!(cpu)(hU[l], hU[l+1], P.clo, G, nvr(H); ndrange=P.nc)
        ct(H) && bcc_kernel!(cpu)(hU[l], hb[l]...; ndrange=ng(P))
        KA.synchronize(cpu)
    end
    for (l, P) in enumerate(H.patches)                        # 5. upload
        copyto!(P.U, hU[l])
        if ct(H); copyto!(P.bx, hb[l][1]); copyto!(P.by, hb[l][2]); copyto!(P.bz, hb[l][3]); end
    end
    _sync(H)
    fill!(H.t, 0.0); fill!(H.nsteps, 0); fill!(H.wall, 0.0)
    return H
end

_interior(P::Patch, A) = Array(A)[G+1:G+P.n[1], G+1:G+P.n[2], G+1:G+P.n[3], ntuple(_ -> Colon(), ndims(A)-3)...]

"Totals over the root level (= the whole box; covered cells hold the restricted fine data)."
function totals(H::Hierarchy)
    P = H.patches[1]; U = _interior(P, P.U); dV = P.dx^3
    s(v) = sum(Float64, @view U[:,:,:,v]) * dV
    (mass=s(1), momx=s(2), momy=s(3), momz=s(4), energy=s(5))
end

"""
    max_divb(H) -> Vector{Float64}

Per level: max over interior cells of |∇·B|·dx / max|B| (dimensionless). CT: from the face fields
(exact discrete divergence). GLM: central differences of the cell-centred field.
"""
function max_divb(H::Hierarchy)
    out = Float64[]
    for P in H.patches
        n = P.n
        if ct(H)
            bx = Array(P.bx); by = Array(P.by); bz = Array(P.bz)
            bmax = max(maximum(abs, bx), maximum(abs, by), maximum(abs, bz), 1e-300)
            # periodic root: upper face of the last cell is the lower face of the first
            up(i, m, d) = P.per[d] ? G + mod(i - G - 1, m) + 1 : i
            d = 0.0
            for k in G+1:G+n[3], j in G+1:G+n[2], i in G+1:G+n[1]
                v = (bx[up(i+1,n[1],1),j,k] - bx[i,j,k]) + (by[i,up(j+1,n[2],2),k] - by[i,j,k]) + (bz[i,j,up(k+1,n[3],3)] - bz[i,j,k])
                d = max(d, abs(Float64(v)))
            end
            push!(out, d / bmax)
        else
            U = Array(P.U)
            bmax = max(maximum(abs, @view U[:,:,:,6:8]), 1e-300)
            d = 0.0
            w(i, m, e) = P.per[e] ? G + mod(i - G - 1, m) + 1 : i
            rg(e) = P.per[e] ? (G+1:G+n[e]) : (G+2:G+n[e]-1)
            for k in rg(3), j in rg(2), i in rg(1)
                v = 0.5*((U[w(i+1,n[1],1),j,k,6] - U[w(i-1,n[1],1),j,k,6]) + (U[i,w(j+1,n[2],2),k,7] - U[i,w(j-1,n[2],2),k,7]) +
                         (U[i,j,w(k+1,n[3],3),8] - U[i,j,w(k-1,n[3],3),8]))
                d = max(d, abs(Float64(v)))
            end
            push!(out, d / bmax)
        end
    end
    out
end

function level_summary(io::IO, H::Hierarchy)
    @printf(io, "  level  cells/dim  dx          steps   wall[s]\n")
    for (l, P) in enumerate(H.patches)
        @printf(io, "  %5d  %9d  %.4e  %6d  %7.2f\n", l, P.n[1], P.dx, H.nsteps[l], H.wall[l])
    end
end
level_summary(H::Hierarchy) = level_summary(stdout, H)
