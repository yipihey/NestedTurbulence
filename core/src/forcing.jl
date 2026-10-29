# forcing.jl — driven turbulence: a finite set of Fourier modes evaluated in real space on every
# level, an isothermal equation of state by energy reset, and Mach-number feedback.
#
# Sources hook into the driver (hierarchy.jl) through four functions with no-op defaults:
#   apply_sources!(src, H, P, t, dt)   after every patch step (P advanced from t to t+dt)
#   post_sync!(src, H, P)              after a parent has been refluxed / restricted from its child
#   after_step!(src, H, l)             after level l has completed a step (children included)
#   finalize!(src, H)                  at the end of evolve!
apply_sources!(src, H, P, t, dt) = nothing
post_sync!(src, H, P) = nothing
after_step!(src, H, l) = nothing
finalize!(src, H) = nothing

# ---------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------
@inline _sinc(u::T) where {T} = abs(u) < T(1e-8) ? one(T) : sin(T(π)*u) / (T(π)*u)

# ρv += ρ a dt, with a(x) = Σ_m 2 Re[ â_m exp(2πi k_m·x) ] averaged EXACTLY over the cell (the
# product of sinc factors), so the restriction of the fine-level force is the coarse-level force.
# A: (6, nm) = Re â_x, Im â_x, Re â_y, Im â_y, Re â_z, Im â_z;  K: (3, nm) integer wave numbers.
# The kinetic energy change is added to E (E is reset afterwards anyway when isothermal).
@kernel function force_kernel!(U, @Const(K), @Const(A), nm::Int, lo::NTuple{3,Int}, g::Int, dx::T, L::T, dt::T) where {T}
    I = @index(Global, NTuple)
    i = I[1] + g; j = I[2] + g; k = I[3] + g
    @inbounds begin
        x = (lo[1] + I[1] - T(1.5)) * dx / L; y = (lo[2] + I[2] - T(1.5)) * dx / L; z = (lo[3] + I[3] - T(1.5)) * dx / L
        h = dx / L
        ax = zero(T); ay = zero(T); az = zero(T)
        for m in 1:nm
            kx = K[1,m]; ky = K[2,m]; kz = K[3,m]
            ph = T(2π) * (kx*x + ky*y + kz*z)
            s = 2 * _sinc(kx*h) * _sinc(ky*h) * _sinc(kz*h)
            c = cos(ph); sn = sin(ph)
            ax += s * (A[1,m]*c - A[2,m]*sn); ay += s * (A[3,m]*c - A[4,m]*sn); az += s * (A[5,m]*c - A[6,m]*sn)
        end
        ρ = U[i,j,k,1]
        mx = U[i,j,k,2]; my = U[i,j,k,3]; mz = U[i,j,k,4]
        e0 = (mx*mx + my*my + mz*mz) / (2ρ)
        mx += ρ*ax*dt; my += ρ*ay*dt; mz += ρ*az*dt
        U[i,j,k,2] = mx; U[i,j,k,3] = my; U[i,j,k,4] = mz
        U[i,j,k,5] += (mx*mx + my*my + mz*mz) / (2ρ) - e0
    end
end

# isothermal EOS: E = ρ cs²/(γ-1) + ½ρv² + ½B² on every interior cell (B: cell-centred).
# Also the density floor for cells that the refluxing pushed below it (the floor of
# update_cells_kernel! only sees the flux update): ρ = smallr at rest, as in GLMMHDTurb.
@kernel function isothermal_kernel!(U, g::Int, cs2::T, gm1::T, smallr::T) where {T}
    I = @index(Global, NTuple)
    i = I[1] + g; j = I[2] + g; k = I[3] + g
    @inbounds begin
        if U[i,j,k,1] < smallr
            U[i,j,k,1] = smallr; U[i,j,k,2] = zero(T); U[i,j,k,3] = zero(T); U[i,j,k,4] = zero(T)
        end
        ρ = U[i,j,k,1]; mx = U[i,j,k,2]; my = U[i,j,k,3]; mz = U[i,j,k,4]
        b6 = U[i,j,k,6]; b7 = U[i,j,k,7]; b8 = U[i,j,k,8]
        U[i,j,k,5] = ρ*cs2/gm1 + (mx*mx + my*my + mz*mz)/(2ρ) + T(0.5)*(b6*b6 + b7*b7 + b8*b8)
    end
end

# Galilean shift v → v - V on all cells (ghosts included), keeping the internal energy
@kernel function shift_kernel!(U, Vx::T, Vy::T, Vz::T) where {T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρ = U[i,j,k,1]; mx = U[i,j,k,2]; my = U[i,j,k,3]; mz = U[i,j,k,4]
        U[i,j,k,5] += -(mx*Vx + my*Vy + mz*Vz) + T(0.5)*ρ*(Vx*Vx + Vy*Vy + Vz*Vz)
        U[i,j,k,2] = mx - ρ*Vx; U[i,j,k,3] = my - ρ*Vy; U[i,j,k,4] = mz - ρ*Vz
    end
end

# ---------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------
"""
    Driving(H; mach=10, cs=1, kmin=1, kmax=2, zeta=0.5, t_corr=nothing, two_d=true, seed=1,
            feedback=true, t_feedback=nothing, mass_weighted=true, isothermal=true)

Large-scale turbulence driving for hierarchy `H`. The acceleration is a sum over the finite set
of Fourier modes with integer wave vectors `kmin ≤ |k| ≤ kmax` (in units of 2π/L; `kz = 0` and in-plane
amplitudes when `two_d`), each with an Ornstein–Uhlenbeck amplitude of correlation time `t_corr`
(default: half the large-scale crossing time t_c = L/(mach·cs)), projected with the solenoidal
weight `zeta` (1 = solenoidal, 0 = compressive). The OU process is sampled on a fixed grid of
nodes (spacing t_corr/32) and interpolated linearly, so the force is a deterministic function of
(x, t): every level evaluates the same field at its own cells and its own times.

`feedback`: after every root step the overall amplitude is multiplied by
(M_target/M_rms)^(Δt/t_feedback) (default t_feedback = t_c/5; log-ratio clamped to ±0.25), with M_rms the rms Mach
number of the root level (which holds the restricted fine data): mass-weighted,
M = (Σρv²/Σρ)^½/cs, by default (`mass_weighted=true`; it measures the kinetic energy), or
volume-weighted, M = ⟨v²⟩^½/cs. Both are logged. In a 2-D Mach-10 flow the volume-weighted value
is dominated by fast, nearly empty voids (an acceleration field drives low-density gas as hard as
dense gas), and controlling on it starved the dense gas of driving in our tests. `isothermal`: the total energy is
reset to ρcs²/(γ-1) + kinetic + magnetic after every update. The mean momentum is removed after
every root step (a Galilean shift applied to all levels).
"""
mutable struct Driving
    mach::Float64; cs::Float64; zeta::Float64; t_corr::Float64; dtn::Float64
    K::Matrix{Int}                 # (3, nm)
    nodes::Vector{Matrix{Float64}} # OU amplitudes (6, nm) at t = (n-1)·dtn, unit rms field
    rng::Xoshiro
    amp::Float64                   # feedback multiplier
    feedback::Bool; t_feedback::Float64; mass_weighted::Bool
    isothermal::Bool
    two_d::Bool
    Kd; Ad                         # device copies
    log_t::Vector{Float64}; log_mach::Vector{Float64}; log_mach_m::Vector{Float64}; log_amp::Vector{Float64}
end

function Driving(H; mach=10.0, cs=1.0, kmin=1.0, kmax=2.0, zeta=0.5, t_corr=nothing, two_d=true, seed=1,
                 feedback=true, t_feedback=nothing, mass_weighted=true, isothermal=true, amp0=nothing)
    L = Float64(H.p.boxlen)
    tc = L / (mach * cs)
    tcor = t_corr === nothing ? tc / 2 : Float64(t_corr)
    ks = Vector{NTuple{3,Int}}()
    km = ceil(Int, kmax)
    for kx in -km:km, ky in -km:km, kz in (two_d ? (0:0) : (-km:km))
        kk = sqrt(kx^2 + ky^2 + kz^2)
        (kmin - 1e-9 <= kk <= kmax + 1e-9) || continue
        # half space: one of each ±k pair (the field is 2 Re[...])
        (kx > 0 || (kx == 0 && (ky > 0 || (ky == 0 && kz > 0)))) || continue
        push!(ks, (kx, ky, kz))
    end
    K = reshape(collect(Iterators.flatten(ks)), 3, length(ks))
    T = H.T
    d = Driving(mach, cs, zeta, tcor, tcor / 32, K, Matrix{Float64}[], Xoshiro(seed),
                amp0 === nothing ? mach^2 * cs^2 / L : Float64(amp0),
                feedback, t_feedback === nothing ? tc / 5 : Float64(t_feedback), mass_weighted, isothermal, two_d,
                KA.zeros(H.backend, T, 3, size(K, 2)), KA.zeros(H.backend, T, 6, size(K, 2)),
                Float64[], Float64[], Float64[], Float64[])
    copyto!(d.Kd, T.(K))
    push!(d.nodes, _ou_draw(d))                     # start in the stationary distribution
    d
end

nmodes(d::Driving) = size(d.K, 2)

# projected complex Gaussian amplitude for every mode, normalised so the field has unit rms per
# component direction summed (⟨|a|²⟩ = 1 over the box for the stationary process)
function _ou_draw(d::Driving)
    nm = nmodes(d); A = zeros(6, nm)
    for m in 1:nm
        k = d.K[:, m]; kh = k ./ sqrt(sum(abs2, k))
        for part in (0, 1)                           # real, imaginary
            w = randn(d.rng, 3)
            d.two_d && (w[3] = 0.0)
            wl = sum(w .* kh) .* kh                  # compressive part
            wt = w .- wl                             # solenoidal part
            v = d.zeta .* wt .+ (1 - d.zeta) .* wl
            for c in 1:3; A[2c-1+part, m] = v[c]; end
        end
    end
    # normalise: ⟨|a|²⟩ = Σ_m 2|â_m|² (each mode appears with its conjugate)
    nrm = sqrt(2 * sum(abs2, A))
    A ./ nrm
end

function _extend!(d::Driving, t)
    f = exp(-d.dtn / d.t_corr)
    while (length(d.nodes) - 1) * d.dtn < t + d.dtn
        # exact OU update in distribution: a ← f a + sqrt(1-f²) ξ, ξ from the stationary law
        push!(d.nodes, f .* d.nodes[end] .+ sqrt(1 - f^2) .* _ou_draw(d))
    end
end

"OU amplitudes (6, nm) at time `t` (unit rms), linearly interpolated between nodes."
function amplitudes(d::Driving, t)
    _extend!(d, t)
    s = t / d.dtn; n = floor(Int, s); w = s - n
    (1 - w) .* d.nodes[n+1] .+ w .* d.nodes[n+2]
end

function apply_sources!(d::Driving, H, P, t, dt)
    be = H.backend; T = H.T
    A = amplitudes(d, t + dt/2) .* d.amp
    copyto!(d.Ad, T.(A))
    force_kernel!(be)(P.U, d.Kd, d.Ad, nmodes(d), P.lo, G, T(P.dx), T(H.p.boxlen), T(dt); ndrange=P.n)
    d.isothermal && isothermal_kernel!(be)(P.U, G, T(d.cs^2), T(H.p.gamma - 1), H.p.smallr; ndrange=P.n)
end

post_sync!(d::Driving, H, P) =
    d.isothermal && isothermal_kernel!(H.backend)(P.U, G, H.T(d.cs^2), H.T(H.p.gamma - 1), H.p.smallr; ndrange=P.n)

"rms Mach number of level `l` (interior), volume-weighted (`mass=false`) or mass-weighted"
function rms_mach(H, l, cs; mass::Bool=false)
    P = H.patches[l]; r = ntuple(d -> G+1:G+P.n[d], 3)
    ρ = @view P.U[r..., 1]
    m2 = @view(P.U[r..., 2]).^2 .+ @view(P.U[r..., 3]).^2 .+ @view(P.U[r..., 4]).^2
    mass ? sqrt(Float64(sum(m2 ./ ρ)) / Float64(sum(ρ))) / cs : sqrt(Float64(sum(m2 ./ ρ.^2)) / prod(P.n)) / cs
end

function after_step!(d::Driving, H, l)
    l == 1 || return
    # 1. remove the mean momentum (whole box = root, which holds the restricted fine data)
    tt = totals(H); m = tt.mass
    V = (tt.momx / m, tt.momy / m, tt.momz / m)
    T = H.T
    for P in H.patches
        shift_kernel!(H.backend)(P.U, T(V[1]), T(V[2]), T(V[3]); ndrange=ng(P))
    end
    # 2. Mach feedback on the root
    Mv = rms_mach(H, 1, d.cs); Mm = rms_mach(H, 1, d.cs; mass=true)
    M = d.mass_weighted ? Mm : Mv
    dtr = H.t[1] - (isempty(d.log_t) ? 0.0 : d.log_t[end])
    if d.feedback && M > 0
        x = clamp(dtr / d.t_feedback * log(d.mach / M), -0.25, 0.25)
        d.amp *= exp(x)
    end
    push!(d.log_t, H.t[1]); push!(d.log_mach, Mv); push!(d.log_mach_m, Mm); push!(d.log_amp, d.amp)
end
