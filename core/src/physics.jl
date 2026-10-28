# physics.jl — pointwise ideal-MHD physics shared by the GLM and CT paths.
#
# Ported from glmmhd_turb.jl (GLMMHDTurb) with three changes:
#   * generic element type T (Float32 production, Float64 for roundoff-level validation);
#     every literal is T(…) so nothing promotes and ifelse never mixes types;
#   * the HLLD degenerate-state guard of Solaceonx/GLMMHDTurb@sweep-fixes (893e82a):
#     |denK| < 1e-4·p*  instead of an absolute 1e-12;
#   * the CT pieces of Solaceonx/GLMMHDTurb@sweep-fixes ct_mhd.jl (9d20e52, 7256715):
#     full MUSCL-Hancock predictor in all three directions and the Gardiner & Stone (2005)
#     CT-contact upwinded edge EMF (`gs05`).
#
# Accelerations kept from GLMMHDTurb: @fastmath MonCen with branchless ifelse (the 398 → 993
# Mcell/s fix), NTuple register-resident states, compile-time Riemann selection via Val, HLL
# default (branchless 2-wave), each reconstruction computed once and each face solved once.
#
# Conserved order (9): (ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz, ψ)
# Primitive order (9): (ρ, vx, vy, vz, p, Bx, By, Bz, ψ)
# The CT path uses the same 9-tuples with ψ ≡ 0 and the normal B of a face set to the face value.

const NV9 = 9

@fastmath @inline function cons2prim(c::NTuple{9,T}, gamma::T, smallr::T, pfl::T) where {T}
    rho = max(c[1], smallr)
    ir  = one(T) / rho
    vx = c[2]*ir; vy = c[3]*ir; vz = c[4]*ir
    bx = c[6]; by = c[7]; bz = c[8]
    ekin = T(0.5)*(c[2]*c[2] + c[3]*c[3] + c[4]*c[4])*ir
    emag = T(0.5)*(bx*bx + by*by + bz*bz)
    p = max((gamma - one(T))*(c[5] - ekin - emag), pfl)
    return (rho, vx, vy, vz, p, bx, by, bz, c[9])
end

@inline function prim2cons(q::NTuple{9,T}, gamma::T) where {T}
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    ekin = T(0.5)*rho*(vx*vx + vy*vy + vz*vz)
    emag = T(0.5)*(bx*bx + by*by + bz*bz)
    E = p/(gamma - one(T)) + ekin + emag
    return (rho, rho*vx, rho*vy, rho*vz, E, bx, by, bz, q[9])
end

@fastmath @inline function fast_speed(q::NTuple{9,T}, gamma::T, bn::T) where {T}
    rho = q[1]
    c2 = gamma * q[5] / rho
    b2 = (q[6]*q[6] + q[7]*q[7] + q[8]*q[8]) / rho
    d2 = T(0.5)*(b2 + c2)
    return sqrt(d2 + sqrt(max(d2*d2 - c2*bn*bn/rho, zero(T))))
end

@inline function phys_flux_x(q::NTuple{9,T}, gamma::T) where {T}
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    b2 = bx*bx + by*by + bz*bz
    ptot = p + T(0.5)*b2
    E = p/(gamma - one(T)) + T(0.5)*rho*(vx*vx + vy*vy + vz*vz) + T(0.5)*b2
    vdotb = vx*bx + vy*by + vz*bz
    return (rho*vx,
            rho*vx*vx + ptot - bx*bx,
            rho*vx*vy - bx*by,
            rho*vx*vz - bx*bz,
            (E + ptot)*vx - bx*vdotb,
            zero(T),
            vx*by - vy*bx,
            vx*bz - vz*bx,
            zero(T))
end

@inline function rot_to(q::NTuple{9}, dir::Int)
    dir == 1 && return q
    dir == 2 && return (q[1], q[3], q[4], q[2], q[5], q[7], q[8], q[6], q[9])
    return (q[1], q[4], q[2], q[3], q[5], q[8], q[6], q[7], q[9])
end
@inline function rot_flux_from(f::NTuple{9}, dir::Int)
    dir == 1 && return f
    dir == 2 && return (f[1], f[4], f[2], f[3], f[5], f[8], f[6], f[7], f[9])
    return (f[1], f[3], f[4], f[2], f[5], f[7], f[8], f[6], f[9])
end

# MonCen, @fastmath + branchless (GLMMHDTurb's decisive optimization).
@fastmath @inline function moncen(dl::T, dr::T) where {T}
    dc  = T(0.5)*(dl + dr)
    sgn = ifelse(dc >= zero(T), one(T), -one(T))
    val = sgn * min(T(2)*min(abs(dl), abs(dr)), abs(dc))
    return ifelse(dl*dr <= zero(T), zero(T), val)
end

# HALF slope (face = centre ± s), the GLMMHDTurb convention. The ct_mhd.jl bug fixed on
# sweep-fixes (9d20e52) was halving this a second time; here every caller uses ±1·s.
@fastmath @inline prim_slope(L::NTuple{9,T}, M::NTuple{9,T}, R::NTuple{9,T}) where {T} =
    ntuple(i -> T(0.5)*moncen(M[i]-L[i], R[i]-M[i]), 9)

@inline padd(q::NTuple{9,T}, s::NTuple{9,T}, a::T) where {T} = ntuple(i -> q[i] + a*s[i], 9)

@inline dir_flux(q::NTuple{9,T}, dir::Int, gamma::T) where {T} =
    rot_flux_from(phys_flux_x(rot_to(q, dir), gamma), dir)

# MUSCL-Hancock half step with the flux differences of all three directions.
@fastmath @inline function hancock(m0::NTuple{9,T}, sx, sy, sz, dtdx::T, gamma::T) where {T}
    u0 = prim2cons(m0, gamma)
    h = T(0.5)*dtdx
    fxp = dir_flux(padd(m0, sx, one(T)), 1, gamma); fxm = dir_flux(padd(m0, sx, -one(T)), 1, gamma)
    fyp = dir_flux(padd(m0, sy, one(T)), 2, gamma); fym = dir_flux(padd(m0, sy, -one(T)), 2, gamma)
    fzp = dir_flux(padd(m0, sz, one(T)), 3, gamma); fzm = dir_flux(padd(m0, sz, -one(T)), 3, gamma)
    return ntuple(i -> u0[i] - h*((fxp[i]-fxm[i]) + (fyp[i]-fym[i]) + (fzp[i]-fzm[i])), 9)
end

@inline function glm_pair(bnL::T, bnR::T, psiL::T, psiR::T, ch::T) where {T}
    bns  = T(0.5)*(bnL + bnR) - T(0.5)*(psiR - psiL)/ch
    psis = T(0.5)*(psiL + psiR) - T(0.5)*ch*(bnR - bnL)
    return bns, psis
end

@fastmath @inline function hll_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, bn::T, fbn::T, fpsi::T) where {T}
    cfL = fast_speed(L, gamma, bn); cfR = fast_speed(R, gamma, bn)
    SL = min(min(L[2]-cfL, R[2]-cfR), zero(T)); SR = max(max(L[2]+cfL, R[2]+cfR), zero(T))
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    ihd = one(T)/(SR - SL)
    f = ntuple(i -> (SR*fL[i] - SL*fR[i] + SL*SR*(uR[i]-uL[i]))*ihd, 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

@fastmath @inline function llf_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, ch::T, fbn::T, fpsi::T) where {T}
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    smax = max(abs(L[2]) + fast_speed(L, gamma, L[6]), abs(R[2]) + fast_speed(R, gamma, R[6]))
    f = ntuple(i -> T(0.5)*(fL[i] + fR[i]) - T(0.5)*smax*(uR[i] - uL[i]), 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

# HLLD (Miyoshi & Kusano 2005) in the x frame, with the sweep-fixes degenerate guard.
@fastmath @inline function hlld_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, bn::T, fbn::T, fpsi::T) where {T}
    dL,uL,vL,wL,pL = L[1],L[2],L[3],L[4],L[5]; byL,bzL = L[7],L[8]
    dR,uR,vR,wR,pR = R[1],R[2],R[3],R[4],R[5]; byR,bzR = R[7],R[8]
    h = T(0.5); g1 = gamma - one(T)
    b2L = bn*bn + byL*byL + bzL*bzL; b2R = bn*bn + byR*byR + bzR*bzR
    ptL = pL + h*b2L; ptR = pR + h*b2R
    EL = pL/g1 + h*dL*(uL*uL+vL*vL+wL*wL) + h*b2L
    ER = pR/g1 + h*dR*(uR*uR+vR*vR+wR*wR) + h*b2R
    cfL = fast_speed(L, gamma, bn); cfR = fast_speed(R, gamma, bn)
    SL = min(min(uL, uR) - max(cfL, cfR), zero(T))
    SR = max(max(uL, uR) + max(cfL, cfR), zero(T))
    UL = (dL, dL*uL, dL*vL, dL*wL, EL, bn, byL, bzL, zero(T))
    UR = (dR, dR*uR, dR*vR, dR*wR, ER, bn, byR, bzR, zero(T))
    FL = phys_flux_x((dL,uL,vL,wL,pL,bn,byL,bzL,zero(T)), gamma)
    FR = phys_flux_x((dR,uR,vR,wR,pR,bn,byR,bzR,zero(T)), gamma)
    den = (SR - uR)*dR - (SL - uL)*dL
    SM  = ((SR - uR)*dR*uR - (SL - uL)*dL*uL - ptR + ptL) / den
    pts = ((SR - uR)*dR*ptL - (SL - uL)*dL*ptR + dL*dR*(SR - uR)*(SL - uL)*(uR - uL)) / den
    dLs = max(dL*(SL - uL)/(SL - SM), T(1e-12)); dRs = max(dR*(SR - uR)/(SR - SM), T(1e-12))
    sqdLs = sqrt(dLs); sqdRs = sqrt(dRs)
    SLs = SM - abs(bn)/sqdLs; SRs = SM + abs(bn)/sqdRs
    function star(d, u, v, w, by, bz, S)
        denK = d*(S-u)*(S-SM) - bn*bn
        if abs(denK) < T(1e-4)*pts          # sweep-fixes guard (relative to p*)
            return v, w, by, bz
        end
        fac = bn*(SM - u)/denK
        num = d*(S-u)*(S-u) - bn*bn
        return v - by*fac, w - bz*fac, by*num/denK, bz*num/denK
    end
    vyLs, vzLs, byLs, bzLs = star(dL,uL,vL,wL,byL,bzL,SL)
    vyRs, vzRs, byRs, bzRs = star(dR,uR,vR,wR,byR,bzR,SR)
    vdotbL = uL*bn + vL*byL + wL*bzL; vdotbLs = SM*bn + vyLs*byLs + vzLs*bzLs
    ELs = ((SL - uL)*EL - ptL*uL + pts*SM + bn*(vdotbL - vdotbLs))/(SL - SM)
    vdotbR = uR*bn + vR*byR + wR*bzR; vdotbRs = SM*bn + vyRs*byRs + vzRs*bzRs
    ERs = ((SR - uR)*ER - ptR*uR + pts*SM + bn*(vdotbR - vdotbRs))/(SR - SM)
    ULs = (dLs, dLs*SM, dLs*vyLs, dLs*vzLs, ELs, bn, byLs, bzLs, zero(T))
    URs = (dRs, dRs*SM, dRs*vyRs, dRs*vzRs, ERs, bn, byRs, bzRs, zero(T))
    sgn = ifelse(bn >= zero(T), one(T), -one(T))
    invsum = one(T)/(sqdLs + sqdRs)
    vyss = (sqdLs*vyLs + sqdRs*vyRs + (byRs - byLs)*sgn)*invsum
    vzss = (sqdLs*vzLs + sqdRs*vzRs + (bzRs - bzLs)*sgn)*invsum
    byss = (sqdLs*byRs + sqdRs*byLs + sqdLs*sqdRs*(vyRs - vyLs)*sgn)*invsum
    bzss = (sqdLs*bzRs + sqdRs*bzLs + sqdLs*sqdRs*(vzRs - vzLs)*sgn)*invsum
    vdotbss = SM*bn + vyss*byss + vzss*bzss
    ELss = ELs - sqdLs*(vdotbLs - vdotbss)*sgn
    ERss = ERs + sqdRs*(vdotbRs - vdotbss)*sgn
    ULss = (dLs, dLs*SM, dLs*vyss, dLs*vzss, ELss, bn, byss, bzss, zero(T))
    URss = (dRs, dRs*SM, dRs*vyss, dRs*vzss, ERss, bn, byss, bzss, zero(T))
    if SLs >= zero(T)
        f = ntuple(i -> FL[i] + SL*(ULs[i] - UL[i]), 9)
    elseif SM >= zero(T)
        f = ntuple(i -> FL[i] + SLs*ULss[i] - (SLs - SL)*ULs[i] - SL*UL[i], 9)
    elseif SRs >= zero(T)
        f = ntuple(i -> FR[i] + SRs*URss[i] - (SRs - SR)*URs[i] - SR*UR[i], 9)
    else
        f = ntuple(i -> FR[i] + SR*(URs[i] - UR[i]), 9)
    end
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

@inline riemann_x(::Val{:hll},  L, R, gamma, bn, fbn, fpsi, ch) = hll_x(L, R, gamma, bn, fbn, fpsi)
@inline riemann_x(::Val{:hlld}, L, R, gamma, bn, fbn, fpsi, ch) = hlld_x(L, R, gamma, bn, fbn, fpsi)
@inline riemann_x(::Val{:llf},  L, R, gamma, bn, fbn, fpsi, ch) = llf_x(L, R, gamma, ch, fbn, fpsi)

# Riemann in direction `dir` for interface prims Lq, Rq.
#   GLM (CT=false): Dedner pair on (Bn, ψ) → F[Bn] = ψ*, F[ψ] = c_h² Bn*.
#   CT  (CT=true) : Bn is the single face value `bface`; F[Bn] = 0, ψ ≡ 0.
@fastmath @inline function riemann_dir(rv::Val, ::Val{CT}, Lq::NTuple{9,T}, Rq::NTuple{9,T}, dir::Int,
                                       gamma::T, ch::T, smallr::T, pfl::T, bface::T) where {CT,T}
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    if CT
        bns = bface; psis = zero(T); fpsi = zero(T)
    else
        bns, psis = glm_pair(L0[6], R0[6], L0[9], R0[9], ch)
        fpsi = ch*ch*bns
    end
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), bns, L0[7],L0[8], L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), bns, R0[7],R0[8], R0[9])
    f = riemann_x(rv, L, R, gamma, bns, psis, fpsi, ch)
    return rot_flux_from(f, dir)
end

# Total directional signal speed for the unsplit CFL (Σ_d |v_d| + c_f,d), as GLMMHDTurb.
@inline function signal_speed(q::NTuple{9,T}, gamma::T) where {T}
    (abs(q[2]) + fast_speed(q, gamma, q[6])) + (abs(q[3]) + fast_speed(q, gamma, q[7])) +
    (abs(q[4]) + fast_speed(q, gamma, q[8]))
end

# Gardiner & Stone (2005) CT-contact edge EMF (eqs. 41-45), as in sweep-fixes ct_mhd.jl (7256715).
# Edge at (a-½, b-½): Ea0/Eam face EMFs on a-faces (a-½,b),(a-½,b-1); Eb0/Ebm on b-faces
# (a,b-½),(a-1,b-½); mA*/mB* the mass fluxes through them; Er** cell-centred reference EMFs of
# cells (a+da, b+db), da,db ∈ {0,-1}.
@inline function gs05(Ea0::T,Eam::T,Eb0::T,Ebm::T,mA0::T,mAm::T,mB0::T,mBm::T,Er00::T,Erm0::T,Er0m::T,Ermm::T) where {T}
    up(m,lower,upper) = ifelse(m > zero(T), lower, ifelse(m < zero(T), upper, T(0.5)*(lower+upper)))
    A1 = up(mA0, Erm0-Ebm, Er00-Eb0)
    A3 = up(mAm, Ebm-Ermm, Eb0-Er0m)
    B1 = up(mB0, Er0m-Eam, Er00-Ea0)
    B3 = up(mBm, Eam-Ermm, Ea0-Erm0)
    T(0.25)*((Ea0+Eam+Eb0+Ebm) + (A3-A1) + (B3-B1))
end
