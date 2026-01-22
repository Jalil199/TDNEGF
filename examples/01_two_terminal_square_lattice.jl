#!/usr/bin/env julia

#=
Example: 2D square lattice device coupled to square-lattice leads

This example demonstrates how to use TDNEGF.jl to:
  * Build a square-lattice Hamiltonian H_ab for a finite Nx × Ny system.
  * Construct 2D square-lattice embedding self-energies using a pole expansion
    (MiniPole residues for a semicircular DOS + Pade poles for the Fermi function).
  * Set up channel vectors ξ_anα for left/right 2D leads and assemble Σ^</>(t).
  * Integrate the TD-NEGF embedding EOM eom!(du,u,p,t) to obtain ρ_ab(t).
  * Compute simple observables: local charge density n_i(t), bond currents, etc.

    This script is intended as a pedagogical example and a template for users to
    build their own setups, rather than a production-grade simulation input

=#
using Pkg
Pkg.activate(joinpath(dirname((@__DIR__))))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using BenchmarkTools

println("Number of threads with JUlIA_NUM used in operations is : " , Threads.nthreads() )
println("Number of threads with BLAS used in operations is : " , BLAS.get_num_threads() )

#### Define model and initial parameters
function init_params(;Nx::Int=6, Ny::Int=2, Nσ::Int=2, N_orb::Int=1,
                     γ::Float64=1.0, γso=0.5 + 0.0im,Nα = 2, 
                     N_λ1::Int=49, N_λ2::Int=20, β::Float64=33.0)
    ### Get the poles 
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    ####Note that we can initiallize the system only with the dimension of the system 
    p = ModelParamsTDNEGF(Nx   = Nx, Ny   = Ny,  Nσ   = Nσ, N_orb = N_orb, #### dimension
                          Nα   = Nα,  N_λ1 = N_λ1, N_λ2 = N_λ2 ) ;
    ### Now we can assign the real initial values 
    H_ab    = build_H_ab(;Nx = p.Nx,Ny = p.Ny, Nσ = p.Nσ, N_orb = p.N_orb, γ = γ, γso = γso)
    Σᴸ_nλ   = build_Σᴸ_nλ(Rλ, zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β=β, γ=1.0) ;
    Σᴳ_nλ   = build_Σᴳ_nλ(Rλ, zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β=β, γ=1.0) ;
    χ_nλ    = build_χ_nλ(zλ, p.Ny, p.Nσ, p.N_orb, p.N_λ1, p.N_λ2; β=β, γ=1.0)
    ξ_anR   = build_ξ_an(p.Nx, p.Ny, p.Nσ, p.N_orb; xcol = p.Nx,y_coup = 1:p.Ny)
    ξ_anL   = build_ξ_an(p.Nx, p.Ny, p.Nσ, p.N_orb; xcol = 1,y_coup = 1:p.Ny)
    #### Asigning initial values of the precalculated things
    p.H_ab             .= H_ab
    p.H0_ab            .= H_ab
    p.Δ_α              .= [0.5,-0.5] 
    #### Self energies
    p.Σᴸ_nλα[:,:,1]    .= Σᴸ_nλ
    p.Σᴸ_nλα[:,:,2]    .= Σᴸ_nλ
    p.Σᴳ_nλα[:,:,1]    .= Σᴳ_nλ
    p.Σᴳ_nλα[:,:,2]    .= Σᴳ_nλ
    Γ_nλα = 1im*copy(p.Σᴳ_nλα - p.Σᴸ_nλα)
    p.Γ_nλα    .= Γ_nλα
    #1im*(p.Σᴳ_nλα - p.Σᴸ_nλα)
    #### Exponent from poles in the residue theorem 
    p.χ_nλα[:,:,1]     .= χ_nλ
    p.χ_nλα[:,:,2]     .= χ_nλ
    #### Channel vectors 
    p.ξ_anα[:,:,1]     .= ξ_anL
    p.ξ_anα[:,:,2]     .= ξ_anR;
    ####
    p.χ′_nλα  .= conj.(p.χ_nλα)
    p.Σᴸ′_nλα .= conj.(p.Σᴸ_nλα)
    p.Γ′_nλα  .= conj.(p.Γ_nλα);
    return p
end ;

p = init_params()
println("The initial parameters have been set")
#### Define and solve the ODE problem 
prob = ODEProblem(eom_tdnegf!,p.u, (0.0,100.0), p )
println("The integrator has been set")
#init(prob,Vern7(),dt = Δt, save_everystep=false,adaptive=true,dense=false)
#@btime sol = solve(prob,Vern7(), save_everystep=false ,adaptive=true,dense=false,reltol=1e-6, abstol=1e-8)
# @btime solve(prob,Vern7(), save_everystep=false ,adaptive=true,dense=false,reltol=1e-6, abstol=1e-8)
# p = init_params()
@time sol = solve(prob,Vern7(),adaptive=true,dense=false,reltol=1e-6, abstol=1e-8) #, save_everystep=false  (In case u only want the final state)
#@time sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-8, save_everystep=false);
println("Solution was obtained")
#### Alocate the observables 
obs = ObservablesTDNEGF(p; N_tmax=length(sol.t) , N_leads = p.Nα)
obs.t = sol.t
for (it,ut) in enumerate(sol.u)
    obs.idx = it
    dv = TDNEGF.pointer(ut,p)
    obs_n_i!(dv, p, obs)            ### Charge density
    obs_σ_i!(dv, p, obs)            ### Spin density 
    obs_Ixα!(dv, p, obs)   ### Spin and charge current
end

#### Save results
using JLD2
@save "./examples/data/two_terminal_square_lattice_observables.jl2" obs
println("The results has been saved")
#### Plot results 
# using PyPlot 
# plt.rc("axes", linewidth=1)  # Set the linewidth of the plot axes
# plt.rc("text", usetex=true)  # Enable LaTeX rendering of text
# fs = 25
# ### Plot the charge denisty
# ###------------------------------------------------------
# fig,axs =  plt.subplots(1,1)
# site = 1
# sites = 1:2:12#range(1,12)
# for site in sites
#     axs.plot(obs.t,obs.n_i[site,:],label= "site=$(site)")#,alpha =1-0.2*i ) ### Charge bound current
# end
# axs.set_ylabel(raw"$\langle\mathrm{\hat{n}_i}\rangle$", fontsize = fs)
# axs.set_xlabel(raw"$\mathrm{Time\ (\hbar/\gamma)}$",fontsize = fs)
# axs.tick_params(axis="both", which="both", labelsize=fs,direction="in", length=6,width=1)
# axs.ticklabel_format(axis="y", style="sci", scilimits=(-1,2), useMathText=true)
# axs.yaxis.offsetText.set_fontsize(fs)
# plt.legend(frameon = false, fontsize = fs-10, loc= (0.7,  0.1))
# plt.tight_layout()
# plt.show()
# ###------------------------------------------------------

# ### Plot the charge current 
# ###------------------------------------------------------
# fig,axs = plt.subplots(1,1)
# axs.plot(obs.t,obs.Iα[1,:]*pi, label = raw"$\mathrm{I_R}$")  ### Units of 2e/h 
# axs.plot(obs.t,obs.Iα[2,:]*pi, label = raw"$\mathrm{I_L}$")
# axs.set_ylabel(L"$\mathrm{I\ (2e\gamma/h)}$", fontsize = fs)
# axs.set_xlabel(raw"$\mathrm{Time\ (\hbar/\gamma)}$",fontsize = fs)
# axs.tick_params(axis="both", which="both", labelsize=fs,direction="in", length=6,width=1)
# axs.ticklabel_format(axis="y", style="sci", scilimits=(-1,2), useMathText=true)
# axs.yaxis.offsetText.set_fontsize(fs)
# plt.legend(frameon = false, fontsize = fs-10, loc= (0.7,  0.7))
# plt.tight_layout()
# plt.show()
# ###------------------------------------------------------

