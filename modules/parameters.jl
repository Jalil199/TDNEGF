module parameters
### Libraries
using LinearAlgebra          ### Linear algebra library
using StaticArrays           ### Small matrices 
using DelimitedFiles         ### Manipulate files 
### Principal parameters
export n, n_lorentz,n_channels,delta_tdep_L,delta_tdep_R,E_F_system, V_bias,
       E_F_left, E_F_right,alpha_r,theta_1,phi_1,theta_2,phi_2,period,N_rash,
       Temp, N_poles,t_0,t_step,t_end,J_sd,n_precessing,J_qsl,A,θ,save_data_qsl,data_bias,
       n_sites,nt,dt,h0,j_exc,g_lambda,j_sd,j_dmi,j_ani,j_dem,e,e_demag,js_pol,js_ana,thop,
       p_theta,p_phi,curr,scurr,rho,sden,solver,cspin_orientation,read_bias_file,
       bias_file,name,save_data, run_llg
### Secondary parameters
export w0_k1α,eps_k1α,gam_k1iα,k_poles,energy_llg,E_F_α,beta,
       dims_Omega1, dims_Omega2, dims_Omega3, size_Omega1, 
       size_Omega2, size_Omega3, dims_psi, dims_rho, size_psi,
       size_rho, size_rkvec, params_0,σ_0,σ_x,σ_y,σ_z,MBOHR,KB,
       GAMMA_R,k_Boltzmann,hbar, I_a1b1,I_ab,σ_x1ab,σ_x2ab,
       σ_x3ab,σ_abx,dOmega_αikβjp,dOmega_αik1βjp1,dOmega_αik1βjp2,
       dOmega_αik2βjp1,summ1_aik1α,summ2_aik1α, summ3_aik2α,dpsi_aikα,
       drho_ab, J_sd_local, thop_local, tso_local 

### Global parameters 
const n = 9                              ### number of sites
const n_lorentz = 31                     ### number of  lorentzians 
const n_channels = 2                     ### number of channels(2 for full spin)
const delta_tdep_L,delta_tdep_R  = 0,0
const E_F_system, V_bias = 0., 0.        ### Fermi energy andd Vbias
const E_F_left = E_F_system + 0.5*V_bias
const E_F_right = E_F_system - 0.5*V_bias
const alpha_r = 0.0                      ### Rashba parameter 
const theta_1, phi_1 = 45.0, 0.0         ### Angles of sub lattice 1
const theta_2, phi_2 = 12.9, 0.0         ### Angles of sub lattice 2
##angles sublattice 1
##angles sublattice 2
const period = 3                         ### Period of precession 
const N_rash = 0                         ### number of magnetic moment with rashba from 0 to Nrash 
const Temp, N_poles = 300., 30           ### Temp, Npoles of the fermi function 
const t_0, t_step, t_end = 0,0.1,500      ### time to start step and end
##light flag 
const J_sd = 0.5                         ### jsd_to_llg from classical spin to H_el
const n_precessing = 0                   ### number of spins precessing 
##domain walll width 
#t_long time before start llg 
#t_bias time before start vbias 
#llg_step Num of TDNEGF steps between LLG 
#---------- Parameters for the spin liquid ----------#
const A=[0,5,6,1,7]   ### Sites where the entropy will be calculateed for the spin liquid 
const J_qsl = 0.2
const θ = pi   ### pi is ferro, 0 is antiferro and 3pi/2 is spin liquid
const save_data_qsl = Dict("curr"=>true, "scurr"=>true, "sden_eq"=>true,"sden_neq"=>true, "rho"=>true, "sclas"=>true, "ent" =>true,"sden_qsl"=> true);
#---------- Parameters in the llg equations----------#
const n_sites = n                        ### number of LLG sites (equal to -1)
const nt = 1                             ### number of time steps in llg 
const dt = 0.1                           ### time interval of a single LLG step 
const h0 = Float64[0., 0., 0.]           ### external magnetic field vector
const j_exc = 0.0                        ### jexc coupling between classical spins
const g_lambda = 0.0                     ### gilbert damping factor
const j_sd = 0.5                         ### jsd in the llg equation
const j_dmi = 0.0                        ### Dyaloshinski-Moriya
const j_ani = 0.0                        ### Magnetic anisotropy
const j_dem = 0.0                        ### Demagnetization field 
const e = Float64[0., 0., 0.]            ### Direction of magnetic anisotropy
const e_demag = Float64[0., 0., 0.]      ### Direction of demagnetization vector
const js_pol = 0.0                       ### Jsd coupling of the polarizer
const js_ana = 0.0                       ### Jsd_coupling of the analyzer
const thop = 1.0                         ### Hopping in the central part
const p_theta = 0.0                      ### polarizer vector theta
const p_phi = 0.0                        ### polarizer vector phi
const run_llg = false
#--------------Save and replay--------------------------------#
const curr,scurr = true, true            ### save currents 
const rho, sden = true,true              ### save densities
#replay_llg
#replay_cspins_file
#save_cspins
#save_spindensity_eq
#save_spindensity_neq
#save_charge_current
#save_spin_currents
#save_rho
#preload_rkvec
#save_rkvec_end
#read_bias
const solver = "denis"
const cspin_orientation = "polarizer"#"precess"#"sym_pump"#"polarizer"#"arb_dir" ;    ### Configuration of the classical spins
const save_data = Dict("curr"=>true, "scurr"=>true, "sden_eq"=>true,"sden_neq"=>true, "rho"=>true, "sclas"=>true );
#cspins_file
#spindensity_eq_file
#spindensity_neq_file
#charge_current_file
#spin_current_sfile
#bond_currents_file
#rho_file
const read_bias_file  =  true
const bias_file  =  "./vtd.txt" ;
const name = "test"
#output_path
#load_rkvector_file
#save_rkvector_file
#selenergy_file_path
#write_freq
##############################################################################################################
### dependent_parametes
data_fit_pdbest = readdlm( "./selfenergy/selfenergy_1DTB_NNLS_31_pbest.csv" , ',', Float64)
data_fit_Ulsq = readdlm( "./selfenergy/selfenergy_1DTB_NNLS_31_Ulsq.csv", ',', Float64) ;

### Elementary matrices
σ_0 = SMatrix{2,2,ComplexF64}([1. 0. ; 0 1]) 
σ_x =  SMatrix{2,2,ComplexF64}([0 1; 1 0]) 
σ_y =  SMatrix{2,2,ComplexF64}([0 -im ; im 0 ])
σ_z = SMatrix{2,2,ComplexF64}([1 0. ; 0. -1])
### Elementaty constants
const MBOHR = 5.788381e-5         ### Bohrs magneton
const KB = 8.6173324e-5           ### Bolzmann factor
const GAMMA_R = 1.760859644e-4    ### Gyromagnetic ratio ##1 for units of G_r=1
const hbar = 0.658211928e0 # (eV*fs)
#0.6582119569
const k_Boltzmann = 0.00008617343 ;#(eV/K) # 1.3806504d-23 (J/K)
#Here all the parameters that depends on global_parameters are iniliallized
I_a1b1 = Matrix{ComplexF64}(I, n, n)                                        ### One in the Hilbert space of the lattice sites
const I_ab = Matrix{ComplexF64}(I,
       n_channels*n, n_channels*n)                                          ### One in the Hilbert of lattice ⊗ spin
const σ_x1ab = kron(I_a1b1,  σ_x)                                                 ### Pauli Matrices in the Hilbert space of the lattices 
const σ_x2ab = kron(I_a1b1,  σ_y) 
const σ_x3ab = kron(I_a1b1,  σ_z) 
const σ_abx = cat(σ_x1ab,σ_x2ab
        , σ_x3ab, dims= (3) )                                               ### Tensor with all the pauli Matrices
### ----- fiting parameters of the lorentzian functions ---- ###
eps_L = copy(data_fit_pdbest[1:2:end])                                      ### Resonant level
eps_R = copy(eps_L)
w0_L  = abs.(data_fit_pdbest[2:2:end])                                      ### Level witdth
w0_R  = copy(w0_L)
const w0_k1α = cat(w0_L,w0_R,dims=2)                                        ### Tensor definition of the fitting parameters
const eps_k1α = cat(eps_L,eps_R,dims=2)
gam_L = zeros(Float64, n_lorentz,n_channels)                                ### Gamma function
gam_R = zeros(Float64,n_lorentz,n_channels)
gam_L[:,1] = data_fit_Ulsq
gam_L[:,2] = data_fit_Ulsq
gam_R = copy(gam_L)
const gam_k1iα = cat(gam_L,gam_R,dims=3)                                    ### Tensor definition of the gamma function 
#ν_αmik2 = zeros(ComplexF64,2,2,2,N_poles)
### ----- Oaki parameters ---- ###
const k_poles = N_poles + n_lorentz                                         ### total number of poles
##R_alpha_L = zeros(2*N_poles)                                                ### Matrix used to calculate the poles properties  
##R_alpha_R = zeros(2*N_poles)
#R_k2α = zeros(N_poles,2)
const energy_llg = 0.5*(E_F_left + E_F_right)
### ----- Lead parameters ---- ###
#csi_L = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
#csi_R = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
#csi_aikα = cat(csi_L, csi_R, dims = 4)
const E_F_α = Float64[E_F_left, E_F_right]
const beta = 1/(k_Boltzmann*Temp )
if read_bias_file                                                              ### Time dependent pulse in the lead
    ### In this case the delta_\alpha must join as a parameter of the hamiltonian 
    const data_bias = readdlm(bias_file, ' ', Float64 )
    const ti_bias = size(data_bias)[1]#100                                                       ### steps to turn off the bias 
else
    const delta_α = [0., 0.]#[delta_tdep_L,delta_tdep_R]  
end
# hi_αmk = zeros(ComplexF64, 2, 2,
#         k_poles )                                                          ### χ Value that contain the poles of the lorentzians and fermi function
# hi_αmk1 = zeros(ComplexF64, 2, 2,
#         n_lorentz )   
# hi_αmk2 = zeros(ComplexF64, 2, 2,
#         N_poles ) 
# fermi_m_αmk1 = zeros(ComplexF64,2,
#     2,n_lorentz)
# fermi_p_αmk1 = zeros(ComplexF64,2,2
#     ,n_lorentz)
# Gam_greater_αmik1 =  zeros(ComplexF64,2, 2,n_channels, n_lorentz )       ### Gamma Matrix initiallizations
# Gam_greater_αmik2 = zeros(ComplexF64,2, 2,n_channels, N_poles )
# Gam_greater_αmk = zeros(ComplexF64,2, 2,n_channels, k_poles )
# Gam_lesser_αmik1 =  zeros(ComplexF64,2, 2,n_channels, n_lorentz )
# Gam_lesser_αmik2 = zeros(ComplexF64,2, 2,n_channels, N_poles )
# Gam_lesser_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
# Gam_greater_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
### ----- variables involved in the dynamics ---- ###
# Omega_αβipjk = zeros(ComplexF64,2,2,n_channels,k_poles,n_channels,k_poles)
# dOmega_αβipjk = zeros(ComplexF64,2,2,n_channels,k_poles,n_channels,k_poles)
# Omega_αikβjp = zeros(ComplexF64,2,n_channels,k_poles,2 ,n_channels,k_poles)
# Omega_αik1βjp1 = zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,n_lorentz)
# Omega_αik1βjp2 = zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,N_poles)
# Omega_αik2βjp1= zeros(ComplexF64,2,n_channels,N_poles,2 ,n_channels,n_lorentz)
dOmega_αikβjp = zeros(ComplexF64,2,n_channels,k_poles,2 ,n_channels,k_poles)
dOmega_αik1βjp1 = zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,n_lorentz)
dOmega_αik1βjp2 = zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,N_poles)
dOmega_αik2βjp1= zeros(ComplexF64,2,n_channels,N_poles,2 ,n_channels,n_lorentz)
summ1_aik1α = zeros(ComplexF64,2*n,2,n_lorentz,2)
summ2_aik1α = zeros(ComplexF64,2*n,2,n_lorentz,2)
summ3_aik2α = zeros(ComplexF64,2*n,2,N_poles,2)
# psi_aikα = zeros(ComplexF64, 2*n, n_channels, k_poles, 2 )
dpsi_aikα = zeros(ComplexF64, 2*n, n_channels, k_poles, 2 )
# rho_ab = zeros(ComplexF64, 2*n, 2*n )
drho_ab = zeros(ComplexF64, 2*n, 2*n )
#Pi_abα = zeros(ComplexF64, 2*n, 2*n, 2 ) 
const dims_Omega1 = (2,n_channels,n_lorentz,2 ,n_channels,n_lorentz)
const dims_Omega2 = (2,n_channels,n_lorentz,2 ,n_channels,N_poles)
const dims_Omega3 = (2,n_channels,N_poles,2 ,n_channels,n_lorentz)
# size_Omega = prod(dims_Omega)
const size_Omega1 = prod(dims_Omega1)
const size_Omega2 = prod(dims_Omega2)
const size_Omega3 = prod(dims_Omega3)
const dims_psi = (2*n, 2, k_poles, 2)
const size_psi = prod(dims_psi)
const dims_rho = (2*n, 2*n)
const size_rho = prod(dims_rho)
#js_sd_to_el = zeros(Float64, n) ;
J_sd_local = J_sd.*ones(n) 
thop_local = thop.*ones(n-1) 
tso_local = alpha_r.*ones(n-1) 
const size_rkvec = size_Omega1+size_Omega2+size_Omega3+size_psi+size_rho 
#vec(Omega_αik1βjp1),vec(Omega_αik1βjp2),vec(Omega_αik2βjp1), vec(psi_aikα), vec(rho_ab)
#rkvec = zeros(ComplexF64, size_rkvec)#to_vector(dOmega_αβipjk, dpsi_aikα, drho_ab )
const params_0 = Dict("curr"=>curr, "scurr"=>scurr, "sden"=>sden, "rho"=>rho);    ### Observables calculated at each time step
# sden_xa1 = zeros(Float64, 3, n);                                              ### spin density
# curr_α = zeros(ComplexF64,2)
# scurr_xα = zeros(ComplexF64, 3,2)
# sden_xab = zeros(ComplexF64, 3,2*n,2*n) ;







end