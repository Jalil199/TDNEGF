### In this module all the parameters are readed and the secondary parameters are defined 
module get_parameters
### Libraries
include("read_parameters.jl")
import .read_parameters: read_params
using LinearAlgebra          ### Linear algebra library
using StaticArrays           ### Small matrices 
using DelimitedFiles         ### Manipulate files 
loaded_parameters = read_params()
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

# Access and assign values to variables from the loaded parameters
const n = get(loaded_parameters, "n", 0)
const n_lorentz = get(loaded_parameters, "n_lorentz", 31)
const n_channels = get(loaded_parameters, "n_channels", 2)
const delta_tdep_L = get(loaded_parameters, "delta_tdep_L", 0.0)
const delta_tdep_R = get(loaded_parameters, "delta_tdep_R", 0.0)
const E_F_system = get(loaded_parameters, "E_F_system", 0.0)
const V_bias = get(loaded_parameters, "V_bias", 0.0)
const alpha_r = get(loaded_parameters, "alpha_r", 0.0)
const theta_1 = get(loaded_parameters, "theta_1", 0.0)
const phi_1 = get(loaded_parameters, "phi_1", 0.0)
const theta_2 = get(loaded_parameters, "theta_2", 0.0)
const phi_2 = get(loaded_parameters, "phi_2", 0.0)
const period = get(loaded_parameters, "period", 0)
const N_rash = get(loaded_parameters, "N_rash", 0)
const Temp = get(loaded_parameters, "Temp", 0.0)
const N_poles = get(loaded_parameters, "N_poles", 0)
const t_0 = get(loaded_parameters, "t_0", 0.0)
const t_step = get(loaded_parameters, "t_step", 0.0)
const t_end = get(loaded_parameters, "t_end", 0.0)
const J_sd = get(loaded_parameters, "J_sd", 0.0)
const n_precessing = get(loaded_parameters, "n_precessing", 0)
const J_qsl = get(loaded_parameters, "J_qsl", 0.0)
const n_sites = get(loaded_parameters, "n_sites", 1)
const nt = get(loaded_parameters, "nt", 1)
const dt = get(loaded_parameters, "dt", 0.1)
const h0 = get(loaded_parameters, "h0", [0.0, 0.0, 0.0])
const j_exc = get(loaded_parameters, "j_exc", 0.0)
const g_lambda = get(loaded_parameters, "g_lambda", 0.0)
const j_sd = get(loaded_parameters, "j_sd", 0.0)
const j_dmi = get(loaded_parameters, "j_dmi", 0.0)
const j_ani = get(loaded_parameters, "j_ani", 0.0)
const j_dem = get(loaded_parameters, "j_dem", 0.0)
const e = get(loaded_parameters, "e", [0.0, 0.0, 0.0])
const e_demag = get(loaded_parameters, "e_demag", [0.0, 0.0, 0.0])
const js_pol = get(loaded_parameters, "js_pol", 0.0)
const js_ana = get(loaded_parameters, "js_ana", 0.0)
const thop = get(loaded_parameters, "thop", 1.0)
const p_theta = get(loaded_parameters, "p_theta", 0.0)
const p_phi = get(loaded_parameters, "p_phi", 0.0)
const run_llg = get(loaded_parameters, "run_llg", false)
const curr = get(loaded_parameters, "curr", true)
const scurr = get(loaded_parameters, "scurr", true)
const rho = get(loaded_parameters, "rho", true)
const sden = get(loaded_parameters, "sden", true)
const solver = get(loaded_parameters, "solver", "denis")
const cspin_orientation = get(loaded_parameters, "cspin_orientation", "sym_pump")
const bias_file = get(loaded_parameters, "bias_file", "./vtd.txt")
const read_bias_file = get(loaded_parameters, "read_bias_file", "false")
const name = get(loaded_parameters, "name", "test_sym_pump")

#### Derived parameters 
##############################################################################################################
const E_F_left = E_F_system + 0.5 * V_bias
const E_F_right = E_F_system - 0.5 * V_bias
const save_data = Dict("curr" => true, "scurr" => true, "sden_eq" => true, "sden_neq" => true, "rho" => true, "sclas" => true)
const save_data_qsl = Dict("curr" => true, "scurr" => true, "sden_eq" => true, "sden_neq" => true,
                     "rho" => true, "sclas" => true, "ent" => true, "sden_qsl" => true)
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