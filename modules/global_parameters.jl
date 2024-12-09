module global_parameters
""" Functions related with the global parameter
"""

using LinearAlgebra          ### Linear algebra library
using DelimitedFiles         ### Manipulate files
using Tullio

function read_params(archivo_parametros::String)#;archivo_parametros=archivo_parametros)
    """ Stores the parameters of a txt file into a dictionary 
    """
    parametros = Dict{String, Any}()
    open(archivo_parametros) do file
        for linea in eachline(file)
            # Check if the line is a comment (starts with ##) and skip it
            # Split the line by '##' and take only the part before '##'
            parts = split(linea, "#")
            cleaned_line = strip(parts[1])
            isempty(cleaned_line) && continue
            # Dividir la línea en nombre y valor utilizando una coma como separador si está presente
            # Si no hay comas, proceder como antes
            # Dividir la línea en nombre y valor
            parts = split(cleaned_line, "=")
            if length(parts) == 2
                nombre = strip(parts[1])
                valor = strip(parts[2])
             # Intentar convertir el valor a un tipo numérico o cadena
               if occursin(",", valor)
                    valores = split(valor, ",")
                    #println("valores", valores[1])
                    # Intentar convertir los elementos en números de punto flotante
                    try
                        valores = Int64[parse(Int64, v) for v in valores]
                    catch
                        try 
                            ### try to conver into a an array of Float pointers
                            valores = Float64[parse(Float64, v) for v in valores]
                        catch
                            # Si no se pueden convertir, se mantienen como cadena
                        end
                    end
                    # Almacenar el parámetro en el diccionario
                    parametros[nombre] = valores
                else
                    try
                        # Si sabes que el valor es un entero, usa parse(Int, valor)
                        valor = parse(Int, valor)
                    catch
                        try
                            # Si sabes que el valor es un número de punto flotante, usa parse(Float64, valor)
                            valor = parse(Float64, valor)
                        catch
                            try 
                                valor = parse(Bool, valor)
                            catch
                                valor = strip(valor, '"')  # Eliminar comillas si es una cadena
                            end
                        end
                    end
                    # Almacenar el parámetro en el diccionario
                    parametros[nombre] = valor
                end
            end
        end
    end
    parametros
end



Base.@kwdef mutable struct PrecSpin   
""" This mutable structure act like a class 
in python, it defines an object with the 
characteristics of a precessing spin
"""
    i::Int64 =0#   
    axis_phi::Float64 = 0.0
    axis_theta::Float64 = 0.0
    phi_zero::Float64 = 0.0
    theta_zero::Float64 = 0.0
    start_time::Float64 = 0.0
    T::Float64 = 1.
    s::Vector{Float64} = [0.,0.,1.] 
end


function global_params(name_file::String)
    """ Store the global parameters in different structures 
    """
    ### We store the parameters from the txt file into a dictionary
    loaded_parameters = read_params(name_file)
    ### Elementary constants
    MBOHR = 5.788381e-5         ### Bohrs magneton
    KB = 8.6173324e-5           ### Bolzmann factor
    GAMMA_R = 1.760859644e-4    ### Gyromagnetic ratio ##1 for units of G_r=1
    hbar = 0.658211928e0 # (eV*fs)
    #0.6582119569
    k_Boltzman = 0.00008617343 ;#(eV/K) # 1.3806504d-23 (J/K)
    ### Now we read all the parameters to be used 
    # Access and assign values to variables from the loaded parameters
    n = get(loaded_parameters, "n", 0)
    n_lorentz = get(loaded_parameters, "n_lorentz", 31)
    n_channels = get(loaded_parameters, "n_channels", 2)
    delta_tdep_L = get(loaded_parameters, "delta_tdep_L", 0.0)
    delta_tdep_R = get(loaded_parameters, "delta_tdep_R", 0.0)
    E_F_system = get(loaded_parameters, "E_F_system", 0.0)
    V_bias = get(loaded_parameters, "V_bias", 0.0)
    alpha_r = get(loaded_parameters, "alpha_r", 0.0)
    theta_1 = get(loaded_parameters, "theta_1", 0.0)
    phi_1 = get(loaded_parameters, "phi_1", 0.0)
    theta_2 = get(loaded_parameters, "theta_2", 0.0)
    phi_2 = get(loaded_parameters, "phi_2", 0.0)
    period = get(loaded_parameters, "period", 0)
    N_rash = get(loaded_parameters, "N_rash", 0)
    Temp = get(loaded_parameters, "Temp", 0.0)
    N_poles = get(loaded_parameters, "N_poles", 0)
    t_0 = get(loaded_parameters, "t_0", 0.0)
    t_step = get(loaded_parameters, "t_step", 0.0)
    t_end = get(loaded_parameters, "t_end", 0.0)
    J_sd = get(loaded_parameters, "J_sd", 0.0)
    θ = get(loaded_parameters, "θ", pi)
    A = get(loaded_parameters, "A", [0., 0. , 0.])
    n_precessing = get(loaded_parameters, "n_precessing", 0)
    J_qsl = get(loaded_parameters, "J_qsl", 0.0)
    n_sites = get(loaded_parameters, "n_sites", 1)
    nt = get(loaded_parameters, "nt", 1)
    dt = get(loaded_parameters, "dt", 0.1)
    h0 = get(loaded_parameters, "h0", [0.0, 0.0, 0.0])
    j_exc = get(loaded_parameters, "j_exc", 0.0)
    g_lambda = get(loaded_parameters,"g_lambda", 0.0)
    j_sd = get(loaded_parameters, "j_sd", 0.0)
    j_dmi = get(loaded_parameters, "j_dmi", 0.0)
    j_ani = get(loaded_parameters, "j_ani", 0.0)
    j_dem = get(loaded_parameters, "j_dem", 0.0)
    e = get(loaded_parameters, "e", [0.0, 0.0, 0.0])
    e_demag = get(loaded_parameters, "e_demag", [0.0, 0.0, 0.0])
    js_pol = get(loaded_parameters, "js_pol", 0.0)
    js_ana = get(loaded_parameters, "js_ana", 0.0)
    thop = get(loaded_parameters, "thop", 1.0)
    p_theta = get(loaded_parameters, "p_theta", 0.0)
    p_phi = get(loaded_parameters, "p_phi", 0.0)
    run_llg = get(loaded_parameters, "run_llg", false)
    curr = get(loaded_parameters, "curr", true)
    scurr = get(loaded_parameters, "scurr", true)
    rho = get(loaded_parameters, "rho", true)
    sden_eq = get(loaded_parameters, "sden_eq", true)
    sden = get(loaded_parameters, "sden", true)
    sden_neq = get(loaded_parameters, "sden_neq", true)
    sden_qsl = get(loaded_parameters, "sden_qsl", true)
    sclas = get(loaded_parameters, "sclas", true)
    cden = get(loaded_parameters, "cden", true)
    bcurrs = get(loaded_parameters, "bcurrs", true)
    ent = get(loaded_parameters, "ent", true)
    solver = get(loaded_parameters, "solver", "denis")
    cspin_orientation = get(loaded_parameters, "cspin_orientation", "arb_dir")
    bias_file = get(loaded_parameters, "bias_file", "./vtd.txt")
    read_bias_file = get(loaded_parameters, "read_bias_file", false)
    name = get(loaded_parameters, "name", "test_sym_pump")
    preload_rkvec = get(loaded_parameters, "preload_rkvec", true)
    name_preload_rkvec = get(loaded_parameters,"name_preload_rkvec", "rkvec_test_jl.txt" )

    #### Derived parameters 
    ##############################################################################################################
    E_F_left = E_F_system + 0.5 * V_bias
    E_F_right = E_F_system - 0.5 * V_bias
    save_data = Dict("curr" => curr, "scurr" => scurr, "sden_eq" => sden_eq, "sden_neq" => sden_neq,
                            "rho" => rho, "sclas" => sclas, "cden" =>cden, "bcurrs"=> bcurrs)
    #params_sden = Dict("curr"=>false, "scurr"=>false, "sden"=>true, "cden" =>false, "bcurrs" =>false);
    params  = Dict("curr" => curr, "scurr" => scurr, "sden" => sden, "cden" =>cden, "bcurrs"=> bcurrs)
    # const save_data_qsl = Dict("curr" => curr, "scurr" => scurr, "sden_eq" => sden_eq, "sden_neq" => sden_neq,
    #                      "rho" => rho, "sclas" => sclas, "ent" => ent, "sden_qsl" => sden_qsl)
    data_fit_pdbest = readdlm( "./selfenergy/selfenergy_1DTB_NNLS_31_pbest.csv" , ',', Float64)
    data_fit_Ulsq = readdlm( "./selfenergy/selfenergy_1DTB_NNLS_31_Ulsq.csv", ',', Float64) ;
                                           ### Tensor with all the pauli Matrices
    ### ----- fiting parameters of the lorentzian functions ---- ###
    eps_L = copy(data_fit_pdbest[1:2:end])                                      ### Resonant level
    eps_R = copy(eps_L)
    w0_L  = abs.(data_fit_pdbest[2:2:end])                                      ### Level witdth
    w0_R  = copy(w0_L)
    w0_k1α = cat(w0_L,w0_R,dims=2)                                        ### Tensor definition of the fitting parameters
    eps_k1α = cat(eps_L,eps_R,dims=2)
    gam_L = zeros(Float64, n_lorentz,n_channels)                                ### Gamma function
    gam_R = zeros(Float64,n_lorentz,n_channels)
    gam_L[:,1] = data_fit_Ulsq
    gam_L[:,2] = data_fit_Ulsq
    gam_R = copy(gam_L)
    gam_k1iα = cat(gam_L,gam_R,dims=3)                                    ### Tensor definition of the gamma function 
    ### ----- Ozaki parameters ---- ###
    k_poles = N_poles + n_lorentz                                         ### total number of poles
    energy_llg = 0.5*(E_F_left + E_F_right)
    ### ----- Lead parameters ---- ###
    E_F_α = Float64[E_F_left, E_F_right]
    beta = 1/(k_Boltzman*Temp )
    csi_L = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
    csi_R = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )

    ####
    σ_0 = Matrix{ComplexF64}([1. 0. ; 0 1]) 
    σ_x =  Matrix{ComplexF64}([0 1; 1 0]) 
    σ_y =  Matrix{ComplexF64}([0 -im ; im 0 ])
    σ_z = Matrix{ComplexF64}([1 0. ; 0. -1])

    #Here all the parameters that depends on global_parameters are iniliallized
    I_a1b1 = Matrix{ComplexF64}(I, n, n)                                        ### One in the Hilbert space of the lattice sites
    I_ab = Matrix{ComplexF64}(I,
           n_channels*n, n_channels*n)                                          ### One in the Hilbert of lattice ⊗ spin
    σ_x1ab = kron(I_a1b1,  σ_x)                                                 ### Pauli Matrices in the Hilbert space of the lattices 
    σ_x2ab = kron(I_a1b1,  σ_y) 
    σ_x3ab = kron(I_a1b1,  σ_z) 
    σ_abx  = cat(σ_x1ab,σ_x2ab
            , σ_x3ab, dims= (3) )    
    ##################### Fixed variables in the dynamics
    dims_Omega1 = (2,n_channels,n_lorentz,2 ,n_channels,n_lorentz)
    dims_Omega2 = (2,n_channels,n_lorentz,2 ,n_channels,N_poles)
    dims_Omega3 = (2,n_channels,N_poles,2 ,n_channels,n_lorentz)
    # size_Omega = prod(dims_Omega)
    size_Omega1 = prod(dims_Omega1)
    size_Omega2 = prod(dims_Omega2)
    size_Omega3 = prod(dims_Omega3)
    dims_psi = (2*n, 2, k_poles, 2)
    size_psi = prod(dims_psi)
    dims_rho = (2*n, 2*n)
    size_rho = prod(dims_rho)
    size_rkvec = size_Omega1+size_Omega2+size_Omega3+size_psi+size_rho 


    ####

    #### Initiallize the structures (global_var)
    global_=global_var(
        n = n, N_rash = N_rash, alpha_r = alpha_r, J_sd = J_sd, thop = thop, theta_1=theta_1, phi_1=phi_1,phi_2=phi_2,
        n_lorentz = n_lorentz, n_channels = n_channels, delta_tdep_L = delta_tdep_L, delta_tdep_R = delta_tdep_R,
        E_F_system = E_F_system, E_F_α = E_F_α, V_bias = V_bias, Temp = Temp, N_poles = N_poles,
        t_0 = t_0, t_step = t_step, t_end = t_end ,save_data=save_data,
        θ = θ, A = A, J_qsl = J_qsl, beta = beta, k_poles = k_poles,
        n_sites = n_sites, h0 = h0, j_exc = j_exc, g_lambda = g_lambda,
        j_sd = j_sd, j_dmi = j_dmi, j_ani = j_ani, j_dem = j_dem,
        e = e, js_pol = js_pol, js_ana = js_ana, p_theta = p_theta,
        p_phi = p_phi, n_precessing = n_precessing, period = period,
        run_llg = run_llg, nt = nt, dt = dt, cspin_orientation = cspin_orientation,
        curr = curr, scurr = scurr, rho = rho, sden_eq = sden_eq,
        sden = sden, sden_neq = sden_neq, sden_qsl = sden_qsl, sclas = sclas,
        ent = ent, solver = solver, bias_file = bias_file, read_bias_file = read_bias_file,
        name = name, preload_rkvec = preload_rkvec, name_preload_rkvec = name_preload_rkvec,
        params = params,
        gam_k1iα = gam_k1iα, w0_k1α = w0_k1α, eps_k1α = eps_k1α,
        MBOHR = MBOHR, KB = KB, GAMMA_R = GAMMA_R, hbar = hbar, k_Boltzman = k_Boltzman,
        σ_0 = σ_0, σ_x = σ_x, σ_y = σ_y, σ_z = σ_z, I_a1b1 = I_a1b1, I_ab = I_ab,
        σ_x1ab = σ_x1ab, σ_x2ab = σ_x2ab, σ_x3ab = σ_x3ab, σ_abx = σ_abx,
        dims_Omega1 = dims_Omega1, dims_Omega2 = dims_Omega2, dims_Omega3 = dims_Omega3,
        dims_psi = dims_psi, dims_rho = dims_rho, size_Omega1 = size_Omega1,
        size_Omega2 = size_Omega2, size_Omega3 = size_Omega3, size_psi = size_psi,
        size_rho = size_rho, size_rkvec = size_rkvec)

    rkvec = zeros(ComplexF64, size_rkvec)
    pr_spins = [PrecSpin(i=u) for u in 1:1:n_precessing]
    vm_a1x = [zeros(Float64, 3) for _ in 1:n]
    sm_eq_a1x = [zeros(Float64, 3) for _ in 1:n]
    diff = [zeros(Float64, 3) for _ in 1:n]
    delta_αi = zeros(Float64, 2, 2)
    H_ab::Array{ComplexF64,2} = zeros(ComplexF64, 2*n, 2*n )
    Omega_αik1βjp1::Array{ComplexF64,6} = Array{ComplexF64}(undef,dims_Omega1)
    Omega_αik1βjp2::Array{ComplexF64,6} = Array{ComplexF64}(undef,dims_Omega2)
    Omega_αik2βjp1::Array{ComplexF64,6} = Array{ComplexF64}(undef,dims_Omega3)
    psi_aikα::Array{ComplexF64,4} = Array{ComplexF64}(undef,dims_psi)
    rho_ab::Array{ComplexF64,2} = Array{ComplexF64}(undef,dims_rho)
    Pi_abα::Array{ComplexF64,3} = zeros(ComplexF64, 2*n, 2*n, 2 )
    ### Variables that should be calculated and later used as a constants
    hi_αmk = zeros(ComplexF64, 2, 2, k_poles )                                      ### χ Value that contain the poles of the lorentzians and fermi function
    hi_αmk1 = zeros(ComplexF64, 2, 2, n_lorentz )   
    hi_αmk2 = zeros(ComplexF64, 2, 2, N_poles ) 
    Gam_greater_αmik1 =  zeros(ComplexF64,2, 2,n_channels, n_lorentz )       ### Gamma Matrix initiallizations
    Gam_greater_αmik2 = zeros(ComplexF64,2, 2,n_channels, N_poles )
    Gam_greater_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
    Gam_lesser_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
    csi_aikα = cat(csi_L, csi_R, dims = 4)

    # Eig_vals, Res_p = get_poles(n_channels*N_poles)
    # Eig_vals_k2α = cat(Eig_vals,Eig_vals,dims=2)
    # R_k2α = cat(Res_p,Res_p,dims=2) 
    # hi_αmk,hi_αmk1,hi_αmk2 = create_hi( Eig_vals_k2α = Eig_vals_k2α)
    
    dynamics = dynamics_var(hbar=hbar,n=n,n_lorentz=n_lorentz,N_poles=N_poles,dims_Omega1=dims_Omega1, dims_Omega2=dims_Omega2, dims_Omega3=dims_Omega3,
                        size_Omega1=size_Omega1, size_Omega2=size_Omega2, size_Omega3=size_Omega3,
                        dims_psi=dims_psi, size_psi=size_psi, dims_rho=dims_rho, size_rho=size_rho,
                        size_rkvec=size_rkvec, rkvec=rkvec, pr_spins=pr_spins, vm_a1x=vm_a1x,
                        sm_eq_a1x=sm_eq_a1x, diff=diff, delta_αi=delta_αi, H_ab=H_ab,
                        Omega_αik1βjp1=Omega_αik1βjp1, Omega_αik1βjp2=Omega_αik1βjp2,
                        Omega_αik2βjp1=Omega_αik2βjp1, psi_aikα=psi_aikα, rho_ab=rho_ab,
                        Pi_abα=Pi_abα, hi_αmk=hi_αmk, hi_αmk1=hi_αmk1, hi_αmk2=hi_αmk2,
                        Gam_greater_αmik1=Gam_greater_αmik1, Gam_greater_αmik2=Gam_greater_αmik2,
                        Gam_greater_αmik=Gam_greater_αmik,Gam_lesser_αmik=Gam_lesser_αmik, csi_aikα=csi_aikα)

    js_exc::Vector{Float64}  = ones(Float64, n_sites-1)*j_exc
    js_sd::Vector{Float64}  = ones(Float64, n_sites)*j_sd
    js_ani::Vector{Float64}  = ones(Float64, n_sites)*j_ani
    js_dem::Vector{Float64}  = ones(Float64, n_sites)*j_dem
    thops::Vector{ComplexF64} = thop.*ones(ComplexF64,n-1)
    Js_sd::Vector{Float64}  = ones(Float64, n_sites)*J_sd
    thops_so::Vector{ComplexF64} = alpha_r.*ones(ComplexF64,n-1)
    
    config_ = config_var(e=e, e_demag=e_demag, js_exc=js_exc, js_sd=js_sd, js_ani=js_ani, js_dem=js_dem,
                        p_theta=p_theta, p_phi=p_phi, thop=thop, Js_sd=Js_sd,
                        thops=thops, thops_so=thops_so)
    poles_denis =  zeros(ComplexF64,2)
    res_denis =  zeros(ComplexF64,2)
    eq_ = eq_var(n=n, n_channels=n_channels, mu=E_F_system, temp=Temp,poles_denis=poles_denis,res_denis=res_denis)
    #### Initiallize the structures (observables)
    sden_xa1::Array{Float64,2} = zeros(Float64, 3, n)
    curr_α::Vector{Float64} = zeros(Float64,2)
    scurr_xα::Array{Float64,2} = zeros(Float64, 3,2)
    sden_xab::Array{ComplexF64,3} = zeros(ComplexF64, 3,2*n,2*n)
    cden_a1::Array{Float64} = zeros(Float64,n)
    # cc::Array{Float64}  = zeros(Float64, n-1)
    # cx::Array{Float64} = zeros(Float64, n-1)
    # cy::Array{Float64} = zeros(Float64, n-1)
    # cz::Array{Float64} = zeros(Float64, n-1)
    bcurrs_a1::Vector{Vector{Float64}} = [zeros(Float64, n-1) for _ in 1:1:4 ]#::Array{Float64} = zeros(Float64, 4*(n-1) ) # [zeros(Float64, n-1),zeros(Float64, n-1),zeros(Float64, n-1), zeros(Float64, n-1)]
    vsden_xa1::Vector{Vector{Float64}} = [sden_xa1[:, i] for i in 1:n :: Int]
    #[ zeros(Float64,3) for _ in n]
    #observables = observables_var(sden_xa1=sden_xa1,curr_α=curr_α,scurr_xα=scurr_xα,sden_xab=sden_xab,cden=cden_a1,bcurrs=bcurrs)
    observables = observables_var(sden_xa1,curr_α,scurr_xα,sden_xab,cden_a1,vsden_xa1,bcurrs_a1)
    #### Initiallize the structures ()
    llg_params =  llg_parameters(n_sites,nt,dt,h0,j_exc,g_lambda,j_sd,
                  j_dmi,j_ani,j_dem,js_pol,js_ana,thop,e,e_demag,js_exc,
                  js_sd,js_ani,js_dem,p_theta,p_phi) 
    return   global_ ,dynamics,observables, llg_params, config_, eq_
    # observables_var(sden_xa1=zeros(Float64, 3, n),curr_α = zeros(Float64,2),scurr_xα=zeros(Float64, 3,2),sden_xab=zeros(ComplexF64, 3,2*n,2*n), cden=zeros(Float64,n),bcurrs = zeros(Float64, 4*(n-1) )  )
end


############################## Inmutable structures that stores the parameters
Base.@kwdef struct global_var
    """ All the global variables are stored in case all are needed
    """
    
    ####################### system-lead parameters
    n::Int
    N_rash::Int
    alpha_r::Float64
    J_sd::Float64
    thop::Float64
    ### lead parameters
    n_lorentz::Int
    n_channels::Int
    delta_tdep_L::Float64
    delta_tdep_R::Float64
    E_F_system::Float64
    E_F_α::Array{Float64}
    V_bias::Float64
    Temp::Float64
    N_poles::Int
    ### Evolution
    t_0::Float64
    t_step::Float64
    t_end::Float64
    ### Spin Liquid
    θ::Float64
    A::Vector{Float64}
    J_qsl::Float64
    beta::Float64
    k_poles::Int
    ######################## LLG
    n_sites::Int
    h0::Vector{Float64}
    j_exc::Float64
    g_lambda::Float64
    j_sd::Float64
    j_dmi::Float64
    j_ani::Float64
    j_dem::Float64
    e::Vector{Float64}
    js_pol::Float64
    js_ana::Float64
    p_theta::Float64
    p_phi::Float64
    ### Precession
    theta_1::Float64
    phi_1::Float64
    phi_2::Float64
    n_precessing::Int
    period::Float64
    ### Evolution LLG
    run_llg::Bool
    nt::Int
    dt::Float64
    cspin_orientation::String#SubString{String}#String
    ######################## Store data and file manipulations 
    curr::Bool
    scurr::Bool
    rho::Bool
    sden_eq::Bool
    sden::Bool
    sden_neq::Bool
    sden_qsl::Bool
    sclas::Bool
    ent::Bool
    solver::String#Bool
    bias_file::String#SubString{String}#String
    read_bias_file::Bool
    name::String#SubString{String}#String
    preload_rkvec::Bool
    name_preload_rkvec::String#SubString{String}#String
    params::Dict{String, Bool}
    save_data::Dict{String, Bool}
    ###################### Cosntants 
    ### Pre calculated data
    gam_k1iα#::Array{Float64}
    w0_k1α#::Array{Float64}
    eps_k1α# ::Array{Float64}
    ### Elementaty constants
    MBOHR::Float64
    KB::Float64
    GAMMA_R::Float64
    hbar::Float64
    k_Boltzman::Float64
    ### Elementary matrices
    σ_0::Matrix{ComplexF64}
    σ_x::Matrix{ComplexF64}
    σ_y::Matrix{ComplexF64}
    σ_z::Matrix{ComplexF64}
    I_a1b1::Matrix{ComplexF64}
    I_ab::Matrix{ComplexF64}
    σ_x1ab::Array{ComplexF64}
    σ_x2ab::Array{ComplexF64}
    σ_x3ab::Array{ComplexF64}
    σ_abx::Array{ComplexF64}
    #Here all the parameters that depends on global_parameters are iniliallized
    ##################### Fixed variables in the dynamics
    dims_Omega1::Tuple{Int, Int, Int, Int, Int, Int}
    dims_Omega2::Tuple{Int, Int, Int, Int, Int, Int}
    dims_Omega3::Tuple{Int, Int, Int, Int, Int, Int}
    dims_psi::Tuple{Int, Int, Int, Int}
    dims_rho::Tuple{Int, Int}
    size_Omega1::Int
    size_Omega2::Int
    size_Omega3::Int
    size_psi::Int
    size_rho::Int
    size_rkvec::Int
end


Base.@kwdef struct llg_parameters
""" Here are contained the main parameters of the LLG evolution
"""
    
    n_sites::Int = n_sites
    nt::Int = nt
    dt::Float64 =dt
    h0::Vector{Float64} = h0 
    j_exc::Float64
    g_lambda::Float64 = g_lambda
    j_sd::Float64 = j_sd
    j_dmi::Float64 = j_dmi
    j_ani::Float64 = j_ani
    j_dem::Float64 = j_dem
    js_pol::Float64 = js_pol
    js_ana::Float64 = js_ana
    thop::Float64 = thop
    e::Vector{Float64}  = e 
    e_demag::Vector{Float64}  = e_demag
    js_exc::Vector{Float64}  = ones(Float64, n_sites-1)*j_exc
    js_sd::Vector{Float64}  = ones(Float64, n_sites)*j_sd
    js_ani::Vector{Float64}  = ones(Float64, n_sites)*j_ani
    js_dem::Vector{Float64}  = ones(Float64, n_sites)*j_dem
    p_theta::Float64 = p_theta
    p_phi::Float64 = p_phi
end

############################## Mutable structures    

Base.@kwdef mutable struct config_var
    """ This structure stores all the variables that can be modified at each time step
    """
    
    ### LLG Variables that can be modified at each time step
    e::Vector{Float64}
    e_demag::Vector{Float64}
    js_exc::Vector{Float64}
    js_sd::Vector{Float64}
    js_ani::Vector{Float64}
    js_dem::Vector{Float64}
    p_theta::Float64
    p_phi::Float64
    
    ### Central Part parameters 
    thop::Float64
    Js_sd::Vector{Float64}
    thops::Vector{ComplexF64}
    thops_so::Vector{ComplexF64}
end

Base.@kwdef mutable struct observables_var
    sden_xa1::Array{Float64,2}
    curr_α::Vector{Float64}
    scurr_xα::Array{Float64,2}
    sden_xab::Array{ComplexF64,3}
    cden::Array{Float64}
    vsden_xa1::Vector{Vector{Float64}} #1::Array{Float64,2}
    # cc::Array{Float64}  = zeros(Float64, n-1)
    # cx::Array{Float64} = zeros(Float64, n-1)
    # cy::Array{Float64} = zeros(Float64, n-1)
    # cz::Array{Float64} = zeros(Float64, n-1)
    bcurrs::Vector{Vector{Float64}} #::Array{Float64} # [zeros(Float64, n-1),zeros(Float64, n-1),zeros(Float64, n-1), zeros(Float64, n-1)]
end


Base.@kwdef mutable struct dynamics_var
    """ This variables can be modified at each time step
    They can be some observables or some auxiliar matrices defined to 
    calculate some intermedia stuffs"""
    ######### Dimensions
    hbar::Float64
    n::Int
    n_lorentz::Int
    N_poles::Int
    dims_Omega1::Tuple{Int, Int, Int, Int, Int, Int}
    dims_Omega2::Tuple{Int, Int, Int, Int, Int, Int}
    dims_Omega3::Tuple{Int, Int, Int, Int, Int, Int}
    # size_Omega = prod(dims_Omega)
    size_Omega1::Int
    size_Omega2::Int
    size_Omega3::Int
    dims_psi::Tuple{Int, Int, Int, Int}
    size_psi::Int
    dims_rho::Tuple{Int, Int}
    size_rho::Int
    size_rkvec::Int
    #########
    rkvec::Array{ComplexF64,1}
    pr_spins::Array{PrecSpin}
    vm_a1x::Vector{Vector{Float64}}##::Vector{Array{Float64,1}}
    sm_eq_a1x::Vector{Vector{Float64}}##::Vector{Array{Float64,1}}
    diff::Vector{Vector{Float64}}#::Vector{Array{Float64,1}}
    delta_αi::Array{Float64,2}
    H_ab::Array{ComplexF64,2}
    Omega_αik1βjp1::Array{ComplexF64,6}
    Omega_αik1βjp2::Array{ComplexF64,6}
    Omega_αik2βjp1::Array{ComplexF64,6}
    psi_aikα::Array{ComplexF64,4}
    rho_ab::Array{ComplexF64,2}
    Pi_abα::Array{ComplexF64,3}
    hi_αmk::Array{ComplexF64,3}
    hi_αmk1::Array{ComplexF64,3}
    hi_αmk2::Array{ComplexF64,3}
    Gam_greater_αmik1::Array{ComplexF64,4}
    Gam_greater_αmik2::Array{ComplexF64,4}
    Gam_lesser_αmik::Array{ComplexF64,4}
    Gam_greater_αmik::Array{ComplexF64,4}
    csi_aikα::Array{ComplexF64,4}
end



                            
Base.@kwdef mutable struct eq_var
    """ Here all the variables related with the equilibrium parameters are stored
    """
    n::Int # = n
    n_channels::Int # = n_channels
    mu::Float64 # = E_F_system
    temp::Float64 # = Temp
    e_min::Float64 = -3.
    p::Int = 21
    KB::Float64 = 8.6173324e-5           ### Bolzmann factor
    temp_im::Float64 = 0.
    temp_re::Float64 = 0.
    mu_im::Float64 = 0.
    mu_re::Float64 = 0.
    n_re::Int = 1
    n_im::Int = 1
    ntot::Int = 1
    poles_denis::Array{ComplexF64} # = zeros(ComplexF64,ntot::Int)
    res_denis::Array{ComplexF64} # = zeros(ComplexF64, ntot::Int) 
end





# Base.@kwdef struct sl_var
#     """ Variables of the spin liquid
#     """
# end

### Mutable structures to dynamical variables
#return global_var, config_var, llg_parameters,eq_var, dynamics_var, observables_var




##################################### Useful functions to build the structures

function fermi(energy)
    """ This function evaluate the fermi energy at the temperature defined in the global 
    parameters
    """
   fermi = 1. / (1. + exp(energy  ) )
end

function spectraldensity(en, epsil,wid )
    """ Computes a lorentzian distribution 
    (Here Usually the resonance and the width are fitted previously )
    args:
    ----
    en : Float64
    variable of energy where the distribution is evaluated
    epsil:Float64
    center of the lorentzian 
    wid: Float64
    width of the lorentzian 

    returns:
    -------
    lorentz: Float64
    Lorentzian distribution evaluated in energy en
    """
    lorentz = wid^2/((en-epsil)^2 + wid^2 )
    return lorentz
end
##(;eps_k1α = eps_k1α , w0_k1α = w0_k1α, E_F_α = E_F_α, Eig_vals_k2α = Eig_vals_k2α)
function create_hi(global_var; Eig_vals_k2α )
    """ This function creates the variable χ_{α,m,k}. This quantitie is related with the poles of the lorentzia
    paramaters:
    -------------------
    Note that the parameters are related with the evaluation of the poles 
    The first 2 parameters correspons to the elements associated with the resonance and the width of the 
    lorentzian function, the third element contain the fermi energy of the leads and the fourth element 
    contain the poles of the ozaki function 
    """
    hi_αmk = zeros(ComplexF64, 2, 2, global_var.k_poles::Int ) 
    @tullio hi_αmk1[α, m ,k1 ] := global_var.eps_k1α[k1,α] + (-1)^m*global_var.w0_k1α[k1,α]*1im  (m in 1:2)
    @tullio hi_αmk2[α, m ,k2 ] := global_var.E_F_α[α] + (-1)^m * 1im/(Eig_vals_k2α[k2,α]*global_var.beta)  (m in 1:2)
    hi_αmk[:, :, 1:global_var.n_lorentz] .= hi_αmk1
    hi_αmk[:,:,global_var.n_lorentz+1:global_var.k_poles] .= hi_αmk2

    return hi_αmk,hi_αmk1,hi_αmk2
end
#(;hi_αmk=hi_αmk,hi_αmk1=hi_αmk1,hi_αmk2=hi_αmk2,E_F_α = E_F_α, R_k2α= R_k2α,gam_k1iα=gam_k1iα,w0_k1α=w0_k1α,eps_k1α=eps_k1α)
function create_Gam(global_var;R_k2α,hi_αmk,hi_αmk1,hi_αmk2 )
    """ This function evaluates the Gamma function using the residue theorem for the 
    lorentzian and fermi poles(Ozaki decomposition) using the tullio
    library. Also is important to note that all the elements are evaluated using tensorial notation
    """
    Gam_lesser_αmik = zeros(ComplexF64,2, 2,global_var.n_channels, global_var.k_poles )
    Gam_greater_αmik = zeros(ComplexF64,2, 2,global_var.n_channels, global_var.k_poles )
    
    ### Fermi function 
    @tullio fermi_m_αmk1[α,m,k1] := fermi((-hi_αmk1[α,m,k1] + global_var.E_F_α[α])*global_var.beta )
    @tullio fermi_p_αmk1[α,m,k1] := fermi((hi_αmk1[α,m,k1] - global_var.E_F_α[α])*global_var.beta  )
    ### Quantity related with the spectral density 
    @tullio ν_αmik2[α,m,j,k2] := spectraldensity(hi_αmk2[α,m,k2],global_var.eps_k1α[l1,α],global_var.w0_k1α[l1,α] )*global_var.gam_k1iα[l1,j,α]
    ### Definition of Gamma greater for the poles of the fermi function and the lorentzian poles 
    @tullio Gam_greater_αmik1[α, m,i, k1 ] := -0.5im*global_var.gam_k1iα[k1,i,α]*global_var.w0_k1α[k1,α]*fermi_m_αmk1[α,m,k1] # Fermi function poles
    @tullio Gam_greater_αmik2[α, m,i, k2 ] := (-1)^(m)*ν_αmik2[α,m,i,k2]*R_k2α[k2,α]/global_var.beta # Lorentzian poles 
    Gam_greater_αmik[:,:,:,1:global_var.n_lorentz] = Gam_greater_αmik1
    Gam_greater_αmik[:,:,:,global_var.n_lorentz+1:global_var.k_poles] = Gam_greater_αmik2
    ### Definition of Gamma lesser for the poles of the fermi function and the lorentzian poles
    @tullio Gam_lesser_αmik1[α, m,i, k1 ] := 0.5im*global_var.gam_k1iα[k1,i,α]*global_var.w0_k1α[k1,α]*fermi_p_αmk1[α,m,k1]
    #@tullio Gam_lesser_αmik2[α, m,i, k2 ] = (-1)^(m)*ν_αmik2[α,m,i,k2]*R_k2α[k2,α]/beta
    #Gam_lesser_αmik2 = copy(Gam_greater_αmik2 )
    Gam_lesser_αmik[:,:,:,1:global_var.n_lorentz] .= Gam_lesser_αmik1
    Gam_lesser_αmik[:,:,:,global_var.n_lorentz+1:global_var.k_poles] .= Gam_greater_αmik2 # Gam_lesser_αmik2 they are equal
    return Gam_greater_αmik, Gam_lesser_αmik 
end
function create_csi(global_var)#(;csi_L=csi_L,csi_R=csi_R)
    """ This function creates the csi eigen vector ζ that corresponds to the 
    eigen vector of the Gamma functions, note that it will be stored in tensorial
    notation
    
    returns:
    --------
    csi_aikα
    """
    n_channels = global_var.n_channels
    n = global_var.n
    k_poles = global_var.k_poles                        
    csi_L = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
    csi_R = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
    csi_aikα = zeros(Float64, n*2,2,k_poles,2 )#cat(csi_L, csi_R, dims = 4)
    csi_L[1,1,:] = csi_L[2,2,:] = csi_R[2*n-1,1,:] = csi_R[2*n,2,:] .= 1.
    csi_aikα .= cat(csi_L, csi_R, dims = 4)
    #nothing
    return csi_aikα #csi_L, csi_R
end
                                                        
function get_poles(local_dim) # = n_channels*N_poles)
    """This function calculates the poles and residues of the Ozaki decomposotion of 
    the fermi energy(Note that there are not difference between right and left poles)
    """
    #:qlocal_dim = n_channels*N_poles + 1
    #Mat = zeros(local_dim,local_dim)
    diag = [1/(2*sqrt(4*x^2-1)) for x in 1:local_dim-1 ]
    ### necesary matrix to compute the Osaki poles and residues 
    Mat = diagm(-1 => diag) + diagm(1 => diag)
    ### engein values of the function 
    Eig_vals, Mat = eigen(Mat)
    ### residues 
    Res = Mat[1,:].^2 ./ (4 .*Eig_vals.^2)
    ### filtering the positive values (only the upper poles in the complex plane are needed)
    Eig_vals_p = [] # positive eigenvalues
    Res_p = []
    for i in 1:local_dim
        if Eig_vals[i]>0.
            #println(Eig_vals[i],"  " ,i )
            push!(Eig_vals_p, Eig_vals[i])
            push!(Res_p, Res[i])
        end
    end
    return Eig_vals_p, Res_p
end

                                                                    
end