### External libraries
using BenchmarkTools
using LinearAlgebra          ### Linear algebra library
using Plots                  ### Library to make plots
using DifferentialEquations  ### Library to use differential equations
using Tullio                 ### Library to work with tensors
using Base.Threads           ### Function to check the number of threads 
using DelimitedFiles         ### Manipulate files 
using LaTeXStrings           ### Latex strings
using StaticArrays
using Serialization
println("Time-dependent NEGF code")
println("------------------------")
println("Developed by Marko Petrovic, and Bogdan S. Popescu")
println("Modified and rewritten in julia by Jalil Varela-Manjarres")
println("Number of threads used in operations is : " , Threads.nthreads()  )
### Internal modules
include("./modules/parameters.jl")
include("./modules/read_parameters.jl")
include("./modules/get_parameters.jl")
include("./modules/derived_constants.jl")
include("./modules/configuration.jl")
include("./modules/create_hamiltonian.jl")
include("./modules/dynamical_variables.jl")
include("./modules/equation_of_motion.jl")
include("./modules/equilibrium_variables.jl")
include("./modules/green_functions.jl")
include("./modules/llg.jl")
include("./modules/observables.jl")
include("./modules/osaki_poles.jl")
include("./modules/precession.jl")
println("Modules were  loaded")
#push!(LOAD_PATH, "/home/jalil/Projects2023/TDNEGF/TDNEGF/modules/")
#using .parameters
import .read_parameters: read_params
read_params(;archivo_parametros= "./modules/$(ARGS[1]).txt")
using .get_parameters
using .derived_constants
import .configuration: configure!
import .create_hamiltonian: create_H
import .llg: llg_parameters, heff, heun
import .precession: PrecSpin, update!
import .observables: Observables
import .equilibrium_variables: init_denis, spindensity_eq
import .equation_of_motion: eom!
println("Parameters were loaded")


function main()
    ### Initiallize the variables in the dynamics
    rkvec = zeros(ComplexF64, size_rkvec)
    pr_spins = [PrecSpin(i) for i in 1:1:n_precessing  ]        ### array with mutables object of preccesin spins
    vm_a1x = [zeros(Float64,3) for _ in 1:n]                    ### array with vectors containiong the initial magnetization
    sm_eq_a1x = [zeros(Float64,3) for _ in 1:n] 
    diff = [zeros(Float64,3) for _ in 1:n]
    delta_α = Float64[0. , 0.]
    ### Set the initial configuration of the classical spins
    configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins, 0.0) ### set the initial values for pr_spins and vm_a1x
    H_ab = create_H(vm_a1x)                                             ### Initiallize the hamiltonian
    ### Initial evaluation of spin density 
    #println("parameterrrrs",params)
    sm_neq_a1x = Observables(rkvec,params_0 )["sden"]                  ### Modify the global parameter rkvec
    ### Parameters of the system in equilibirum 
    poles_denis, res_denis = init_denis(mu = E_F_system,temp=Temp,e_min=-3.,p=21)
    ### Read bias file
    if read_bias_file #& (i <= ti_bias)
        delta_α::Vector{Float64} .= data_bias[1,:]
    else
        delta_α .= [0. , 0.]
    end
    ### Seting ODE for electrons-bath
    prob = ODEProblem(eom!,rkvec, (0.0,t_end), [H_ab,delta_α] )         ### defines the problem for the differentia equation 
    ### Open the files were the data is saved
    if save_data["curr"]
        cc_f = open("./data/cc_$(name)_jl.txt", "w+")
    end
    if save_data["scurr"]
        sc_f = open("./data/sc_$(name)_jl.txt", "w+")
    end
    if save_data["sden_eq"]
        seq_f = open("./data/seq_$(name)_jl.txt", "w+")
    end
    if save_data["sden_neq"]
        sneq_f = open("./data/sneq_$(name)_jl.txt", "w+")
    end
    if save_data["rho"]
        rkvec_f = open("./data/rkvec_$(name)_jl.txt", "w+")
    end
    if save_data["sclas"]
        cspins_f = open("./data/cspins_$(name)_jl.txt", "w+")
    end
    ## Vern7 RK7/8
    integrator =  init(prob,Vern7(),dt=t_step, save_everystep=false,adaptive=true,dense=false)
    #,reltol=1e-12,abstol=1e-12)#,dt=t_step,reltol=1e-6,abstol=1e-6 )
    elapsed_time = @elapsed begin
    ### For loop for the evolution of a single step
    for (i,t) in enumerate(t_0:t_step:(t_end-t_step) )
        tt = round((i)*t_step,digits=2)
        println("time: ", tt  )
        flush(stdout)                                                ### asure that time_step is printed
        step!(integrator,t_step, true)                               ### evolve one time step  
        sm_neq_a1x .= Observables(integrator.u , params_0, false )["sden"]  ### update the spin density for electrons  
        sm_eq_a1x .= spindensity_eq(vm_a1x,energy_llg; t = 1.0, Temp = Temp ) 
        diff .= sm_neq_a1x .- sm_eq_a1x
        ### Now the magnetization is computed at time t + dt
        if run_llg
            vm_a1x .= heun(vm_a1x, diff,t_step,llg_parameters)        ### magnetization at time t+dt
        end
        ### If n_precessing > than 0 then the values are updated 
        ### Depending of the configuration, it modifies the configuration of the system
        ### at each time step 
        configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins,tt) 
        ### Calculate the needed observables at each time step 
        obs = Observables(integrator.u , params_0, false )
        if save_data["curr"]
            writedlm(cc_f, transpose(vcat(t,obs["curr"]...) ), ' ' )
        end
        if save_data["scurr"]
            writedlm(sc_f, transpose(vcat(t,obs["scurr"]...) ), ' ' )
        end
        if save_data["sden_eq"]
            writedlm(seq_f, transpose(vcat(t,sm_eq_a1x...) ), ' ' )
        end
        if save_data["sden_neq"]
            writedlm(sneq_f, transpose(vcat(t,obs["sden"]...) ), ' ' )
        end
        if save_data["sclas"]
            writedlm(cspins_f, transpose(vcat(t,vm_a1x...) ), ' ' )
        end
        ### This way to modify the parameter can be improved !!!
        if read_bias_file# & (i <= ti_bias)
            delta_α::Vector{Float64} .= data_bias[i+1,:]
        else
            delta_α .= [0. , 0.]
        end
        integrator.p[1] .= create_H(vm_a1x)  
        integrator.p[2] .=  delta_α
    end ### This end is for the "for-loop"
    end ### This end is for the elapsed time
    ### The storage of this files must be checked 
    if save_data["curr"]
        close(cc_f)
    end
    if save_data["scurr"]
        close(sc_f)
    end
    if save_data["sden_eq"]
        close(seq_f)
    end
    if save_data["sden_neq"]
        close(sneq_f)
    end
    if save_data["sclas"]
        close(cspins_f )
    end
    if save_data["rho"]
        #### save the last step of the rkvec 
        writedlm(rkvec_f, transpose(integrator.u ), ' ')
        close(rkvec_f)
    end
    println("Total time of simulation: ", elapsed_time, " s" )
    nothing
end
##############################################################################################
using Pkg
using PyCall                 ### In case that quspin will be used
pushfirst!(PyVector(pyimport("sys")."path"), "./modules/") ### link to my own python modules 
Kf= pyimport("Kitaev_func")
np = pyimport("numpy")
quspin_tools_measurements = pyimport("quspin.tools.measurements")
ED_state_vs_time_f = quspin_tools_measurements.ED_state_vs_time

function main_qsl()#(;t_0=t_0, t_step=t_step, t_end=t_end, llg_params = llg_parameters,name="ferropumpT5J1",θ=pi )
    #### Initial values for the variables 
    rkvec = zeros(ComplexF64, size_rkvec)
    pr_spins = [PrecSpin(i) for i in 1:1:n_precessing  ]        ### array with mutables object of preccesin spins
    vm_a1x = [zeros(Float64,3) for _ in 1:n]                    ### array with vectors containiong the initial magnetization
    sm_eq_a1x = [zeros(Float64,3) for _ in 1:n] 
    diff = [zeros(Float64,3) for _ in 1:n]
    delta_α = Float64[0., 0.]
    configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins,0.0)    ### set the initial values for pr_spins and vm_a1x
    H_ab = create_H(vm_a1x)                                     ### Initiallize the matrix with the density of the configuration
    ### Initial evaluation of spin density 
    sm_neq_a1x = Observables(rkvec,params_0 )["sden"]           ### Modify the global parameter rkvec
    ### Parameters of the system in equilibirum 
    poles_denis, res_denis = init_denis(mu = E_F_system,temp=Temp,e_min=-3.,p=21)
    ### Compute the Eigen values and the GS of the Kitaev model
    H_k = Kf.Kitaev_H(alpha = θ, Js = [1.,1.,1.],J_coup = J_qsl) ### the system is initially at heisenberg
    E_S, psi_S = np.linalg.eig(H_k.toarray())  ### Compute the eigen values and the eigen vectors
    #H_k.eigh()#.eigsh(which = "SA")#.eigsh(k = 1,which = "SA") 
    psi_GS = psi_S[:,1]   ### Just take the first eigen value
    #println(psi_GS)
    #psi_S[1,:] 
    if read_bias_file #& (i <= ti_bias)
        delta_α::Vector{Float64} .= data_bias[1,:]
    else
        delta_α .= [0. , 0.]
    end
    ### Seting ODE for electrons-bath
    prob = ODEProblem(eom!,rkvec, (0.0,t_end), [H_ab,delta_α] )         ### defines the problem for the differentia equation 
    integrator =  init(prob,Vern7(),dt=t_step, save_everystep=false,adaptive=true,dense=false)
    ### Open the files were the data is saved
    if save_data_qsl["curr"]
        cc_f = open("./data/cc_$(name)_jl.txt", "w+")
    end
    if save_data_qsl["scurr"]
        sc_f = open("./data/sc_$(name)_jl.txt", "w+")
    end
    if save_data_qsl["sden_eq"]
        seq_f = open("./data/seq_$(name)_jl.txt", "w+")
    end
    if save_data_qsl["sden_neq"]
        sneq_f = open("./data/sneq_$(name)_jl.txt", "w+")
    end
    if save_data_qsl["rho"]
        rkvec_f = open("./data/rkvec_$(name)_jl.txt", "w+")
    end
    if save_data_qsl["sclas"]
        cspins_f = open("./data/cspins_$(name)_jl.txt", "w+")
    end
    #### Spin liquid paramaters
    if save_data_qsl["ent"]
        entropy_f = open("./data/entropy_$(name)_sl_jl.txt", "w+")
    end
    if save_data_qsl["sden_qsl"]
        sden_sl_f = open("./data/sden_$(name)_sl_jl.txt", "w+")
    end

    
    elapsed_time = @elapsed begin
    ### Time evolution loop 
    for (i,t) in enumerate(t_0:t_step:(t_end-t_step) )
        ###inclusion of the bias 
        tt = round((i)*t_step,digits=2)
        println("time: ", tt  )
        flush(stdout)                                                          ### ensure that time_step is printed
        ### evolvution of the electron-bath 
        step!(integrator,t_step, true)                                
        ### Evolution of the Kitaev model 
        #psi_GS = vec(ED_state_vs_time_f(psi_GS, E_S,psi_S, np.array([0.1*hbar]) ) )#,iterate=False) hbar is because in the quspin svol hbar=1
        psi_GS = vec(Kf.evolve(H_static=H_k, H_dynamic=[], psi=psi_GS, dt=0.1, method= "CN", time_dep=0) )
        #### Evaluation of spin densities
        m_qsl = real(Kf.spindensity_qsl(psi=psi_GS,sites=[5,6,7])    )
        #### Note that this only returns 3 spin spin densities, then 
        ### we must acomodate the hilbert space in order to couple this to 
        ### the hilbert space of the electrons
        vm_qsl_a1x = [real(m_qsl[i, :]) for i in 1:3 ]#:: Int]                 ### spin density of qsl
        pushfirst!(vm_qsl_a1x, [zeros(Float64,3) for _ in 1:6]... )              ### the fisrt 3 sites of the electron lattice is not coupled
        sm_neq_a1x .= Observables(integrator.u , params_0, false )["sden"]     ### update the spin density for electrons  
        sm_eq_a1x .= spindensity_eq(vm_a1x,energy_llg; t = 1.0, Temp = Temp )  ### spin density in eq
        diff .= sm_neq_a1x .- sm_eq_a1x      
        ### Now the magnetization is computed at time t + dt
        if run_llg
            vm_a1x .= heun(vm_a1x, diff,t_step,llg_parameters)        ### magnetization at time t+dt
        end
        ### at each time step 
        configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins,tt) 
        obs = Observables(integrator.u , params_0, false )
        if save_data_qsl["curr"]
            writedlm(cc_f, transpose(vcat(t,obs["curr"]...) ), ' ' )
        end
        if save_data_qsl["scurr"]
            writedlm(sc_f, transpose(vcat(t,obs["scurr"]...) ), ' ' )
        end
        if save_data_qsl["sden_eq"]
            writedlm(seq_f, transpose(vcat(t,sm_eq_a1x...) ), ' ' )
        end
        if save_data_qsl["sden_neq"]
            writedlm(sneq_f, transpose(vcat(t,obs["sden"]...) ), ' ' )
        end
        if save_data_qsl["sclas"]
            writedlm(cspins_f, transpose(vcat(t,vm_a1x...) ), ' ' )
        end
        if save_data_qsl["ent"]
        ### Entropy
            ent = Kf.basis.ent_entropy(psi_GS ,sub_sys_A=A,alpha=1,density=true)["Sent_A"][1]
            writedlm(entropy_f, transpose(vcat(t, ent  ) ), ' ' )
        end
        if save_data_qsl["sden_qsl"]
        # Spin density
        sden = real(Kf.spindensity_qsl(psi=psi_GS,sites=[0,1,2,3,4,5,6,7,8,9]))
        writedlm(sden_sl_f, transpose(vcat(t, sden...) ), ' ' )
                
        end            
        # Read bias file
        if read_bias_file #& (i <= ti_bias)
            delta_α::Vector{Float64} .= data_bias[i+1,:]
        else
            delta_α .= [0. , 0.]
        end
        integrator.p[1] .= create_H(vm_a1x,vm_qsl_a1x)  
        integrator.p[2] .=  delta_α#create_H(vm_a1x,vm_qsl_a1x) 
        ### The Kitaev hamiltonian is updated with the expected values of the electronic spins
        H_k = Kf.Kitaev_H(alpha = θ,S = hcat(sm_neq_a1x...), Js = [1.0,1.0,1.0],J_coup = -J_qsl) #sm_neq_a1x
        ##E_S, psi_S = H_k.eigh() #.eigsh(which = "SA") 
    end ### end for the elapsed time
    end ### End for for

    if save_data_qsl["curr"]
        close(cc_f)
    end
    if save_data_qsl["scurr"]
        close(sc_f)
    end
    if save_data_qsl["sden_eq"]
        close(seq_f)
    end
    if save_data_qsl["sden_neq"]
        close(sneq_f)
    end
    if save_data_qsl["sclas"]
        close(cspins_f )
    end
    if save_data["rho"]
        #### save the last step of the rkvec 
        writedlm(rkvec_f, transpose(integrator.u ), ' ')
        close(rkvec_f)
    end
    ### spin liquid
    if save_data_qsl["ent"]
        ### Entropy
        close(entropy_f)
    end
    if save_data_qsl["sden_qsl"]
        # Spin density
        close(sden_sl_f)
    end       
    println("Total time of simulation: ", elapsed_time, " s" )
    nothing
end


###############################################################################################
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    #main_qsl()
end