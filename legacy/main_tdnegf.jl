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
include("./modules/configuration.jl")
include("./modules/create_hamiltonian.jl")
include("./modules/global_parameters.jl")
include("./modules/equation_of_motion.jl")
include("./modules/equilibrium_variables.jl")
include("./modules/llg.jl")
include("./modules/observables.jl")
println("Modules were  loaded")
import .global_parameters: global_params, get_poles,create_hi, create_Gam, create_csi
import .configuration: configure!
import .create_hamiltonian: create_H
import .llg: heun
import .observables: Observables!
import .equilibrium_variables: init_denis!,spindensity_eq,cden_eq,bcurrs_eq #rho_denis
import .equation_of_motion: eom!, to_matrix
println("Parameters were loaded")
function main()
    println("Join the Main function")
    ### Initiallize the variables in the dynamics
    global_var ,dynamics_var ,observables_var, llg_parameters,config_var,eq_var = global_params("./modules/$(ARGS[1]).txt")
    Eig_vals, Res_p = get_poles(global_var.n_channels*global_var.N_poles)
    Eig_vals_k2α = cat(Eig_vals,Eig_vals,dims=2)
    R_k2α = cat(Res_p,Res_p,dims=2) 
    dynamics_var.hi_αmk,dynamics_var.hi_αmk1,dynamics_var.hi_αmk2 = create_hi(global_var; Eig_vals_k2α = Eig_vals_k2α)
dynamics_var.Gam_greater_αmik,dynamics_var.Gam_lesser_αmik=create_Gam(global_var;R_k2α=R_k2α,hi_αmk=dynamics_var.hi_αmk,hi_αmk1=dynamics_var.hi_αmk1,hi_αmk2=dynamics_var.hi_αmk2 )
    dynamics_var.csi_aikα = create_csi(global_var) ;
    ### The initial configuraration is defined inside the function configure!()
    configure!(0.0 ,dynamics_var; global_var = global_var, config_var = config_var)
    dynamics_var.H_ab = create_H(dynamics_var.vm_a1x; global_var = global_var, config_var = config_var )     ## Initiallize the hamiltonian
    ### Only modifies sm_neq_a1x This should be especified in paras_0
    params_sden = Dict("curr"=>false, "scurr"=>false, "sden"=>true, "cden" =>false, "bcurrs" =>false);  #### Only the spin density is obtained 
    #params   = Dict("curr"=>false, "scurr"=>false, "sden"=>true, "cden" =>true, "bcurrs" =>true);  #### Only the spin density is obtained 
    Observables!(dynamics_var.rkvec,params_sden,dynamics_var,observables_var,global_var ) ### Modifies the observables
    ### Parameters of the system in equilibirum 
    init_denis!(eq_var) 
    global_var.preload_rkvec &&   (rkvec = readdlm("./data/rkvec_test_jl.txt",',', ComplexF64) )
    println("preload rkvec file : " , global_var.preload_rkvec, ", with name : ", global_var.name_preload_rkvec)
    ### Seting ODE for electrons-bath
    prob = ODEProblem(eom!,dynamics_var.rkvec, (0.0,global_var.t_end), dynamics_var) #[H_ab,delta_αi] )         ### defines the problem for the differentia equation 
    ### Open the files were the data is saved
    global_var.save_data["curr"] && (cc_f = open("./data/cc_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["scurr"] && (sc_f = open("./data/sc_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["sden_eq"] && ( seq_f = open("./data/seq_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["sden_neq"] && (sneq_f = open("./data/sneq_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["rho"] && (rkvec_f = open("./data/rkvec_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["sclas"] && (cspins_f = open("./data/cspins_$(global_var.name)_jl.txt", "w+") )
    global_var.save_data["cden"] &&  (cden_f = open("./data/cden_$(global_var.name)_jl.txt", "w+")  )
    global_var.save_data["bcurrs"] && (bcurr_f = open("./data/bcurr_$(global_var.name)_jl.txt", "w+")  )
    ## Vern7 RK7/8
    integrator =  init(prob,Vern7(),dt = global_var.t_step, save_everystep=false,adaptive=true,dense=false)
    pulse_light = readdlm("./pulse_12_gap.txt")
    #,reltol=1e-12,abstol=1e-12)#,dt=t_step,reltol=1e-6,abstol=1e-6 )
    j=0
    thops = config_var.thops
    rho_avg_ab = zeros(ComplexF64, 2*global_var.n, 2*global_var.n )
    U_ab = ones(ComplexF64, 2*global_var.n, 2*global_var.n )
    elapsed_time = @elapsed begin
    ## For loop for the evolution of a single step
    for (i,t) in enumerate(global_var.t_0:global_var.t_step:(global_var.t_end-global_var.t_step) )
        tt = round((i)*global_var.t_step,digits=2)
        println("time: ", tt  )
        flush(stdout)                                                ### asure that time_step is printed
        step!(integrator,global_var.t_step, true)                               ### evolve one time step  
        Observables!(integrator.u, params_sden,dynamics_var,observables_var,global_var) 
        ### The equilibirum spin density is calculated with the insatanteous hamiltonian. 
        dynamics_var.sm_eq_a1x .= spindensity_eq(dynamics_var.H_ab,eq_var,global_var)
        dynamics_var.diff .= observables_var.vsden_xa1 .- dynamics_var.sm_eq_a1x
        ### Now the magnetization is computed at time t + dt
        #### Here llg runs 
        configure!(tt, dynamics_var; global_var = global_var, config_var = config_var  ) ### Note that the Rice-Mele configuration is used 
        #global_var.run_llg && (dynamics_var.vm_a1x .= heun(dynamics_var.vm_a1x, dynamics_var.diff,global_var.t_step,llg_parameters))
        if tt > 200#1000    
            ### magnetization at time t+dt
            try
                config_var.thops = thops*exp(im*0.2*pulse_light[j+1]) ### The parameters are modified again to include the time dependence 
                j=j+1
            catch 
                config_var.thops = thops
                nothing
            end
        end
        ### Calculate the needed observables at each time step 
        Observables!(integrator.u, global_var.params, dynamics_var, observables_var, global_var) 
        observables_var.cden = observables_var.cden - cden_eq(dynamics_var.H_ab, eq_var,global_var)
        observables_var.bcurrs = observables_var.bcurrs - bcurrs_eq(dynamics_var.H_ab, eq_var,global_var)
        ### Save the data at each time step
        global_var.save_data["curr"] && writedlm(cc_f, [t;observables_var.curr_α...]', ' ' )
        global_var.save_data["scurr"] && writedlm(sc_f, [t;observables_var.scurr_xα...]' , ' ' )
        global_var.save_data["sden_eq"] && writedlm(seq_f, [t;dynamics_var.sm_eq_a1x...]', ' ' )
        global_var.save_data["sden_neq"] && writedlm(sneq_f, [t;observables_var.vsden_xa1...]', ' ' )
        global_var.save_data["sclas"] && writedlm(cspins_f,[t;dynamics_var.vm_a1x...]', ' ' )
        global_var.save_data["cden"] && writedlm(cden_f, [t;observables_var.cden...]', ' ' )
        global_var.save_data["bcurrs"] && writedlm(bcurr_f, [t;observables_var.bcurrs...]', ' ' )
        #### Lets add the interaction term 
        #cden_f
        rho_i(i) = dynamics_var.rho_ab[2*i-1:2*i, 2*i-1:2*i]
        U=0.5
        function rho_avg(i) 
            if (i>1) & (i<global_var.n)
                rho_i(i+1) + rho_i(i-1) + rho_i(i)
            elseif i==1
                rho_i(i+1) +rho_i(i)
            elseif i == global_var.n
                rho_i(i-1)+rho_i(i)
            end
        end
            
        for i in range(1,global_var.n)
            rho_avg_ab[2*i-1:2*i, 2*i-1:2*i] = rho_avg(i)
        end
        U_ab = U*rho_avg_ab.*U_ab
        
        #### The variables parameters are updated depending on the configurarion 
        dynamics_var.H_ab .= create_H(dynamics_var.vm_a1x; global_var = global_var, config_var = config_var ) + U_ab    ## update the hamiltonian                  
        # Update hamiltonian
        integrator.p.H_ab = dynamics_var.H_ab
        integrator.p.delta_αi
    end ### This end is for the "for-loop"
    end ### This end is for the elapsed time
    ### The storage of this files must be checked 
    global_var.save_data["curr"] && close(cc_f)
    global_var.save_data["scurr"] && close(sc_f)
    global_var.save_data["sden_eq"] && close(seq_f)
    global_var.save_data["sden_neq"] && close(sneq_f)
    global_var.save_data["sclas"] && close(cspins_f )
    global_var.save_data["cden"] && close(cden_f)
    global_var.save_data["bcurrs"] && close(bcurr_f)
    if global_var.save_data["rho"]
        #### save the last step of the rkvec 
        writedlm(rkvec_f, integrator.u, ',' )
        close(rkvec_f)
    end
    println("Total time of simulation: ", elapsed_time, " s" )
    nothing
end
##############################################################################################
# using Pkg
# using PyCall                 ### In case that quspin will be used
#Pkg.add("Conda")
#Pkg.build("PyCall")
# pushfirst!(PyVector(pyimport("sys")."path"), "./modules/") ### link to my own python modules 
# Kf= pyimport("Kitaev_func")
# np = pyimport("numpy")
# quspin_tools_measurements = pyimport("quspin.tools.measurements")
# ED_state_vs_time_f = quspin_tools_measurements.ED_state_vs_time

# function main_qsl()#(;t_0=t_0, t_step=t_step, t_end=t_end, llg_params = llg_parameters,name="ferropumpT5J1",θ=pi )
#     #### Initial values for the variables 
#     rkvec = zeros(ComplexF64, size_rkvec)
#     pr_spins = [PrecSpin(i) for i in 1:1:n_precessing  ]        ### array with mutables object of preccesin spins
#     vm_a1x = [zeros(Float64,3) for _ in 1:n]                    ### array with vectors containiong the initial magnetization
#     sm_eq_a1x = [zeros(Float64,3) for _ in 1:n] 
#     diff = [zeros(Float64,3) for _ in 1:n]
#     delta_αi = zeros(Float64,2,2)#Float64[0., 0.]
#     configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins,0.0)    ### set the initial values for pr_spins and vm_a1x
#     H_ab = create_H(vm_a1x)                                     ### Initiallize the matrix with the density of the configuration
#     ### Initial evaluation of spin density 
#     sm_neq_a1x = Observables(rkvec,params_0 )["sden"]           ### Modify the global parameter rkvec
#     ### Parameters of the system in equilibirum 
#     poles_denis, res_denis = init_denis(mu = E_F_system,temp=Temp,e_min=-3.,p=21)
#     ### Compute the Eigen values and the GS of the Kitaev model
#     H_k = Kf.Kitaev_H(alpha = θ, Js = [1.,1.,1.],J_coup = 	J_qsl) ### the system is initially at heisenberg
#     E_S, psi_S = np.linalg.eig(H_k.toarray())  ### Compute the eigen values and the eigen vectors
#     #H_k.eigh()#.eigsh(which = "SA")#.eigsh(k = 1,which = "SA") 
#     psi_GS = psi_S[:,1]   ### Just take the first eigen value
#     #println(psi_GS)
#     #psi_S[1,:] 
#     if read_bias_file #& (i <= ti_bias)
#         #delta_αi[:,1] .= data_bias[1,:]
#         delta_αi[1,1] = data_bias[1,1]
#         delta_αi[2,2] = data_bias[1,1]
#     else
#         delta_αi .= zeros(Float64,2,2)#[0. , 0.]
#     end
#     ### Seting ODE for electrons-bath
#     prob = ODEProblem(eom!,rkvec, (0.0,t_end), [H_ab,delta_αi] )         ### defines the problem for the differentia equation 
#     integrator =  init(prob,Vern7(),dt=t_step, save_everystep=false,adaptive=true,dense=false)
#     ### Open the files were the data is saved
#     save_data_qsl["curr"] && (cc_f = open("./data/cc_$(name)_jl.txt", "w+") )
#     save_data_qsl["scurr"] && (sc_f = open("./data/sc_$(name)_jl.txt", "w+") )
#     save_data_qsl["sden_eq"] && (seq_f = open("./data/seq_$(name)_jl.txt", "w+") )
#     save_data_qsl["sden_neq"] && (sneq_f = open("./data/sneq_$(name)_jl.txt", "w+") )
#     save_data_qsl["rho"] && (rkvec_f = open("./data/rkvec_$(name)_jl.txt", "w+") )
#     save_data_qsl["sclas"] && (cspins_f = open("./data/cspins_$(name)_jl.txt", "w+") )
#     #### Spin liquid paramaters
#     save_data_qsl["ent"] && (entropy_f = open("./data/entropy_$(name)_sl_jl.txt", "w+") )
#     save_data_qsl["sden_qsl"] && (sden_sl_f = open("./data/sden_$(name)_sl_jl.txt", "w+") )
    
#     elapsed_time = @elapsed begin
#     ### Time evolution loop 
#     for (i,t) in enumerate(t_0:t_step:(t_end-t_step) )
#         ###inclusion of the bias 
#         tt = round((i)*t_step,digits=2)
#         println("time: ", tt  )
#         flush(stdout)                                                          ### ensure that time_step is printed
#         ### evolvution of the electron-bath 
#         step!(integrator,t_step, true)                                
#         ### Evolution of the Kitaev model 
#         #psi_GS = vec(ED_state_vs_time_f(psi_GS, E_S,psi_S, np.array([0.1*hbar]) ) )#,iterate=False) hbar is because in the quspin svol hbar=1
#         psi_GS = vec(Kf.evolve(H_static=H_k, H_dynamic=[], psi=psi_GS, dt=0.1, method= "CN", time_dep=0) )
#         #### Evaluation of spin densities
#         m_qsl = real(Kf.spindensity_qsl(psi=psi_GS,sites=[5,6,7,8,9])    )
#         #### Note that this only returns 3 spin spin densities, then 
#         ### we must acomodate the hilbert space in order to couple this to 
#         ### the hilbert space of the electrons
#         vm_qsl_a1x = [real(m_qsl[i, :]) for i in 1:5 ]#:: Int]                 ### spin density of qsl
#         ## pushfirst!(vm_qsl_a1x, [zeros(Float64,3) for _ in 1:6]... )              ### the fisrt 3 sites of the electron lattice is not coupled
#         sm_neq_a1x .= Observables(integrator.u , params_0, false )["sden"]     ### update the spin density for electrons  
#         sm_eq_a1x .= spindensity_eq(vm_a1x,energy_llg; t = 1.0, Temp = Temp )  ### spin density in eq
#         diff .= sm_neq_a1x .- sm_eq_a1x      
#         ### Now the magnetization is computed at time t + dt
#         run_llg && (vm_a1x .= heun(vm_a1x, diff,t_step,llg_parameters) )       ### magnetization at time t+dt
#         ### at each time step 
#         configure!(cspin_orientation,llg_parameters,vm_a1x,pr_spins,tt) 
#         obs = Observables(integrator.u , params_0, false;vm_a1x =vm_a1x   , H_ab = H_ab ) ### Note the two addisional arguments used to calculate the observables
#         save_data_qsl["curr"] && writedlm(cc_f, transpose(vcat(t,obs["curr"]...) ), ' ' )
#         save_data_qsl["scurr"] && writedlm(sc_f, transpose(vcat(t,obs["scurr"]...) ), ' ' )
#         save_data_qsl["sden_eq"] && writedlm(seq_f, transpose(vcat(t,sm_eq_a1x...) ), ' ' )
#         save_data_qsl["sden_neq"] && writedlm(sneq_f, transpose(vcat(t,obs["sden"]...) ), ' ' )
#         save_data_qsl["sclas"]&& writedlm(cspins_f, transpose(vcat(t,vm_a1x...) ), ' ' )
#         if save_data_qsl["ent"]
#         ### Entropy
#             ent = Kf.basis.ent_entropy(psi_GS ,sub_sys_A=A,alpha=1,density=true)["Sent_A"][1]
#             writedlm(entropy_f, transpose(vcat(t, ent  ) ), ' ' )
#         end
#         if save_data_qsl["sden_qsl"]
#             # Spin density
#             sden_save= Kf.spindensity_qsl(psi=psi_GS,sites=[0,1,2,3,4,5,6,7,8,9])
#             sden = [real(sden_save[i, :]) for i in 1:10 ]
#             #real(Kf.spindensity_qsl(psi=psi_GS,sites=[0,1,2,3,4,5,6,7,8,9]))
#             writedlm(sden_sl_f, transpose(vcat(t, sden...) ), ' ' )    
#         end            
#         # Read bias file
#         if read_bias_file #& (i <= ti_bias)
#             #delta_αi[:,1] .= data_bias[i+1,:]
#             delta_αi[1,1] = data_bias[i+1,1]
#             delta_αi[2,2] = data_bias[i+1,1]
#         else
#             delta_αi .= zeros(Float64,2,2)#[0. , 0.]
#         end
#         integrator.p[1] .= create_H(vm_a1x,vm_qsl_a1x)  
#         integrator.p[2] .=  delta_αi#create_H(vm_a1x,vm_qsl_a1x) 
#         ### The Kitaev hamiltonian is updated with the expected values of the electronic spins
#         H_k = Kf.Kitaev_H(alpha = θ,S = hcat(sm_neq_a1x...), Js = [1.0,1.0,1.0],J_coup = J_qsl) #sm_neq_a1x
#         ##E_S, psi_S = H_k.eigh() #.eigsh(which = "SA") 
#     end ### end for the elapsed time
#     end ### End for for
#     save_data_qsl["curr"] && close(cc_f)
#     save_data_qsl["scurr"] && close(sc_f)
#     save_data_qsl["sden_eq"] && close(seq_f)
#     save_data_qsl["sden_neq"] && close(sneq_f)
#     save_data_qsl["sclas"] && close(cspins_f)
#     if save_data["rho"]
#         #### save the last step of the rkvec 
#         writedlm(rkvec_f, transpose(integrator.u ), ' ')
#         close(rkvec_f)
#     end
#     ### spin liquid
#     save_data_qsl["ent"] && close(entropy_f)
#     ### Entropy
#     save_data_qsl["sden_qsl"] && close(sden_sl_f)
#     println("Total time of simulation: ", elapsed_time, " s" )
#     nothing
# end

###############################################################################################
if abspath(PROGRAM_FILE) == @__FILE__
   main()
   #main_qsl()
end
