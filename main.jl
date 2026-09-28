
using DifferentialEquations  ### Library to use differential equations
using LinearAlgebra
using DelimitedFiles         ### Manipulate files 
⊗(A,B) = kron(A,B)
### Inlcude Libraries
include("./modules/configuration.jl")
include("./modules/global_parameters.jl")
include("./modules/equation_of_motion.jl")
include("./modules/equilibrium_variables.jl")
include("./modules/observables.jl")
import .global_parameters: global_params, get_poles,create_hi, create_Gam, create_csi
import .configuration: configure!
import .observables: Observables!
import .equilibrium_variables: init_denis!,spindensity_eq,cden_eq#,bcurrs_eq #rho_denis
import .equation_of_motion: eom!, to_matrix


include("./modules/create_hamiltonian.jl")
import .create_hamiltonian: Central_H, A_light #,central_hamiltonian,

include("./modules/llg.jl")
import .llg: heun

function main(;kwargs...)

    #### New parameters to be added in the global_params function 
    #1) Return the new llg parameters in the function
    # nx, ny and all the other variables  lg = new_llg_parameters(nx=nx,ny=ny)
    ################## Note that we only need this to define the Leads
    #### Initiallize the function and parameters
    #### If something is modified it should be from this point
    gv ,dv ,ov,lv,cv,ev = global_params("./modules/parameters.txt";kwargs...)
    Eig_vals, Res_p = get_poles(gv.n_channels*gv.N_poles)
    Eig_vals_k2α = cat(Eig_vals,Eig_vals,dims=2)
    R_k2α = cat(Res_p,Res_p,dims=2) 
    dv.hi_αmk,dv.hi_αmk1,dv.hi_αmk2 = create_hi(gv; Eig_vals_k2α = Eig_vals_k2α)
    dv.Gam_greater_αmik,dv.Gam_lesser_αmik = create_Gam(gv;R_k2α=R_k2α
                                                        ,hi_αmk=dv.hi_αmk,
                                                        hi_αmk1=dv.hi_αmk1,
                                                        hi_αmk2=dv.hi_αmk2 )
    dv.csi_aikα = create_csi(gv) ;
    #### Initial conditions: 
    println("Parameters were loaded")
    vm_a1x = zeros(Float64,gv.nx*gv.ny,3)
    # Initial AFM state
    # vm_a1x[1:2:end,1] .= 1.
    # vm_a1x[2:2:end,1] .= -1.
    dv.vm_a1x = vm_a1x
    ### Initial Hamiltonian and parameters in eq
    dv.H_ab = Central_H(vm_a1x;gv=gv,cv=cv)
    #H_ab_eq = copy(dv.H_ab)
    ### Only modifies sm_neq_a1x This should be especified in paras_0
    params_sden = Dict("curr"=>false, "scurr"=>false, "sden"=>true, "cden" =>false, "bcurrs" =>false); 
    #params   = Dict("curr"=>false, "scurr"=>false, "sden"=>true, "cden" =>true, "bcurrs" =>true);  #### Only the spin density is obtained 
    Observables!(dv.rkvec,params_sden,dv,ov,gv ) ### Modifies the observables
    ### Parameters of the system in equilibirum 
    init_denis!(ev) 
    ### Seting ODE for electrons-bath
    println("Initial conditions were setted")
    prob = ODEProblem(eom!,dv.rkvec, (0.0, gv.t_end), dv) ### defines the problem for the differentia 
    ### Open the files were the data is saved
    gv.save_data["curr"] && (cc_f = open("./data/cc_$(gv.name)_jl.txt", "w+") )
    gv.save_data["scurr"] && (sc_f = open("./data/sc_$(gv.name)_jl.txt", "w+") )
    #gv.save_data["sden_eq"] && ( seq_f = open("./data/seq_$(gv.name)_jl.txt", "w+") )
    gv.save_data["sden_neq"] && (sneq_f = open("./data/sneq_$(gv.name)_jl.txt", "w+") )
    gv.save_data["rho"] && (rkvec_f = open("./data/rkvec_$(gv.name)_jl.txt", "w+") )
    gv.save_data["sclas"] && (cspins_f = open("./data/cspins_$(gv.name)_jl.txt", "w+") )
    gv.save_data["cden"] &&  (cden_f = open("./data/cden_$(gv.name)_jl.txt", "w+")  )
    #gv.save_data["bcurrs"] && (bcurr_f = open("./data/bcurr_$(gv.name)_jl.txt", "w+")  )
    ##### New observable to save
    
    ### Setting the integrator 
    integrator =  init(prob,Vern7(),dt = gv.t_step, save_everystep=false,adaptive=true,dense=false)
    ### Read the light pulse if avaible 
    A_t_pulse = A_light(gv)
    println("Light is loaded")
    println("Starting time evolution")
    elapsed_time = @elapsed begin
    ## For loop for the evolution of a single step
    for (i,t) in enumerate(gv.t_0:gv.t_step:(gv.t_end-gv.t_step) )
        tt = round((i)*gv.t_step,digits=2)
        println("time: ", tt  )
        flush(stdout)                                                ### asure that time_step is printed
        step!(integrator,gv.t_step, true)                            ### evolve one time step  
        Observables!(integrator.u, params_sden,dv,ov,gv) 
        ### The equilibirum spin density is calculated with the instanteous hamiltonian. 
        #dv.sm_eq_a1x .= spindensity_eq(dv.H_ab,ev,gv)
        #dv.diff .= ov.sden_a1x ## .- dv.sm_eq_a1x
        #gv.run_llg && (dv.vm_a1x .= heun(dv.vm_a1x,dv.diff,gv.t_step,lv))
        ### Calculate the needed observables at each time step
        Observables!(integrator.u, gv.params, dv, ov, gv) 
        # Update the interaction term and the Electric Field 
        cv.tc1 = gv.tc1*exp(im*gv.z*A_t_pulse[i]) 
        # cv.tc2 = gv.tc2*exp(im*gv.z*A_t_pulse[i])
        cv.E_t = A_t_pulse[i]
        # Update hamiltonian in the integrator 
        dv.H_ab .= Central_H(dv.vm_a1x;gv,cv)
        integrator.p.H_ab .= dv.H_ab  #dynamics_var.H_ab  
        ### Save the data at each time step
        gv.save_data["curr"] && writedlm(cc_f, ov.curr_α, ' ' )
        gv.save_data["scurr"] && writedlm(sc_f, ov.scurr_xα , ' ' )
        #gv.save_data["sden_eq"] && writedlm(seq_f, dv.sm_eq_a1x, ' ' )
        gv.save_data["sden_neq"] && writedlm(sneq_f, real(ov.sden_a1x), ' ' )
        gv.save_data["sclas"] && writedlm(cspins_f,dv.vm_a1x, ' ' )
        gv.save_data["cden"] && writedlm(cden_f, ov.cden, ' ' )
        #gv.save_data["bcurrs"] && writedlm(bcurr_f, ov.bcurrs, ' ' )
        ###### New Observable to save 
    end
    end
    gv.save_data["curr"] && close(cc_f)
    gv.save_data["scurr"] && close(sc_f)
    #gv.save_data["sden_eq"] && close(seq_f)
    gv.save_data["sden_neq"] && close(sneq_f)
    gv.save_data["sclas"] && close(cspins_f )
    gv.save_data["cden"] && close(cden_f)
    #gv.save_data["bcurrs"] && close(bcurr_f)
    ###### New Observable 
    println("Total time of simulation: ", elapsed_time, " s" )
    nothing
end