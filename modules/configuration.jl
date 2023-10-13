module configuration

### Libraries
using LinearAlgebra
include("parameters.jl")
using .parameters
include("precession.jl")
import .precession: PrecSpin, update!

function configure!(cname, llg_params, vm_a1x, pr_spins, t )
    """ This function stablish the configuration 
    of the magnetic moments
    variables:
    ---------
    cname: name of the configuration to be used
    lp: mutable structure or parameters 
    pr_spins:vector of mutable structure
    contain the object PrecSpin associated with the 
    loca magnetic moments 
    
    returns:
    -------
    initiallize and modify 
    the magnetic moment configuration and
    the llg_parameters
    """
    ##############################################################################
    if cname ==  "arb_dir"
        if t == 0.0 #### Initial conditions 
            println("join configuration: $(cname)")
                    ### Initiallize the values 
            for jj in 1: n::Int  ### Run over the number of sites of the lattice 
                vm_a1x[jj][1] = 0.
                vm_a1x[jj][2] = 0.
                vm_a1x[jj][3] = 0.
            end
        end
        ### Global parameters ares modified
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        J_sd_local .= J_sd.*ones(n) 
        thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################

        ### put the spins to precces
        for jj in 1: n_precessing::Int 
            pr_spins[jj].i = jj ## lattice site 
            pr_spins[jj].theta_zero = theta_1
            pr_spins[jj].axis_phi = phi_1
            pr_spins[jj].T = period 
            #println(pr_spins[jj].i)
        end
        for j in 1:n_precessing#length(pr_spins) 3:1:5
            update!(pr_spins[j], t )
            vm_a1x[pr_spins[j].i ] .= pr_spins[j].s
        end   
    end 
    ##############################################################################
    if cname == "precess"
        ### Global parameters ares modified
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        J_sd_local .= J_sd.*ones(n) 
        thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        if t == 0.0 #### Initial conditions 
            println("join configuration: $(cname)")
            for jj in 1: n::Int  ### Run over the number of sites of the lattice 
                vm_a1x[jj][1] = 0.
                vm_a1x[jj][2] = 0.
                vm_a1x[jj][3] = 1.
            end
            for jj in 1: n_precessing::Int 
                pr_spins[jj].i = jj ## lattice site 
                pr_spins[jj].theta_zero = theta_1
                pr_spins[jj].axis_phi = phi_1
                pr_spins[jj].T = period 
                #println(pr_spins[jj].i)
            end
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        J_sd_local .= J_sd.*ones(n) 
        thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        end

        
        ### Notice that the range of the spins that are precessing can be modified
        for j in 1:n_precessing#length(pr_spins) 3:1:5
            update!(pr_spins[j], t )
            vm_a1x[pr_spins[j].i ] .= pr_spins[j].s
        end   
    end
    ##############################################################################
    if cname == "sym_pump"
    ### This configuration was created to check the different symmetries when we include the SOC
    if t == 0.0 #### Initial conditions 
            println("join configuration: $(cname)")
            ### Initiallize the values 
            for jj in 1: n::Int  ### Run over the number of sites of the lattice 
                vm_a1x[jj][1] = 0.
                vm_a1x[jj][2] = 0.
                vm_a1x[jj][3] = 0.
            end
            for jj in 1:n_precessing::Int #::Int 
                pr_spins[jj].i = jj ## lattice site 
                pr_spins[jj].theta_zero = theta_1
                pr_spins[jj].axis_phi = phi_1
                pr_spins[jj].T = period 
                #println(pr_spins[jj].i)
            end
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        J_sd_local .= J_sd.*ones(n) 
        thop_local .= thop.*ones(n-1) 
        tso_local .= alpha_r.*zeros(ComplexF64,n-1)
        tso_local[1] = -alpha_r#*im
        tso_local[4] =  -alpha_r#*im
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        end   

     for j in 3:1:5#length(pr_spins) 3:1:5
        vm_a1x[j][1] = 0.
        vm_a1x[j][2] = 0.
        vm_a1x[j][3] = 1
        update!(pr_spins[j], t )
        vm_a1x[pr_spins[j].i ] .= pr_spins[j].s
      end     

    end
    ##############################################################################
    if cname == "polarizer"
        ### Polarize the spin charge current comming from the leads 
        if t == 0
            println("join configuration: $(cname)")
            for jj in 1:3
                vm_a1x[jj][1] = 0.
                vm_a1x[jj][2] = 0.
                vm_a1x[jj][3] = 1.0
            end
            for jj in 4:9
                vm_a1x[jj][1] = 0.
                vm_a1x[jj][2] = 0.
                vm_a1x[jj][3] = 0.
            end
        end
        
    end
    ##############################################################################


    
    ### Note that it returns a configuuration for the magnetic moments and 
    ### a configuration for the precessing spins
    nothing
end

end