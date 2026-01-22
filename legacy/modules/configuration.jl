module configuration
### Libraries
using LinearAlgebra
function configure!(t, dynamics_var;global_var,config_var)#(cname, llg_params, vm_a1x, pr_spins, t )
    """ This function stablish the configuration 
    of the magnetic moments and the parameters of the configurations
    modificating the mutable structure config_var and using the global 
    configuration global_var
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
    #println("1234344")
    ##############################################################################
    if global_var.cspin_orientation ==  "arb_dir"
        if t == 0.0 #### Initial conditions 
            #println("Holiii")
            println("join configuration: $(global_var.cspin_orientation)")
                    ### Initiallize the values 
            for jj in 1: global_var.n::Int  ### Run over the number of sites of the lattice 
                dynamics_var.vm_a1x[jj][1] = 1.0 #0.1 + (rand()-0.5)*0.1
                dynamics_var.vm_a1x[jj][2] = 0.0 #0.1 + (rand()-0.5)*0.1
                dynamics_var.vm_a1x[jj][3] = 0.0 #0.9#(-1)^(jj+1)#1.
                #println(dynamics_var.vm_a1x)
            end
        end
        # if t == 0.0 #### Initial conditions 
        #     println("join configuration: $(global_var.cspin_orientation)")
        #             ### Initiallize the values 
        #     for jj in 1: 6#global_var.n::Int  ### Run over the number of sites of the lattice 
        #         dynamics_var.vm_a1x[jj][1] = 0.9
        #         dynamics_var.vm_a1x[jj][2] = 0.1
        #         dynamics_var.vm_a1x[jj][3] = 0.1
        #     end
        #     for jj in 7: 12#global_var.n::Int  ### Run over the number of sites of the lattice 
        #         dynamics_var.vm_a1x[jj][1] = -0.9
        #         dynamics_var.vm_a1x[jj][2] = -0.1
        #         dynamics_var.vm_a1x[jj][3] = -0.1
        #     end
        # end
        ### Global parameters ares modified
        ### Hamiltonian parameters can be modified in wanted
        config_var.thops .= global_var.thop.*ones(global_var.n-1) 
        config_var.Js_sd .= global_var.j_sd.*ones(global_var.n) 
        ### LLG parameters can be modified if wanted
        config_var.js_exc .= ones(Float64, global_var.n_sites-1)*global_var.j_exc #.= ones(Float64, n_sites-1)*j_exc
        config_var.js_sd .= ones(Float64, global_var.n_sites)*global_var.j_sd #.= ones(Float64, n_sites)*j_sd
        #println(dynamics_var.pr_spins[1])
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        # J_sd_local .= J_sd.*ones(n) 
        # thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        # llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        # llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        #println("start to precess")
        ### put the spins to precces
        for jj in 1: global_var.n_precessing::Int 
            dynamics_var.pr_spins[jj].i = jj ## lattice site 
            dynamics_var.pr_spins[jj].theta_zero = global_var.theta_1
            dynamics_var.pr_spins[jj].axis_phi = global_var.phi_1
            dynamics_var.pr_spins[jj].T = global_var.period 
            #println(pr_spins[jj].i)
        end
        for j in 1:global_var.n_precessing::Int #length(pr_spins) 3:1:5
            update!(dynamics_var.pr_spins[j], t )
            dynamics_var.vm_a1x[dynamics_var.pr_spins[j].i ] .= dynamics_var.pr_spins[j].s
        end   
    end
    ##############################################################################
    ##############################################################################
    if global_var.cspin_orientation ==  "light_pulse"
        #println("Holiii")
        if t == 0.0 #### Initial conditions 
            #println("Holiii")
            println("join configuration: $(global_var.cspin_orientation)")
                    ### Initiallize the values 
            for jj in 1: global_var.n::Int  ### Run over the number of sites of the lattice 
                dynamics_var.vm_a1x[jj][1] = 0.9 #0.1 + (rand()-0.5)*0.1
                dynamics_var.vm_a1x[jj][2] =  0.2#(rand()-0.5)*0.1
                dynamics_var.vm_a1x[jj][3] =  0.2#(rand()-0.5)*0.1#0.0 #0.9#(-1)^(jj+1)#1.
            end
        end
        #println("11")
        ### Global parameters ares modified
        ### Here we use the parameters of the Rice-Mele model 
        gamma = 1.
        B = 1
        t1 = -gamma - B/2
        t2 = -gamma + B/2
        diagonal_1 = zeros(global_var.n-1)
        diagonal_1[1:2:end] .= t1
        diagonal_1[2:2:end] .= t2
        ### Hamiltonian parameters can be modified if wanted
        config_var.thops .= diagonal_1#global_var.thop.*ones(global_var.n-1) 
        
        config_var.Js_sd .= global_var.j_sd.*ones(global_var.n) 
        ### LLG parameters can be modified if wanted
        config_var.js_exc .= ones(Float64, global_var.n_sites-1)*global_var.j_exc #.= ones(Float64, n_sites-1)*j_exc
        config_var.js_sd .= ones(Float64, global_var.n_sites)*global_var.j_sd #.= ones(Float64, n_sites)*j_sd
        #println(dynamics_var.pr_spins[1])
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        # J_sd_local .= J_sd.*ones(n) 
        # thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        # llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        # llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        #println("start to precess")
        ### put the spins to precces
        for jj in 1: global_var.n_precessing::Int 
            dynamics_var.pr_spins[jj].i = jj ## lattice site 
            dynamics_var.pr_spins[jj].theta_zero = global_var.theta_1
            dynamics_var.pr_spins[jj].axis_phi = global_var.phi_1
            dynamics_var.pr_spins[jj].T = global_var.period 
            #println(pr_spins[jj].i)
        end
        for j in 1:global_var.n_precessing::Int #length(pr_spins) 3:1:5
            update!(dynamics_var.pr_spins[j], t )
            dynamics_var.vm_a1x[dynamics_var.pr_spins[j].i ] .= dynamics_var.pr_spins[j].s
        end   
    end
    ##############################################################################
##############################################################################
    if global_var.cspin_orientation ==  "light_pulse_af"
        if t == 0.0 #### Initial conditions 
            println("join configuration: $(global_var.cspin_orientation)")
                    ### Initiallize the values 
            for jj in 1: global_var.n::Int  ### Run over the number of sites of the lattice 
                dynamics_var.vm_a1x[jj][1] = 0.
                dynamics_var.vm_a1x[jj][2] = 0.
                dynamics_var.vm_a1x[jj][3] = 1#(-1)^(jj+1)
            end
        end
        ### Global parameters ares modified
        ### Here we use the parameters of the Rice-Mele model 
        gamma = 1.
        B = 1
        t1 = -gamma - B/2
        t2 = -gamma + B/2
        diagonal_1 = zeros(global_var.n-1)
        diagonal_1[1:2:end] .= t1
        diagonal_1[2:2:end] .= t2
        ### Hamiltonian parameters can be modified if wanted
        config_var.thops .= diagonal_1#global_var.thop.*ones(global_var.n-1) 
        
        config_var.Js_sd .= global_var.j_sd.*ones(global_var.n) 
        ### LLG parameters can be modified if wanted
        config_var.js_exc .= ones(Float64, global_var.n_sites-1)*global_var.j_exc #.= ones(Float64, n_sites-1)*j_exc
        config_var.js_sd .= ones(Float64, global_var.n_sites)*global_var.j_sd #.= ones(Float64, n_sites)*j_sd
        #println(dynamics_var.pr_spins[1])
        #####################################
        ### Modify the local parameters in the electron Hamiltonian
        # J_sd_local .= J_sd.*ones(n) 
        # thop_local .= thop.*ones(n-1) 
        ### Modify the llg parameter (Note that more parameters can be modified around the evolution)
        ## Notice that j_sd is taken from the global parameters but it can be modified
        # llg_params().js_exc .= ones(Float64, n_sites-1)*j_exc
        # llg_params().js_sd .= ones(Float64, n_sites)*j_sd
        #####################################
        #println("start to precess")
        ### put the spins to precces
        for jj in 1: global_var.n_precessing::Int 
            dynamics_var.pr_spins[jj].i = jj ## lattice site 
            dynamics_var.pr_spins[jj].theta_zero = global_var.theta_1
            dynamics_var.pr_spins[jj].axis_phi = global_var.phi_1
            dynamics_var.pr_spins[jj].T = global_var.period 
            #println(pr_spins[jj].i)
        end
        for j in 1:global_var.n_precessing::Int #length(pr_spins) 3:1:5
            update!(dynamics_var.pr_spins[j], t )
            dynamics_var.vm_a1x[dynamics_var.pr_spins[j].i ] .= dynamics_var.pr_spins[j].s
        end   
    end
    ##############################################################################
    nothing
end


function update!(this, time)#(this::PrecSpin, time  )
    """ This function  update the magnetic moment associated
    to the mutable structure PrecSpin 
    
    parameters:
    ----------
    this: mutable structure 
    contain an structure with the characteristics of a spin 
    time: Float64
    time where the spin is evaluated 
    
    returns:
    -------
    Update the strucure associated to a  precessing spin 
    """
    if time >= this.start_time
        t = time - this.start_time
    else
        t = 0.0
    end
    omega = 2*pi / this.T
    otheta = pi* this.theta_zero/180.
    ophi = pi*this.phi_zero /180. ##########
    # compute spin position for precession along the z-axis
    sz = cos(otheta)
    sx = cos(ophi+ omega*t)*sin(otheta)
    sy = sin(ophi+ omega*t)*sin(otheta)
    #Now rotate along y 
    atheta = pi * this.axis_theta/ 180.
    aphi = pi * this.axis_phi / 180. 
    sx = sx*cos(atheta) - sz* sin(atheta)
    sz = sx* sin(atheta) + sz*cos(atheta)
    #No rotate along the z 
    sx = sx*cos(aphi) + sy*sin(aphi)
    sy = -sx*sin(aphi) + sy*cos(aphi)
    # sx = 0.
    # sy = 0. 
    # sz = 1. + sin(omega*t)*0.5 
    this.s .= [sx, sy, sz]
    nothing
end





end
