module llg
### Libraries
using LinearAlgebra          ### Linear algebra library
### Effective hamiltonian

function heff(vm_a1x,vs_a1x, llg_params) 
    """ This function computes the effective hamiltonian 
    of the LLG equations
    parameters:
    -----------
    return:
    ------
    heff: vector of vectors
    """
    MBOHR = 5.788381e-5         ### Bohrs magneton
    js_exc = llg_params.js_exc
    js_sd = llg_params.js_sd
    js_ani = llg_params.js_ani
    js_dem = llg_params.js_dem
    e = llg_params.e
    e_demag = llg_params.e_demag
    # Exchange matrix for NN (It can be modified for more than NN)
    J_exc = diagm(-1 => js_exc ) + diagm(1 => js_exc ) 
    # Exchange term 
    hef = (J_exc*vm_a1x + js_sd.*vs_a1x + js_ani.*([llg_params.e].⋅vs_a1x).*[llg_params.e]
          - js_dem.*([llg_params.e_demag].⋅vs_a1x).*[llg_params.e_demag])/MBOHR .+ [llg_params.h0] #+ J_exc*vm_a1x/MBOHR
    # Note that most of the quantities are defined locally(Magnetic field and so)
    # But they can be generalized to more elements
    return hef
end
### Evolution and propagation
function corrector(vm_a1x,vs_a1x,llg_params)
    """This function calculates the correction associated to the 
    evolution in the heun propagation: 
    parameters:
    ----------
    returns:
    -------
    del_m
    """
    GAMMA_R = 1.760859644e-4    ### Gyromagnetic ratio ##1 for units of G_r=1
    hef = heff(vm_a1x,vs_a1x, llg_params )
    g_lambda = llg_params.g_lambda
    sh = vm_a1x .× hef
    shh = vm_a1x .× sh
    del_m = @. (-GAMMA_R/(1. + g_lambda^2) )*(sh +  g_lambda*shh)
    return del_m
end
function heun(vm_a1x,vs_a1x, dt, llg_params )
    """ This function propagates the vector vm_a1x in a time step dt
    using heuns method (RK2)
    """
    vm_a1x = normalize.(vm_a1x)
    del_m = corrector(vm_a1x,vs_a1x,llg_params)
    vm_a1x_prime = vm_a1x + del_m*dt
    vm_a1x_prime = normalize.(vm_a1x_prime)
    del_m_prime = corrector(vm_a1x_prime,vs_a1x , llg_params)
    vm_a1x = vm_a1x + 0.5*(del_m + del_m_prime )*dt
    vm_a1x = normalize.(vm_a1x)
    return vm_a1x
end


end