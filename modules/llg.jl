module llg
### Libraries
using LinearAlgebra          ### Linear algebra library
include("parameters.jl")
using .parameters

mutable struct llg_parameters
    """ Here are contained the main parameters of the LLG evolution
    """
    n_sites::Int
    nt::Int
    dt::Float64
    h0::Vector{Float64} 
    j_exc::Float64
    g_lambda::Float64
    j_sd::Float64
    j_dmi::Float64
    j_ani::Float64
    j_dem::Float64
    js_pol::Float64
    js_ana::Float64
    thop::Float64
    e::Vector{Float64} 
    e_demag::Vector{Float64} 
    js_exc::Vector{Float64} 
    js_sd::Vector{Float64} 
    js_ani::Vector{Float64} 
    js_dem::Vector{Float64} 
    p_theta::Float64
    p_phi::Float64
    function llg_parameters( n_sites = n_sites, nt=nt, dt=dt, h0=h0, j_exc=j_exc, g_lambda=g_lambda,
            j_sd=j_sd, j_dmi=j_dmi, j_ani=j_ani,j_dem=j_dem, js_pol=js_pol, js_ana=js_ana, thop=thop, e=e,
            e_demag=e_demag,js_exc=ones(Float64, n_sites-1)*j_exc,js_sd=ones(Float64, n_sites)*j_sd,js_ani=ones(Float64, n_sites)*j_ani,
            js_dem=ones(Float64, n_sites)*j_dem, p_theta=0., p_phi=0.)  # Outer constructor for default values
        new(n_sites, nt, dt, h0 ,j_exc, g_lambda, j_sd, j_dmi, j_ani, j_dem, js_pol, js_ana, thop, e , e_demag,
            js_exc, js_sd , js_ani, js_dem,p_theta, p_phi)
    end
end

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
    js_exc = llg_params().js_exc
    js_sd = llg_params().js_sd
    js_ani = llg_params().js_ani
    js_dem = llg_params().js_dem
    e = llg_params().e
    e_demag = llg_params().e_demag
    # Exchange matrix for NN (It can be modified for more than NN)
    J_exc = diagm(-1 => js_exc ) + diagm(1 => js_exc ) 
    # Exchange term 
    hef = (J_exc*vm_a1x + js_sd.*vs_a1x + js_ani.*([e].⋅vs_a1x).*[e]
          - js_dem.*([e_demag].⋅vs_a1x).*[e_demag])/MBOHR .+ [h0] #+ J_exc*vm_a1x/MBOHR
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
    hef = heff(vm_a1x,vs_a1x, llg_params )
    g_lambda = llg_params().g_lambda
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