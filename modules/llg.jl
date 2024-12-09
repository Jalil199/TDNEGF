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
    J_exc = diagm(-1 => js_exc ) + diagm(1 => js_exc) ### Ferro
    # Exchange term 
    ### if J_exc>0 the the interaction is ferro
    ### positive sign in front of the variables means ferro
    hef = (  J_exc*vm_a1x + js_sd.*vs_a1x + js_ani.*([llg_params.e].⋅vm_a1x).*[llg_params.e]
          - js_dem.*([llg_params.e_demag].⋅vm_a1x).*[llg_params.e_demag])/MBOHR .+ [llg_params.h0] #+ J_exc*vm_a1x/MBOHR
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

#### Hamiltonian for CrSBr

    # Exchange matrix for NN (It can be modified for more than NN)
# j_ani_y = 0.02##0.014*3e-3 ### the factor of 3 is due to 3/2 of the spin magnitude
# j_ani_z = 0.04##0.038*3e-3
# e_y =[0.,1.,0.] 
# e_z =[0.,0.,1.] 
    # js_exc = -ones(Float64, 12-1)*0.4#*0.1*3/2
    # J_exc[6,7] = 0.
    # J_exc[7,6] = 0.
    # js_exc_2 = ones(6)*0.02#*0.013*(3/2)*1e-3
    # J_exc_2 = diagm(6 => js_exc_2)  + diagm(-6 => js_exc_2) ###AntiFerro
    # #println(11)
    # J_exc  = J_exc + J_exc_2
    # #println(22)
    # ###
    # js_sd2 = 0.2#(0.1)*3/2#e-3
    # js_sd_2 = zeros(12,12)
    # js_sd_2[1,1] = js_sd2
    # js_sd_2[4,2] = js_sd2
    # js_sd_2[2,3] = js_sd2
    # js_sd_2[5,4] = js_sd2
    # js_sd_2[3,5] = js_sd2
    # js_sd_2[6,6] = js_sd2
    # js_sd_2[7,7] = js_sd2
    # js_sd_2[10,8] = js_sd2
    # js_sd_2[8,9] = js_sd2
    # js_sd_2[11,10] = js_sd2
    # js_sd_2[9,11] = js_sd2
    # js_sd_2[12,12] = js_sd2
    #println(33)
#js_sd_2*vs_a1x
#-  j_ani_y.*([e_y].⋅vm_a1x).*[e_y] - j_ani_z.*([e_z].⋅vm_a1x).*[e_z]
    #J_exc*vm_a1x 
    #println(34)
    #js_sd_2*vs_a1x
    #println(35)
    #j_ani_y.*([e_y].⋅vs_a1x).*[e_y]
    #println(36)