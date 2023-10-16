module green_functions
### Libraries
# include("parameters.jl")
# using .parameters
include("get_parameters.jl")
using .get_parameters
include("create_hamiltonian.jl")
import .create_hamiltonian: create_H

### Green funuction of the leads
function selfenergy(energy, thop = thop, t_ls = 1.) 
    """ This function computes the exact self energy 
    for a 1d chain of electrons 
    parameters:
    ----------
    energy: Float64
    energy of the system 
    thop: complex-Float64
    hopping parameter
    t_ls: complex Float-64
    hopping parameter that determines if the system is connected or not
    returns:
    -------
    selfenergy: ComplexF64
    Value of the self-energy 
    """
    rad = 4 * thop^2 - energy^2
    if real(rad) > 0
        selfenergy = energy - im * sqrt(rad)
    else
        if real(energy) > 0
            sgn = 1
        else
            sgn = -1
        end
        selfenergy = energy - sgn * sqrt(-rad)
    end
    selfenergy = selfenergy * (thop^2 / (2 * t_ls^2))
    return selfenergy
end

function green(vm_a1x,energy,t=1.0,jsd=1.0 ) 
    """ This function computes the Green function of the central system
    parameters:
    ----------
    energy: float64
    energy of the system 

    External parameters:
    -------------------
    n: Int64
    number of lattice sites
    n_channels: Int64
    number of channels(2 if spin)
    H: Matrix of ComplexFloat64 numbers of size n_channels^2*n^2
    
    returns:
    --------
    green: Matrix with the size of the system 
    n_channels^2*n^2
    """
    H = create_H(vm_a1x)   ## Hamiltonian it comes from global_parameters
    se = selfenergy(energy)  ## Self energy of the system
    #Addition of the self energy (Note that it is added just in the first and last site)
    H[1:2,1:2] = H[1:2,1:2] .+ se.*σ_0 #I(2)
    H[2*n-1:2*n,2*n-1:2*n] = H[2*n-1:2*n,2*n-1:2*n] .+ se.*σ_0
    green = inv(energy.*I_ab .- H ) # Green function inverting the system 
    return green
end


end