module create_hamiltonian
### Libraries
using LinearAlgebra
# include("parameters.jl")
# using .parameters
include("get_parameters.jl")
using .get_parameters
include("derived_constants.jl")
using .derived_constants

function create_H(vm_a1x, m_qsl = nothing )
    """ This function creates the Hamiltonian of the central system 
    Note that in general vm_a1x depends on the specific time, and corresponds
    to the classical spin density 
    """
   # tso1=0.1
   # tso2=0.1
   # hops_tso1 = zeros(ComplexF64,(n,n))
   # hops_tso1[1,2] = -tso1*im
   # hops_tso1[2,1] =  tso1*im
   # hops_tso2 = zeros(ComplexF64,(n,n))
   # hops_tso2[6,7] = -tso2*im
   # hops_tso2[7,6] =  tso2*im
    #### For the moment we will only have quadratic hamiltonians 
    # Hopping hamiltonian
    #hops = -thop_local#.*ones(n-1) 
    H = -(diagm(-1 =>  thop_local) .+ diagm(1 =>  thop_local))
    # Include the spin degree of freedom 
    H_so = -(diagm(-1 =>  tso_local*im ) .+ diagm(1 =>  tso_local*(-im) ))
    H = kron(H,σ_0) + kron(H_so, σ_y)
    m_a1x = hcat(vm_a1x...)#[:, :]
    # if we want to include the j_sd depending on the sites, we just must take the dot product 
    #println(m_a1x[1,:])
    m_x = -J_sd_local.*diagm(m_a1x[1,:]) # x component in a matrix form 
    m_y = -J_sd_local.*diagm(m_a1x[2,:]) # y component in a matrix form
    m_z = -J_sd_local.*diagm(m_a1x[3,:]) # z component in a matrix form
    #println(m_x)
    H +=    ( kron(m_x, σ_x ) .+
                kron(m_y, σ_y ) .+
                kron(m_z, σ_z ) )
    ### This is applied when the system is coupled to the spin liquid
    if m_qsl != nothing
            m_qsl_a1x = hcat(m_qsl...)
            m_qsl_x = diagm(m_qsl_a1x[1,:]) # x component in a matrix form 
            m_qsl_y = diagm(m_qsl_a1x[2,:]) # y component in a matrix form
            m_qsl_z = diagm(m_qsl_a1x[3,:]) # z component in a matrix form
            H += -J_qsl.*( kron(m_qsl_x, σ_x ) .+
            kron(m_qsl_y, σ_y ) .+
            kron(m_qsl_z, σ_z ) )
    end
        
    return H
end


end
