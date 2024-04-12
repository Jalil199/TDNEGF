module create_hamiltonian
### Libraries
using LinearAlgebra
# include("parameters.jl")
# using .parameters
# include("get_parameters.jl")
# using .get_parameters
# include("derived_constants.jl")
# using .derived_constants

function create_H(vm_a1x; global_var , config_var, m_qsl = nothing )#(vm_a1x, m_qsl = nothing )
    """ This function creates the Hamiltonian of the central system 
    Note that in general vm_a1x depends on the specific time, and corresponds
    to the classical spin density 
    """
    #### For the moment we will only have quadratic hamiltonians 
    #### Rice Mele Hamiltonian
    ### parameters


    DD = 1. 
    diagonal_0 = zeros(global_var.n)
    diagonal_0[1:2:end] .= DD/2
    diagonal_0[2:2:end] .= -DD/2
    
    H = diagm(-1 =>  config_var.thops) .+ diagm(1 =>  config_var.thops) .+ diagm(0 => diagonal_0)
    
    # Hopping hamiltonian
    #hops = -thop_local#.*ones(n-1) 
    #H = -(diagm(-1 =>  config_var.thops) .+ diagm(1 =>  config_var.thops))
    # Include the spin degree of freedom 
    H_so = -(diagm(-1 =>  config_var.thops_so*im ) .+ diagm(1 =>  config_var.thops_so*(-im) ))
    H = kron(H,global_var.σ_0) + kron(H_so, global_var.σ_y)
    m_a1x = hcat(vm_a1x...)#[:, :]
    # if we want to include the j_sd depending on the sites, we just must take the dot product 
    # m_x = -J_sd_local.*diagm(m_a1x[1,:]) # x component in a matrix form 
    # m_y = -J_sd_local.*diagm(m_a1x[2,:]) # y component in a matrix form
    # m_z = -J_sd_local.*diagm(m_a1x[3,:]) # z component in a matrix form
    m_x = -config_var.Js_sd.*diagm(m_a1x[1,:]) # x component in a matrix form 
    m_y = -config_var.Js_sd.*diagm(m_a1x[2,:]) # y component in a matrix form
    m_z = -config_var.Js_sd.*diagm(m_a1x[3,:]) # z component in a matrix form
    H +=    ( kron(m_x, global_var.σ_x ) .+
                kron(m_y, global_var.σ_y ) .+
                kron(m_z, global_var.σ_z ) )
    ### This is applied when the system is coupled to the spin liquid
    if m_qsl != nothing
            m_qsl_a1x = hcat(m_qsl...)
            m_qsl_x = diagm(m_qsl_a1x[1,:]) # x component in a matrix form 
            m_qsl_y = diagm(m_qsl_a1x[2,:]) # y component in a matrix form
            m_qsl_z = diagm(m_qsl_a1x[3,:]) # z component in a matrix form
            H += -global_params.J_qsl.*( kron(m_qsl_x, global_var.σ_x ) .+
                                         kron(m_qsl_y, global_var.σ_y ) .+
                                         kron(m_qsl_z, global_var.σ_z ) )
    end
        
    return H
end


end
