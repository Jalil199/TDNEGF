module create_hamiltonian
### Libraries
using LinearAlgebra
function create_H(vm_a1x; global_var , config_var, m_qsl = nothing )#(vm_a1x, m_qsl = nothing )
    """ This function creates the Hamiltonian of the central system 
    Note that in general vm_a1x depends on the specific time, and corresponds
    to the classical spin density 
    """
    #### For the moment we will only have quadratic hamiltonians 
    #### Rice Mele Hamiltonian
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
    m_a1x = hcat(vm_a1x...)
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


 # #### Hamiltonian of CrBr
 #    t_0c=-1##0.478
 #    t_0b=1##0.353
 #    t_c=0.2##31e-3
 #    t_b=0.4#84e-3
 #    n_c = Int(global_var.n/4)
 #    n_b = Int(global_var.n/4)
 #    diag_c = zeros(n_c-1)
 #    diag_c .= t_0c
 #    diag_b = zeros(n_b-1)
 #    diag_b.= t_0b
 #    diag_c0=zeros(n_c) 
 #    diag_b0=zeros(n_b)
 #    diag_c0.=3.5
 #    diag_b0.=-3.5
 #    h_c = diagm(-1 => diag_c) .+ diagm(1 => diag_c).+ diagm(0 => diag_c0 ) ### + (local hoppins Jsd)
 #    h_b = diagm(-1 => diag_b) .+ diagm(1 => diag_b).+ diagm(0 => diag_b0 ) ### + (local hoppins Jsd)
 #    diag_hops_c = zeros(n_c)
 #    diag_hops_b = zeros(n_b)
 #    diag_hops_c .= t_c
 #    diag_hops_b .= t_b
 #    hop_chain_c12 = diagm(0 => diag_hops_c)
 #    hop_chain_b12 = diagm(0 => diag_hops_b)
 #    ## Structure Matrices 
 #    # S_ij = zeros(4,4)
 #    S11 = zeros(4,4) ; S11[1,1]= 1.
 #    S22 = zeros(4,4) ; S22[2,2]= 1.
 #    S33 = zeros(4,4) ; S33[3,3]= 1.
 #    S44 = zeros(4,4) ; S44[4,4]= 1.
 #    S31 = zeros(4,4) ; S31[3,1]= 1.
 #    S42 = zeros(4,4) ; S42[4,2]= 1.
 #    Hc = kron(S11, h_c) +kron(S33, h_c)
 #    Hb = kron(S22,h_b) + kron(S44, h_b)
 #    H_hops_c = kron(S31,hop_chain_c12 )
 #    H_hops_b = kron(S42,hop_chain_b12 )
 #    t_light = config_var.thops[1]
 #    H  = Hc + Hb + (H_hops_c + H_hops_b)*t_light + ((H_hops_c + H_hops_b)*t_light)' 
 #    # Hopping hamiltonian
 #    # Include the spin degree of freedom 
 #    H_so = -(diagm(-1 =>  config_var.thops_so*im ) .+ diagm(1 =>  config_var.thops_so*(-im) ))
 #    H = kron(H,global_var.σ_0) + kron(H_so, global_var.σ_y)
 #    m_a1x = hcat(vm_a1x...)#[:, :]
 #    # if we want to include the j_sd depending on the sites, we just must take the dot product 
 #    js_sd2 = 0.2#(0.1)#*3/2#*1e-3
 #    js_sd_2 = zeros(12,12)
 #    js_sd_2[1,1] = js_sd2
 #    js_sd_2[2,4] = js_sd2
 #    js_sd_2[3,2] = js_sd2
 #    js_sd_2[4,5] = js_sd2
 #    js_sd_2[5,3] = js_sd2
 #    js_sd_2[6,6] = js_sd2
 #    js_sd_2[7,7] = js_sd2
 #    js_sd_2[8,10] = js_sd2
 #    js_sd_2[9,8] = js_sd2
 #    js_sd_2[10,11] = js_sd2
 #    js_sd_2[11,9] = js_sd2
 #    js_sd_2[12,12] = js_sd2

end
