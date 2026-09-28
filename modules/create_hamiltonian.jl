module create_hamiltonian
### Libraries
using LinearAlgebra
using Tullio
⊗(A,B) = kron(A,B)

function A_light(gv)
    "This function returns a pulse of the light"
    ts=0.0:gv.t_step:gv.t_end
    return gv.A_max*exp.(-((ts.-gv.t_p).^2)/(2*gv.sigma^2)).*sin.(gv.Omega_0*ts)
end
### Block Hamiltonian 
function Block_H(gv,cv)
    #γ::Float64,γso::ComplexF64,Bz::Float64,ny::Int)
    "Creates the building blocks for a general nx x ny square lattice "
    dim = gv.ny*2 # We include the spin degree of freedom 
    ######
    H0 = zeros(ComplexF64,dim,dim)
    T  = zeros(ComplexF64,dim,dim)
    One_y = Diagonal(ones(gv.ny))
    ######
    Ty = diagm(-1 =>  ones(gv.ny-1))
    T0 = Ty⊗(-cv.tc1*gv.σ_0 - 1im*gv.alpha_r*gv.σ_x)
    H0 .= T0 + T0' #-Bz*kron(One_y, σ_z)
    ######
    T .= One_y⊗(-cv.tc1*gv.σ_0 + gv.alpha_r*1im*gv.σ_y)
    ###### Addition of local spin 

    return H0, T
end
### Central Hamiltonian
function Central_H(vm_a1x::Array{Float64,2}; gv, cv)
    "This function build the central hamiltonian wwith two band"
    #γ::Float64,γso::ComplexF64,Bz::Float64,nx::Int,ny::Int)
    dim = gv.nx*gv.ny*2 #*2
    zero = zeros(ComplexF64,gv.nx,gv.nx)
    HC = zeros(ComplexF64,dim,dim)
    One_x = Diagonal(ones(gv.nx))
    H0,T = Block_H(gv,cv)
    Tx = diagm( -1 =>  ones(gv.nx-1))⊗T #, 1 =>  ones(nx-1))
    HC = (One_x⊗H0) +  Tx + Tx'
    ### Local moments
    for i in range(1,gv.nx) 
        zero[i,i] = 1.0
        HC += zero⊗(vm_a1x[i,1]*gv.σ_x
                    +vm_a1x[i,2]*gv.σ_y
                    +vm_a1x[i,3]*gv.σ_z)
        zero[i,i] = 0.0
    end
    
    return HC
end



function create_H(vm_a1x; global_var , config_var)#(vm_a1x, m_qsl = nothing )
    """ This function creates the Hamiltonian of the central system 
    Note that in general vm_a1x depends on the specific time, and corresponds
    to the classical spin density 
    """
    m_x = zeros(Float64,dim,dim)
    m_y = zeros(Float64,dim,dim)
    m_z = zeros(Float64,dim,dim)           #js_sd_c = [gv.j_sd, gv.j_sd]
    @tullio m_x[c1,c1] = -js_sd_c[c1]*vm_cx[c1,1]
    @tullio m_y[c1,c1] = -js_sd_c[c1]*vm_cx[c1,2]
    @tullio m_z[c1,c1] = -js_sd_c[c1]*vm_cx[c1,3]
    H0 += m_x⊗gv.σ_x + m_y⊗gv.σ_y + m_z⊗gv.σ_z
    dim = gv.nx*gv.ny*2
    H = zeros(Float64, 40,40)
    #H = Central_H(γ,γso,Bz,nx,ny)
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




end
