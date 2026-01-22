module equation_of_motion
### Libraries
using LinearAlgebra          ### Linear algebra library
using Tullio                 ### Library to work with tensors
#include("parameters.jl")
#using .parameters
# include("get_parameters.jl")
# using .get_parameters
# include("derived_constants.jl")
# using .derived_constants

function to_matrix(vector,p)
    """ This function transform a vector into a tensor 
    """
    ##### With the macro views it joins to the link vector and it modify (It is not necesary to define it !?)
    Omega_αik1βjp1 = reshape(vector[1:p.size_Omega1], p.dims_Omega1 )
    #Omega_αik1βjp1 = reshape( Omega_αik1βjp1, dims_Omega1 )
    Omega_αik1βjp2 = reshape(vector[p.size_Omega1+1:p.size_Omega1+ p.size_Omega2],p.dims_Omega2)
    #Omega_αik1βjp2 = reshape( Omega_αik1βjp2, dims_Omega2 )
    Omega_αik2βjp1 = reshape(vector[p.size_Omega1+p.size_Omega2+1 : p.size_Omega1+p.size_Omega2+p.size_Omega3],p.dims_Omega3)
    #Omega_αik2βjp1 = reshape( Omega_αik2βjp1, dims_Omega3 )
    psi_aikα = reshape(vector[p.size_Omega1+ p.size_Omega2+p.size_Omega3+1 : p.size_Omega1+ p.size_Omega2+p.size_Omega3+ p.size_psi ],
                        p.dims_psi)
    #psi_aikα = reshape(psi_aikα, dims_psi )
    rho_ab = reshape(vector[p.size_Omega1+ p.size_Omega2+p.size_Omega3+ p.size_psi+ 1 : p.size_Omega1+ 
            p.size_Omega2+p.size_Omega3+ p.size_psi+ p.size_rho ], p.dims_rho)

    return  Omega_αik1βjp1,Omega_αik1βjp2,Omega_αik2βjp1, psi_aikα, rho_ab
end

function to_vector(dOmega_αik1βjp1,dOmega_αik1βjp2,dOmega_αik2βjp1, dpsi_aikα, drho_ab )
    """ This function transforms the Omega,rho and psi in a vector in order to solve the differential 
    equation using rk4 or other merhod 
    args:
    -----
    This function must already know the quantities 
    Omega,rho and psi
    return:
    -------
    d_vector: n dimensional array 
    this vector contains all the quantities in a vector 
    in order to solve the differential equation
    """
    dvector = vcat(vec(dOmega_αik1βjp1),vec(dOmega_αik1βjp2),vec(dOmega_αik2βjp1), vec(dpsi_aikα), vec(drho_ab) ) 
    return dvector 
end 

function eom!(du, u, p, t)
    """ This function creates the differential equation that must be solved 
    it must be putted in a vectorial form 
    args:
    ----
    du: represents the derivate of the vector 
    u: reprenset the component of the vector to be solved
    returns:
    -------
    it defines qthe differential equation 
    in this case the variable du is modified 
    to deffine the differential equation
    """
    #dOmega_αikβjp = #zeros(ComplexF64,2,n_channels,k_poles,2 ,n_channels,k_poles)
    dOmega_αik1βjp1 = Array{ComplexF64}(undef,p.dims_Omega1)#zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,n_lorentz)
    dOmega_αik1βjp2 = Array{ComplexF64}(undef,p.dims_Omega2)#zeros(ComplexF64,2,n_channels,n_lorentz,2 ,n_channels,N_poles)
    dOmega_αik2βjp1= Array{ComplexF64}(undef,p.dims_Omega3)#zeros(ComplexF64,2,n_channels,N_poles,2 ,n_channels,n_lorentz)
    summ1_aik1α = zeros(ComplexF64,2*p.n,2,p.n_lorentz,2)
    summ2_aik1α = zeros(ComplexF64,2*p.n,2,p.n_lorentz,2)
    summ3_aik2α = zeros(ComplexF64,2*p.n,2,p.N_poles,2)
    dpsi_aikα = Array{ComplexF64}(undef,p.dims_psi)#zeros(ComplexF64, 2*n, n_channels, k_poles, 2 )
    drho_ab = Array{ComplexF64}(undef,p.dims_rho)#zeros(ComplexF64, 2*n, 2*n )
     #### will be better Preallocate them? need to be checked
    H_ab::Array{ComplexF64} = p.H_ab#p[1]
    delta_αi::Matrix{Float64} = p.delta_αi#p[2]
    Pi_abα::Array{ComplexF64} = p.Pi_abα#zeros(ComplexF64, 2*n, 2*n, 2 )
    hi_αmk::Array{ComplexF64} = p.hi_αmk
    hi_αmk1::Array{ComplexF64} = p.hi_αmk1
    hi_αmk2::Array{ComplexF64} = p.hi_αmk2
    Gam_greater_αmik::Array{ComplexF64} = p.Gam_greater_αmik
    Gam_lesser_αmik::Array{ComplexF64} = p.Gam_lesser_αmik
    csi_aikα::Array{ComplexF64} = p.csi_aikα
    hbar::Float64 = p.hbar  # (eV*fs)
    n_lorentz::Int = p.n_lorentz
    size_Omega1::Int = p.size_Omega1
    size_Omega2::Int = p.size_Omega2
    size_Omega3::Int = p.size_Omega3
    size_psi::Int = p.size_psi
    size_rho::Int = p.size_rho
    p.Omega_αik1βjp1::Array{ComplexF64,6},p.Omega_αik1βjp2::Array{ComplexF64,6},p.Omega_αik2βjp1::Array{ComplexF64,6}, p.psi_aikα::Array{ComplexF64,4}, p.rho_ab::Array{ComplexF64,2} = to_matrix(u,p)
    Omega_αik1βjp1=p.Omega_αik1βjp1
    Omega_αik1βjp2=p.Omega_αik1βjp2
    Omega_αik2βjp1=p.Omega_αik2βjp1
    psi_aikα=p.psi_aikα
    rho_ab=p.rho_ab
    
    ### Notice that i take out the conjugate
    @tullio threads=true   dOmega_αik1βjp1[α,i,k1+0,β,j,p1+0] = begin 
    (   conj( csi_aikα[a,j,p1,β])*psi_aikα[a,i,k1,α]*(Gam_greater_αmik[β,1,j,p1] 
                - Gam_lesser_αmik[β,1,j,p1]) )
    end
    @tullio dOmega_αik1βjp1[α,i,k1+0,β,j,p1+0] += conj(psi_aikα[a,j,p1,β])*csi_aikα[a,i,k1,α]*(Gam_greater_αmik[α,2,i,k1] 
                                                - Gam_lesser_αmik[α,2,i,k1] ) 
    
    @tullio dOmega_αik1βjp1[α,i,k1+0,β,j,p1+0] +=-1im/$hbar*(hi_αmk[β,1,p1] + delta_αi[β,j]
                                          -(hi_αmk[α,2,k1] + delta_αi[α,i]) )*Omega_αik1βjp1[α,i,k1,β,j,p1]
    
    du[1:size_Omega1] .= vec(dOmega_αik1βjp1)
    
    @tullio  dOmega_αik1βjp2[α,i,k1+0,β,j,p2] = begin 
    (    conj(csi_aikα[a,j,$n_lorentz + p2 ,β])*psi_aikα[a,i,k1,α]*(Gam_greater_αmik[β,1,j,$n_lorentz + p2] 
                - Gam_lesser_αmik[β,1,j,$n_lorentz  + p2])  )
    end
    
    @tullio dOmega_αik1βjp2[α,i,k1+0,β,j,p2] += conj( psi_aikα[a,j,$n_lorentz + p2,β])*csi_aikα[a,i,k1,α]*(Gam_greater_αmik[α,2,i,k1] 
                                            - Gam_lesser_αmik[α,2,i,k1] )
    @tullio dOmega_αik1βjp2[α,i,k1+0,β,j,p2] +=  -1im/$hbar*(hi_αmk[β,1,$n_lorentz+ p2] 
                                + delta_αi[β,j] - ( hi_αmk[α,2,k1] + delta_αi[α,i]  ) )*Omega_αik1βjp2[α,i,k1,β,j, p2]
    
    du[size_Omega1 + 1 : size_Omega1 + size_Omega2 ] .= vec(dOmega_αik1βjp2)

    @tullio  dOmega_αik2βjp1[α,i,k2+0,β,j,p1+0] = begin 
    (   conj(csi_aikα[a,j,p1,β])*psi_aikα[a,i,$n_lorentz + k2,α]*(Gam_greater_αmik[β,1,j,p1] - Gam_lesser_αmik[β,1,j,p1])   )
    end
    @tullio dOmega_αik2βjp1[α,i,k2+0,β,j,p1+0] += conj(psi_aikα[a,j,p1,β])*csi_aikα[a,i,$n_lorentz + k2,α]*(Gam_greater_αmik[α,2,i,$n_lorentz  + k2] 
                                    - Gam_lesser_αmik[α,2,i,$n_lorentz  + k2] )
    @tullio dOmega_αik2βjp1[α,i,k2+0,β,j,p1+0] += -1im/$hbar*(hi_αmk[β,1,p1] + delta_αi[β,j] 
                                - ( hi_αmk[α,2,$n_lorentz+k2] + delta_αi[α,i]  ) )*Omega_αik2βjp1[α,i,k2,β,j,p1] 
    
    du[size_Omega1 + size_Omega2 + 1 : size_Omega1 + size_Omega2 + size_Omega3] .= vec(dOmega_αik2βjp1)

    @tullio summ1_aik1α[a,i,k1,α] = begin
         Omega_αik1βjp1[α,i,k1,β,j,p1 + 0]*csi_aikα[a,j,p1+0,β]
    end
    ###### This summ2 is defined differently in the Fortran code (IT must be checked) 
    @tullio summ2_aik1α[a,i,k1,α] = begin
         Omega_αik1βjp2[α,i,k1,β,j,p2+0]*csi_aikα[a,j,$n_lorentz+p2 ,β] #$n_lorentz + p2 in csi ### need to be checked
    end
    ######
    @tullio summ3_aik2α[a,i,k2,α] = begin
         Omega_αik2βjp1[α,i,k2,β,j,p1+0]*csi_aikα[a,j,p1,β]
    end
    #### in this case the sum must be splitted due to the sum over index i
    @tullio dpsi_aikα[a,i,k,α] = -1im*Gam_lesser_αmik[α,2,i,k]*csi_aikα[a,i,k,α] + 1im/$hbar*(hi_αmk[α,2,k]+ delta_αi[α,i])*psi_aikα[a,i,k,α]  ## check
    @tullio  dpsi_aikα[a,i,k,α] += -1im*rho_ab[a,b]*csi_aikα[b,i,k,α]*(Gam_greater_αmik[α,2,i,k] - Gam_lesser_αmik[α,2,i,k]) ##SEMIcheck or not sure by rho
    @tullio  dpsi_aikα[a,i,k,α] += -1im/$hbar*H_ab[a,b]*psi_aikα[b,i,k,α] ### check
    
    @tullio dpsi_aikα[a,i,k1+0,α] += -1im/$hbar^2*(summ1_aik1α[a,i,k1,α] + summ2_aik1α[a,i,k1,α] )  ### SEMIcheck (if rho=0)
    @tullio dpsi_aikα[a,i,k2,α]  += -1im/$hbar^2*summ3_aik2α[a,i,k2-$n_lorentz,α] ### SEMIcheck (if rho=0)
    
    du[size_Omega1 + size_Omega2 + size_Omega3 + 1 : size_Omega1 + size_Omega2 + size_Omega3 + size_psi ] .= vec(dpsi_aikα)

    @tullio Pi_abα[a,b,β] = psi_aikα[a,i,k,β]*conj(csi_aikα[b,i,k,β] )/$hbar
    ### It must be splited in two expression due to the over sum of @tullio
    @tullio drho_ab[a,b] =  -1im/$hbar*(H_ab[a,c]*rho_ab[c,b])
    @tullio drho_ab[a,b] += 1im/$hbar*(rho_ab[a,c]* H_ab[c,b])   #+
    @tullio drho_ab[a,b] += 1/$hbar*(Pi_abα[a,b,β]+ conj(Pi_abα[b,a,β]) )
    du[ size_Omega1 + size_Omega2 + size_Omega3 + size_psi + 1 : size_Omega1 + size_Omega2 + size_Omega3 + size_psi + size_rho ] .= vec(drho_ab)
    #println("elements of PI_L:", Pi_abα[:,:,1] )#+ Pi_abα[:,:,1]' )#Pi_abα[1,1,1],  Pi_abα[2,2,1],Pi_abα[3,3,1], Pi_abα[4,4,1]  )
    #println("elements of H:", H_ab[1,1],  H_ab[2,2],H_ab[3,3], H_ab[4,4]  )
    ##println("elements of ψ:", psi_aikα[:,1,1,1], psi_aikα[:,1,end,1] )
    ##println("elements of χ:", hi_αmk[1,1,:] )
    #println("Greater of ψ:", psi_aikα[findall(x -> x > 100, psi_aikα[1,1,:,1]) ] )
    #println("elements of ζ:", csi_aikα[:,1,1,1], csi_aikα[:,1,end,1] )
    #println("elements of rho:", rho_ab[1,1],  rho_ab[2,2], rho_ab[3,3], rho_ab[4,4]  )
    nothing
end


end
