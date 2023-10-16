module dynamical_variables 
### Libraries
using LinearAlgebra          ### Linear algebra library
using Tullio                 ### Library to work with tensors
# include("parameters.jl")
# using .parameters
include("get_parameters.jl")
using .get_parameters
###include("derived_constants.jl")
###using .derived_constants

function fermi(energy,beta = beta)
    """ This function evaluate the fermi energy at the temperature defined in the global 
    parameters
    """
   fermi = 1. / (1. + exp(beta*energy  ) )
end

function spectraldensity(en, epsil,wid )
    """ Computes a lorentzian distribution 
    (Here Usually the resonance and the width are fitted previously )
    args:
    ----
    en : Float64
    variable of energy where the distribution is evaluated
    epsil:Float64
    center of the lorentzian 
    wid: Float64
    width of the lorentzian 

    returns:
    -------
    lorentz: Float64
    Lorentzian distribution evaluated in energy en
    """
    lorentz = wid^2/((en-epsil)^2 + wid^2 )
    return lorentz
end

function create_hi(;eps_k1α = eps_k1α , w0_k1α = w0_k1α, E_F_α = E_F_α, Eig_vals_k2α = Eig_vals_k2α)
    """ This function creates the variable χ_{α,m,k}. This quantitie is related with the poles of the lorentzia
    paramaters:
    -------------------
    Note that the parameters are related with the evaluation of the poles 
    The first 2 parameters correspons to the elements associated with the resonance and the width of the 
    lorentzian function, the third element contain the fermi energy of the leads and the fourth element 
    contain the poles of the ozaki function 
    """
    hi_αmk = zeros(ComplexF64, 2, 2, k_poles ) 
    @tullio hi_αmk1[α, m ,k1 ] := eps_k1α[k1,α] + (-1)^m*w0_k1α[k1,α]*1im  (m in 1:2)
    @tullio hi_αmk2[α, m ,k2 ] := E_F_α[α] + (-1)^m * 1im/(Eig_vals_k2α[k2,α]*beta)  (m in 1:2)
    hi_αmk[:, :, 1:n_lorentz] .= hi_αmk1
    hi_αmk[:,:,n_lorentz+1:k_poles] .= hi_αmk2

    return hi_αmk,hi_αmk1,hi_αmk2
end

function create_Gam(;hi_αmk=hi_αmk,hi_αmk1=hi_αmk1,hi_αmk2=hi_αmk2,E_F_α = E_F_α, R_k2α= R_k2α,gam_k1iα=gam_k1iα,w0_k1α=w0_k1α,eps_k1α=eps_k1α)
    """ This function evaluates the Gamma function using the residue theorem for the 
    lorentzian and fermi poles(Ozaki decomposition) using the tullio
    library. Also is important to note that all the elements are evaluated using tensorial notation
    """
    Gam_lesser_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
    Gam_greater_αmik = zeros(ComplexF64,2, 2,n_channels, k_poles )
    
    ### Fermi function 
    @tullio fermi_m_αmk1[α,m,k1] := fermi(-hi_αmk1[α,m,k1] + E_F_α[α] )
    @tullio fermi_p_αmk1[α,m,k1] := fermi(hi_αmk1[α,m,k1] - E_F_α[α]  )
    ### Quantity related with the spectral density 
    @tullio ν_αmik2[α,m,j,k2] := spectraldensity(hi_αmk2[α,m,k2],eps_k1α[l1,α],w0_k1α[l1,α] )*gam_k1iα[l1,j,α]
    ### Definition of Gamma greater for the poles of the fermi function and the lorentzian poles 
    @tullio Gam_greater_αmik1[α, m,i, k1 ] := -0.5im*gam_k1iα[k1,i,α]*w0_k1α[k1,α]*fermi_m_αmk1[α,m,k1] # Fermi function poles
    @tullio Gam_greater_αmik2[α, m,i, k2 ] := (-1)^(m)*ν_αmik2[α,m,i,k2]*R_k2α[k2,α]/beta # Lorentzian poles 
    Gam_greater_αmik[:,:,:,1:n_lorentz] = Gam_greater_αmik1
    Gam_greater_αmik[:,:,:,n_lorentz+1:k_poles] = Gam_greater_αmik2
    ### Definition of Gamma lesser for the poles of the fermi function and the lorentzian poles
    @tullio Gam_lesser_αmik1[α, m,i, k1 ] := 0.5im*gam_k1iα[k1,i,α]*w0_k1α[k1,α]*fermi_p_αmk1[α,m,k1]
    #@tullio Gam_lesser_αmik2[α, m,i, k2 ] = (-1)^(m)*ν_αmik2[α,m,i,k2]*R_k2α[k2,α]/beta
    #Gam_lesser_αmik2 = copy(Gam_greater_αmik2 )
    Gam_lesser_αmik[:,:,:,1:n_lorentz] .= Gam_lesser_αmik1
    Gam_lesser_αmik[:,:,:,n_lorentz+1:k_poles] .= Gam_greater_αmik2 # Gam_lesser_αmik2 they are equal
    return Gam_greater_αmik, Gam_lesser_αmik 
end
function create_csi()#(;csi_L=csi_L,csi_R=csi_R)
    """ This function creates the csi eigen vector ζ that corresponds to the 
    eigen vector of the Gamma functions, note that it will be stored in tensorial
    notation
    
    returns:
    --------
    csi_aikα
    """
    csi_L = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
    csi_R = zeros(ComplexF64,n_channels*n, n_channels, k_poles  )
    csi_aikα = zeros(Float64, n*2,2,k_poles,2 )#cat(csi_L, csi_R, dims = 4)
    csi_L[1,1,:] = csi_L[2,2,:] = csi_R[2*n-1,1,:] = csi_R[2*n,2,:] .= 1.
    csi_aikα .= cat(csi_L, csi_R, dims = 4)
    #nothing
    return csi_aikα #csi_L, csi_R
end


end