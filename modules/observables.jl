module observables 
using LinearAlgebra     ### Linear algebra library
using Tullio            ### Library to work with tensors

include("get_parameters.jl")
using .get_parameters
##include("parameters.jl")
##using .parameters
include("derived_constants.jl")
using .derived_constants
include("equation_of_motion.jl")
import .equation_of_motion: to_matrix
# include("green_functions.jl")
# import .green_functions: green
# include("equilibrium_variables.jl")
# import .equilirbrium_variables: rho_denis


function Observables(vector,params=params,just_rho = false)#; vm_a1x =  [zeros(Float64,3) for _ in 1:n], H_ab = zeros(ComplexF64,2*n,2*n) )
    """ return a Dictionary with the observables calculated by the system
    """
    ### Need to be checkced if it is better to preallocate the variables or define 
    ### the variables globally ?
    if just_rho == true
        rho_ab = vector
        sden_xa1 = zeros(Float64, 3, n);
        sden_xab = zeros(ComplexF64, 3,2*n,2*n) ;
    else 
        Omega_αik1βjp1 = Array{ComplexF64}(undef,dims_Omega1)
        Omega_αik1βjp2 = Array{ComplexF64}(undef,dims_Omega2)
        Omega_αik2βjp1= Array{ComplexF64}(undef,dims_Omega3)
        psi_aikα = Array{ComplexF64}(undef,dims_psi)
        rho_ab = Array{ComplexF64}(undef,dims_rho)
        Pi_abα = zeros(ComplexF64, 2*n, 2*n, 2 )
        Omega_αik1βjp1,Omega_αik1βjp2,Omega_αik2βjp1, psi_aikα, rho_ab = to_matrix(vector)
        
        sden_xa1 = zeros(Float64, 3, n);                              ### spin density
        curr_α = zeros(Float64,2)
        scurr_xα = zeros(Float64, 3,2)
        sden_xab = zeros(ComplexF64, 3,2*n,2*n) ;
        @tullio Pi_abα[a,b,β] = psi_aikα[a,i,k,β]*conj(csi_aikα[b,i,k,β] )/hbar
    end
    #println("join to observables___ inifite loop?")
    #rho_ozaki = rho_denis(green,0.0, 1.0, vm_a1x, Temp, 1.0 )
    return_params = Dict( )  
    if params["curr"] == true                                          ### Current
        ##### Current
        @tullio curr_α[α] = 4*pi*real(Pi_abα[a,a,α]) 
        ccurr = 0.5* (curr_α[1] - curr_α[2])
        currs = [ccurr, curr_α]                                         ### Total charge current and Current_left and Current_right
        return_params["curr"] = currs
    end
    if params["scurr"] == true
        ##### Spin_Current
        #scurr_xα=zeros(ComplexF64, 3,2)
        @tullio scurr_xα[x,α] =  real <| 4*pi*σ_abx[a,b,x]*Pi_abα[b,a,α] ### Note that sigma_full must be computed 
        return_params["scurr"] = scurr_xα
    end
    if params["sden"] == true
        ##### Spin density 
        @tullio sden_xab[x,a,b] = rho_ab[a,c]*σ_abx[c,b,x]              
        @tullio sden_xa1[x,a1] = real(sden_xab[x,2a1-1,2a1-1] + sden_xab[x,2a1,2a1])
        #sden_xa1 ### Numbers that contains the localized spin density 
                  ### note that this term must be evolved in each time step 
                  ### otherwise sden_xab can be taken directly 
        vsden_xa1 = [sden_xa1[:, i] for i in 1:n :: Int]
        return_params["sden"] = vsden_xa1
    end
    # if params["cden"]
    #     ### Here the charge charge density that remain out of equilibrium is calculated 
    #     cden = zeros(Float64, n )
    #     for i in range(1, n)
    #         cden[i] = tr(rho_ab[2*i-1:2*i, 2*i-1:2*i]  - rho_ozaki[2*i-1:2*i, 2*i-1:2*i] )
    #     end
    #     return_params["cden"] = real(cden)
        
    # end
    
    

    # if params["bcurr"]
    #     ### Here the bonds currents are calculated
    #     #@tullio bcurr[a1] =-2*pi*im*rho_ab[a1,2a1] 
    #     #vm_a1x,energy_llg; t = 1.0, Temp = Temp energy==0.0, t==1.0, [[jsd=1.0 it is never used ! ]] 
    #     # This is related with the density matrix in equilibrium 
    #     cc = zeros(Float64, n-1)
    #     cx = zeros(Float64, n-1)
    #     cy = zeros(Float64, n-1)
    #     cz = zeros(Float64, n-1)
    #     for i in range(1,n-1)
    #         ### I need to add rho_ozaki but first lets see if it works 
    #         rho_u = rho_ab[2*i-1:2*i, 2*i+1:2*i+2]  - rho_ozaki[2*i-1:2*i, 2*i+1:2*i+2]
    #         ham_u = H_ab[2*i+1:2*i+2, 2*i-1:2*i] 
    #         rho_d = rho_ab[2*i+1:2*i+2, 2*i-1:2*i] - rho_ozaki[2*i+1:2*i+2, 2*i-1:2*i]
    #         ham_d = H_ab[2*i-1:2*i, 2*i+1:2*i+2]
    #         cc_m = -2*pi*im*(rho_u*ham_u - rho_d*ham_d )
    #         cc[i] = tr(cc_m) 
    #         cx[i] = tr(σ_x*cc_m) 
    #         cy[i] = tr(σ_y*cc_m) 
    #         cz[i] = tr(σ_z*cc_m) 
    #     end
    #     return_params["bcurr"] = real([cc, cx, cy, cz ])
            
    # end

    
    return return_params
end

end
