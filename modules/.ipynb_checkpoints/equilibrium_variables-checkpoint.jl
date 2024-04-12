module equilibrium_variables
### Libraries
using Tullio
using LinearAlgebra

################ Green function of the leads
function selfenergy(energy, global_var)# thop = thop, t_ls = 1.) 
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
    ### Note that this configuration for the self energy can be modified later
    thop = global_var.thop
    t_ls = 1.
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

function green(H,energy,global_var)#(vm_a1x,energy )###,t=1.0,jsd=1.0 ) 
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
    #H = create_H(vm_a1x)   ## Hamiltonian it comes from global_parameters
    se = selfenergy(energy,global_var)  ## Self energy of the system
    #Addition of the self energy (Note that it is added just in the first and last site)
    H[1:2,1:2] = H[1:2,1:2] .+ se.*global_var.σ_0 #I(2)
    H[2*global_var.n-1:2*global_var.n,2*global_var.n-1:2*global_var.n] = H[2*global_var.n-1:2*global_var.n,2*global_var.n-1:2*global_var.n] .+ se.*global_var.σ_0
    green = inv(energy.*global_var.I_ab .- H ) # Green function inverting the system 
    return green
end

##########################

function fermi_mu(mu, Temp, energy)
    """ This function computes the fermi energy used in the computation
    of equilibrium quantities
    """
    #KB = 8.6173324e-5           ### Bolzmann factor
   1/(1. + exp((energy-mu)/(Temp)  ))   
end

function spindensity_eq(H, eq_var,global_var)#(vm_a1x, energy; t,Temp,jsd=1.0, solver = solver)
    """ This function computes the spin density in equilirbrium 
    """
    # if solver == "ozaki"
    # ## The ozaki calculation should be improved 
    # ## rho = rho_ozaki(green,energy, t, vm_a1x, Temp,jsd)
    # elseif solver == "denis"
    rho = rho_denis(H, eq_var,global_var)#(energy, t, vm_a1x, Temp,jsd)
    #end
    # params_sden = Dict( "sden" => true, "scurr"=>false
    #                 , "curr"=>false, "rho"=>false,"cden"=>false ,"bcurr" =>false )
    #spin_eq = Observables(rho, params_sden, true)["sden"]
    ### Initiallize the vectors to calculate the espin densiy. Note that this can be improved in the future
    sden_xab::Array{ComplexF64,3} = zeros(ComplexF64, 3,2*global_var.n,2*global_var.n) ;
    sden_xa1::Array{Float64,2} = zeros(Float64, 3, global_var.n); 
    @tullio sden_xab[x,a,b] = rho[a,c]*global_var.σ_abx[c,b,x]              
    @tullio sden_xa1[x,a1] = real(sden_xab[x,2a1-1,2a1-1] + sden_xab[x,2a1,2a1])
    spin_eq = [sden_xa1[:, i] for i in 1:global_var.n :: Int]
    return  spin_eq
end

function cden_eq(H, eq_var,global_var)
    cden::Array{Float64} = zeros(Float64,global_var.n)
    rho_eq = rho_denis(H, eq_var,global_var) 
    for i in range(1, global_var.n)
        cden[i] = real(tr(rho_eq[2*i-1:2*i, 2*i-1:2*i]))
    end
    cden = real(cden)
    return cden
end

function bcurrs_eq(H, eq_var,global_var)
    cc  = zeros(Float64, global_var.n-1)
    cx = zeros(Float64, global_var.n-1)
    cy = zeros(Float64, global_var.n-1)
    cz = zeros(Float64, global_var.n-1)
    rho_eq = rho_denis(H, eq_var,global_var)
    for i in range(1,global_var.n-1)
    cc_m = -2*pi*im*(rho_eq[2*i-1:2*i, 2*i+1:2*i+2]*H[2*i+1:2*i+2, 2*i-1:2*i] 
            - rho_eq[2*i+1:2*i+2, 2*i-1:2*i]*H[2*i-1:2*i, 2*i+1:2*i+2] )
    cc[i] =real(tr(cc_m))
    cx[i] =real(tr(global_var.σ_x*cc_m))  
    cy[i] =real(tr(global_var.σ_y*cc_m)) 
    cz[i] =real(tr(global_var.σ_z*cc_m))  
    end
    bcurrs = real([cc, cx, cy, cz ])
    return bcurrs
end


### Denis density matrix functions

function adjust!(eq_var)#(temp_im,mu_re,mu)
    """ This function modifies temp_im and mu_re in order 
    nto assure not having an overlapping between the poles
    """
    m_max = ceil(0.5*(eq_var.mu/(pi*eq_var.KB*eq_var.temp_im) -1.)  )
    if m_max == 0. 
        eq_var.temp_im = 2*eq_var.temp_im
    else
        eq_var.temp_im = eq_var.mu/(2*pi*eq_var.KB*m_max)
    end
    n_max = ceil(0.5*(eq_var.mu_re/(pi*eq_var.KB*eq_var.temp_im ) -1.)  )
    eq_var.mu_re = 2*pi*eq_var.KB*eq_var.temp_im*n_max

    #return temp_im, mu_re 
    return nothing
end

# function get_temperatures(eq_var)#(mu, e_min, temp, p)
#     #temperatures = zeros(ComplexF64,2)
#     term1 = 0.5*eq_var.temp
#     term2 = eq_var.temp + (eq_var.mu - eq_var.e_min)/(eq_var.p*eq_var.KB)
#     temp = sqrt(term1*term2)
#     #temperatures = [temp,temp] ### real and imaginary part 
#     return Float64[temp,temp]#temperatures 
# end

function denis_no_of_poles!(eq_var)#(mu,mu_im,mu_re,temp,temp_im,temp_re,p=21)
    """ Number of poles with the denis method 
    """
    ### Number of poles of imaginary kind 
    up = eq_var.mu - eq_var.mu_re + eq_var.p*eq_var.KB*eq_var.temp_re + eq_var.p*eq_var.KB*eq_var.temp
    dn = 2*pi*eq_var.KB*eq_var.temp_im
    eq_var.n_im = Int(div(up,dn))#Int(round(up/dn) )
    ### Number of poles of conventional kind
    up = 2*eq_var.mu_im
    dn = 2*pi*eq_var.KB*eq_var.temp
    eq_var.n = Int(div(up,dn))
    ### Number of poles of real kind
    up = 2*eq_var.mu_im
    dn = 2*pi*eq_var.KB*eq_var.temp_re
    eq_var.n_re = Int(div(up,dn))
    ### Number of poles in an array 
    #no_of_poles = Int[n,n_re,n_im] 
    return nothing#Int[n,n_re,n_im] #no_of_poles
end 
    
function denis_poles!(eq_var)#(mu, mu_im,mu_re, temp,temp_im,temp_re,n,n_im,n_re)
    """ This is used to compute the poles and residues of the Modified fermi
    function 
    args: 
    ----
    mu,mu_im,mu_re,temp,temp_im,temp_re,n,n_im,nre : Float64
    output:
    ------
    poles_denis: n-dimensional array of Float64
    array containing the real, imaginary and normal poles of the modified
    fermi function
    res_denis: n-dimensional array of Float64
    array containing the real, imaginary and normal Residues 
    of the modified fermi function
    """
    ## First the m_max is computed in order to avoid an 
    ## overlapping in the poles 
    m_max = ceil( 0.5*(eq_var.mu /(pi*eq_var.KB*eq_var.temp_im) - 1. )   )
    ### Compute poles and residues for the imaginary term 
    for i in 1:eq_var.n_im
        m = m_max-(i-1)
        z =  pi*eq_var.KB*eq_var.temp_im*(2*m+1) + 1im*eq_var.mu_im 
        eq_var.poles_denis[i] =  z
        eq_var.res_denis[i] = -(fermi_mu(eq_var.mu,eq_var.temp*eq_var.KB,z ) 
            -fermi_mu(eq_var.mu_re,eq_var.temp_re*eq_var.KB,z) )*1im*eq_var.KB*eq_var.temp_im
    end
    ### Compute poles and residues for the conventional term 
    for i in 1:eq_var.n
        z = eq_var.mu + 1im*pi*eq_var.KB*eq_var.temp*(2*(i-1) +1 )
        eq_var.poles_denis[eq_var.n_im + i] = z
        eq_var.res_denis[eq_var.n_im + i] = - eq_var.KB*eq_var.temp*fermi_mu(1im*eq_var.mu_im, 1im*eq_var.temp_im*eq_var.KB, z )
    end 
    ### Compute poles and residues for the real term 
    for i in 1: eq_var.n_re
        z = eq_var.mu_re + 1im*pi*eq_var.KB*eq_var.temp_re*(2*(i-1)+ 1)
        eq_var.poles_denis[eq_var.n_im + eq_var.n + i] = z
        eq_var.res_denis[eq_var.n_im+eq_var.n+i] = eq_var.KB*eq_var.temp_re*fermi_mu(1im*eq_var.mu_im,1im*eq_var.temp_im*eq_var.KB,z )
    end
    return nothing #poles_denis, res_denis
end

function init_denis!(eq_var)#(;mu=E_F_system ,temp=Temp,e_min=-3.,p=21)
    """ Set the adjusted (non overlapping poles) and their 
    corresponding residues
    """
    term1 = 0.5*eq_var.temp
    term2 = eq_var.temp + (eq_var.mu-eq_var.e_min)/(eq_var.p*eq_var.KB)
    eq_var.temp = sqrt(term1*term2)
    eq_var.temp_im = eq_var.temp #t_im_re[1]
    eq_var.temp_re = eq_var.temp #t_im_re[2]
    eq_var.mu_im = eq_var.p*eq_var.KB*eq_var.temp_im
    eq_var.mu_re = eq_var.e_min - eq_var.p*eq_var.KB*eq_var.temp_re
    adjust!(eq_var)            ### Adjust temp_im and mu_re
    denis_no_of_poles!(eq_var) ### calculate ntot
    eq_var.ntot = eq_var.n + eq_var.n_re + eq_var.n_im 
    eq_var.poles_denis = zeros(ComplexF64,eq_var.ntot)
    eq_var.res_denis = zeros(ComplexF64, eq_var.ntot)
    denis_poles!(eq_var)        ### Find poles_denis and res_denis
    #return poles_denis, res_denis
    return nothing
end

function rho_denis(H,eq_var,global_var)#,global_var)#(energy,t,vm_a1x,Temp,jsd =1,rdist =1e30)
    """ Computes the density matrix using the denis method
    """
    ### First we need the poles and the residues 
    rho_denis = zeros(ComplexF64, global_var.n*global_var.n_channels,global_var.n*global_var.n_channels )
    for i in 1:eq_var.ntot::Int
        temp_denis = 2im*eq_var.res_denis[i].*green(H,eq_var.poles_denis[i],global_var) #### Note that all only the hopping parameter is needed for the calculation
        #(vm_a1x, poles_denis[i],t,jsd) 
        rho_denis += 0.5im*(temp_denis .- temp_denis')
    end
    return rho_denis
end

### Osaki density matrix

# function rho_ozaki(vm_a1x,energy,t,vs_a1x,Temp,jsd,rdist =1e30,n_poles=200)
#     """ Computes the density matrix using the osaki method
#     """
#     ### first we find the poles
#     rho_ozaki = 0.0
#     sum_ozaki = [0. ,0.]
#     poles,res = get_poles(2*n_poles +1)
#     poles = 1/poles
#     poles = poles[count_poles-1:-1:2]
#     res = res[count_poles-1:-1:2]
#     ### note that the order of the poles is modified 
#     for i in 1: length(poles)
#         pole = energy +1im*poles[i]/beta
#         sum_ozaki +=residues[i]*green_func(vm_a1x,pole,t,jsd) 
#     end
#     sum_ozaki = 2im*sum_ozaki/beta

#     rho_ozaki = (sum_ozaki - sum_ozaki')/(2im)
#     rho_ozaki += 0.5im*rdist_in*green_func(vm_a1x,1im*rdist_in,t,j_sd )
# end

# function get_poles(local_dim) # = n_channels*N_poles)
#     """This function calculates the poles and residues of the Ozaki decomposotion of 
#     the fermi energy(Note that there are not difference between right and left poles)
#     """
#     #:qlocal_dim = n_channels*N_poles + 1
#     #Mat = zeros(local_dim,local_dim)
#     diag = [1/(2*sqrt(4*x^2-1)) for x in 1:local_dim-1 ]
#     ### necesary matrix to compute the Osaki poles and residues 
#     Mat = diagm(-1 => diag) + diagm(1 => diag)
#     ### engein values of the function 
#     Eig_vals, Mat = eigen(Mat)
#     ### residues 
#     Res = Mat[1,:].^2 ./ (4 .*Eig_vals.^2)
#     ### filtering the positive values (only the upper poles in the complex plane are needed)
#     Eig_vals_p = [] # positive eigenvalues
#     Res_p = []
#     for i in 1:local_dim
#         if Eig_vals[i]>0.
#             #println(Eig_vals[i],"  " ,i )
#             push!(Eig_vals_p, Eig_vals[i])
#             push!(Res_p, Res[i])
#         end
#     end
#     return Eig_vals_p, Res_p
# end




end
