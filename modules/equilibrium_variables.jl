module equilibrium_variables
### Libraries
# include("parameters.jl")
# using .parameters
include("get_parameters.jl")
using .get_parameters
include("green_functions.jl")
import .green_functions: green
include("observables.jl")
import .observables: Observables

function fermi_mu(mu, Temp, energy)
    """ This function computes the fermi energy used in the computation
    of equilibrium quantities
    """
   1/(1. + exp((energy-mu)/(KB*Temp)  ))   
end

function spindensity_eq(vm_a1x, energy; t,Temp,jsd=1.0, solver = solver)
    """ This function computes the spin density in equilirbrium 
    """
    if solver == "ozaki"
    rho = rho_ozaki(green,energy, t, vm_a1x, Temp,jsd)
    elseif solver == "denis"
    rho = rho_denis(green,energy, t, vm_a1x, Temp,jsd)
    end
    params_sden = Dict( "sden" => true, "scurr"=>false
                    , "curr"=>false, "rho"=>false,"cden"=>false ,"bcurr" =>false )
    spin_eq = Observables(rho, params_sden, true)["sden"]
    return  spin_eq
end

### Denis density matrix functions

function adjust!(temp_im,mu_re,mu)
    """ This function modifies temp_im and mu_re in order 
    nto assure not having an overlapping between the poles
    """
    m_max = ceil(0.5*(mu/(pi*KB*temp_im) -1.)  )
    if m_max == 0. 
        temp_im = 2*temp_im
    else
        temp_im = mu/(2*pi*KB*m_max)
    end
    n_max = ceil(0.5*(mu_re/(pi*KB*temp_im ) -1.)  )
    mu_re_p = 2*pi*KB*temp_im*n_max

    return temp_im, mu_re 
end

function get_temperatures(mu, e_min, temp, p)
    #temperatures = zeros(ComplexF64,2)
    term1 = 0.5*temp
    term2 = temp + (mu-e_min)/(p*KB)
    temp = sqrt(term1*term2)
    #temperatures = [temp,temp] ### real and imaginary part 
    return Float64[temp,temp]#temperatures 
end

function denis_no_of_poles(mu,mu_im,mu_re,temp,temp_im,temp_re,p=21)
    """ Number of poles with the denis method 
    """
    ### Number of poles of imaginary kind 
    up = mu - mu_re + p*KB*temp_re + p*KB*temp
    dn = 2*pi*KB*temp_im
    n_im = Int(div(up,dn))#Int(round(up/dn) )
    ### Number of poles of conventional kind
    up = 2*mu_im
    dn = 2*pi*KB*temp
    n = Int(div(up,dn))
    ### Number of poles of real kind
    up = 2*mu_im
    dn = 2*pi*KB*temp_re
    n_re = Int(div(up,dn))
    ### Number of poles in an array 
    #no_of_poles = Int[n,n_re,n_im] 
    return Int[n,n_re,n_im] #no_of_poles
end 
    
function denis_poles(mu, mu_im,mu_re, temp,temp_im,temp_re,n,n_im,n_re)
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
    m_max=ceil( 0.5*(mu /( pi*KB*temp_im) - 1. )   )
    ### Compute poles and residues for the imaginary term 
    for i in 1:n_im
        m = m_max-(i-1)
        z =  pi*KB*temp_im * (2*m+1) + 1im*mu_im 
        poles_denis[i] =  z
        res_denis[i] = -(fermi_mu(mu,temp,z ) 
            -fermi_mu(mu_re,temp_re,z) )*1im*KB*temp_im
    end
    ### Compute poles and residues for the conventional term 
    for i in 1:n
        z = mu + 1im*pi*KB*temp*(2*(i-1) +1 )
        poles_denis[n_im + i] = z
        res_denis[n_im + i] = -KB*temp*fermi_mu(1im*mu_im, 1im*temp_im, z )
    end 
    ### Compute poles and residues for the real term 
    for i in 1: n_re
        z = mu_re + 1im*pi*KB*temp_re*(2*(i-1)+ 1)
        poles_denis[n_im + n + i] = z
        res_denis[n_im+n+i] = KB*temp_re*fermi_mu(1im*mu_im,1im*temp_im,z )
    end
    return poles_denis, res_denis
end

function init_denis(;mu=E_F_system ,temp=Temp,e_min=-3.,p=21)
    """ Set the adjusted (non overlapping poles) and their 
    corresponding residues
    """
    #p_in = p
    global delta = (mu-e_min)/(KB*temp)
    global t_im_re = get_temperatures(mu,e_min,temp,p)
    global temp_im = t_im_re[1]
    global temp_re = t_im_re[2]
    global mu_im = p*KB*temp_im
    global mu_re = e_min - p*KB*temp_re

    temp_im, mu_re = adjust!(temp_im,mu_re,mu)

    n_con_re_im = denis_no_of_poles(mu, mu_im,mu_re
        ,temp,temp_im,temp_re,p)
    n,n_re,n_im = n_con_re_im
    global ntot = n + n_re + n_im 
    global poles_denis = zeros(ComplexF64,ntot)
    global res_denis = zeros(ComplexF64, ntot)
    poles_denis, res_denis = denis_poles(mu,mu_im,mu_re,
       temp,temp_im, temp_re,n,n_im,n_re)
    
    return poles_denis, res_denis
end

function rho_denis(green,energy,t,vm_a1x,Temp,jsd =1,rdist =1e30)
    """ Computes the density matrix using the denis method
    """
    ### First we need the poles and the residues 
    rho_denis = zeros(ComplexF64, n*n_channels,n*n_channels )
    for i in 1:ntot::Int
        temp_denis = 2im*res_denis[i].*green(vm_a1x, poles_denis[i],t,jsd) 
        rho_denis += 0.5im*(temp_denis .- temp_denis')
    end
    return rho_denis
end


### Osaki density matrix

function rho_ozaki(vm_a1x,energy,t,vs_a1x,Temp,jsd,rdist =1e30,n_poles=200)
    """ Computes the density matrix using the osaki method
    """
    ### first we find the poles
    rho_ozaki = 0.0
    sum_ozaki = [0. ,0.]
    poles,res = get_poles(2*n_poles +1)
    poles = 1/poles
    poles = poles[count_poles-1:-1:2]
    res = res[count_poles-1:-1:2]
    ### note that the order of the poles is modified 
    for i in 1: length(poles)
        pole = energy +1im*poles[i]/beta
        sum_ozaki +=residues[i]*green_func(vm_a1x,pole,t,jsd) 
    end
    sum_ozaki = 2im*sum_ozaki/beta

    rho_ozaki = (sum_ozaki - sum_ozaki')/(2im)
    rho_ozaki += 0.5im*rdist_in*green_func(vm_a1x,1im*rdist_in,t,j_sd )
end

end
