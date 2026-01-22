module observables 
### Libraries
using LinearAlgebra     ### Linear algebra library
using Tullio            ### Library to work with tensors



function Observables!(vector,params, dynamics_var,observables_var,global_var)
    """ Update the observables determined by the dictionary params
    """
    if params["curr"] == true                                          ### Current
        ##### Current
        #println("join current")
        @tullio observables_var.curr_α[α] = real( 4*pi*dynamics_var.Pi_abα[a,a,α])
        #println("current calculated")
        #ccurr = 0.5* (curr_α[1] - curr_α[2])
        #currs = [ccurr, curr_α]                                       ### Total charge current and Current_left and Current_right
    end
    if params["scurr"] == true
        ##### Spin_Current
        @tullio observables_var.scurr_xα[x,α] =  real( 4*pi*global_var.σ_abx[a,b,x]*dynamics_var.Pi_abα[b,a,α] )### Note that sigma_full must be computed 
    end
    if params["sden"] == true
        ##### Spin density 
        #println("join sden")
        @tullio observables_var.sden_xab[x,a,b] = dynamics_var.rho_ab[a,c]*global_var.σ_abx[c,b,x]              
        @tullio observables_var.sden_xa1[x,a1] = real(observables_var.sden_xab[x,2a1-1,2a1-1] + observables_var.sden_xab[x,2a1,2a1] )
        observables_var.vsden_xa1 = [observables_var.sden_xa1[:, i] for i in 1:global_var.n :: Int]
        #println("sden calcualted")
        #return_params["sden"] = vsden_xa1
    end

    if params["cden"] == true
        #println("join cden ")
        cden::Array{Float64} = zeros(Float64,global_var.n)  #zeros(Float64, n )
        for i in range(1, global_var.n)
            cden[i] = real(tr(dynamics_var.rho_ab[2*i-1:2*i, 2*i-1:2*i] ) )
        end
        observables_var.cden = real(cden)
        #println("cden calculated ")
    end

    if params["bcurrs"] == true
        #println("join bcurrs")
        cc::Array{Float64}  = zeros(Float64, global_var.n-1)
        cx::Array{Float64} = zeros(Float64, global_var.n-1)
        cy::Array{Float64} = zeros(Float64, global_var.n-1)
        cz::Array{Float64} = zeros(Float64, global_var.n-1)
        for i in range(1,global_var.n-1)
            cc_m = -2*pi*im*(dynamics_var.rho_ab[2*i-1:2*i, 2*i+1:2*i+2]*dynamics_var.H_ab[2*i+1:2*i+2, 2*i-1:2*i] 
                - dynamics_var.rho_ab[2*i+1:2*i+2, 2*i-1:2*i]*dynamics_var.H_ab[2*i-1:2*i, 2*i+1:2*i+2] )
            cc[i] = real(tr(cc_m)  )
            cx[i] = real(tr(global_var.σ_x*cc_m) )  
            cy[i] = real(tr(global_var.σ_y*cc_m) )
            cz[i] = real(tr(global_var.σ_z*cc_m) )
        end
        #println("calculation of individual components")
        observables_var.bcurrs = real([cc, cx, cy, cz ])
        #println("bcurrs calculated")
    end

    

    nothing
end

end
