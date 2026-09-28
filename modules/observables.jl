module observables 
### Libraries
using LinearAlgebra     ### Linear algebra library
using Tullio            ### Library to work with tensors



function Observables!(vector,params,dv,ov,gv)
    """ Update the observables determined by the dictionary params
    """
    if params["curr"] == true                                          ### Current
        ##### Current
        @tullio ov.curr_α[α] = real( 4*pi*dv.Pi_abα[a,a,α])
        #currs = [ccurr, curr_α]                                       ### Total charge current and Current_left and Current_right
    end
    
    if params["scurr"] == true
        ##### Spin_Current
        @tullio ov.scurr_xα[x,α] =  real( 4*pi*gv.σ_abx[a,b,x]*dv.Pi_abα[b,a,α] )### Note that sigma_full must be computed 
    end
    
    if params["sden"] == true

        ##### Spin density 
        @tullio ov.sden_xab[x,a,b] = dv.rho_ab[a,c]*gv.σ_abx[c,b,x]              
        @tullio ov.sden_a1x[a1,x] = real(ov.sden_xab[x,2a1-1,2a1-1] + ov.sden_xab[x,2a1,2a1] )
        #ov.vsden_xa1 = [ov.sden_xa1[:, i] for i in 1:gv.n :: Int]
        #println("done")
        
    end

    if params["cden"] == true
        for i in range(1, gv.n)
            ov.cden[i] = real(tr(dv.rho_ab[2*i-1:2*i, 2*i-1:2*i] ) )
        end
        ov.cden = real(ov.cden)
        #println(ov.cden)
    end
    
    ###################################################### The bond currents should be implemented later
    # if params["bcurrs"] == true
    #     #println("join bcurrs")
    #     cc::Array{Float64}  = zeros(Float64, global_var.n-1)
    #     cx::Array{Float64} = zeros(Float64, global_var.n-1)
    #     cy::Array{Float64} = zeros(Float64, global_var.n-1)
    #     cz::Array{Float64} = zeros(Float64, global_var.n-1)
    #     for i in range(1,global_var.n-1)
    #         cc_m = -2*pi*im*(dynamics_var.rho_ab[2*i-1:2*i, 2*i+1:2*i+2]*dynamics_var.H_ab[2*i+1:2*i+2, 2*i-1:2*i] 
    #             - dynamics_var.rho_ab[2*i+1:2*i+2, 2*i-1:2*i]*dynamics_var.H_ab[2*i-1:2*i, 2*i+1:2*i+2] )
    #         cc[i] = real(tr(cc_m)  )
    #         cx[i] = real(tr(global_var.σ_x*cc_m) )  
    #         cy[i] = real(tr(global_var.σ_y*cc_m) )
    #         cz[i] = real(tr(global_var.σ_z*cc_m) )
    #     end
    #     #println("calculation of individual components")
    #     observables_var.bcurrs = real([cc, cx, cy, cz ])
    #     #println("bcurrs calculated")
    # end
    nothing
end

end
