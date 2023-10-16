module derived_constants
export Eig_vals, Res_p,Eig_vals_k2α,hi_αmk, R_k2α,
       hi_αmk1,hi_αmk2,Gam_greater_αmik, Gam_lesser_αmik,
       csi_aikα

#include("parameters.jl")
#using .parameters

include("get_parameters.jl")
using .get_parameters

include("osaki_poles.jl")
import .osaki_poles: get_poles
include("dynamical_variables.jl")
import .dynamical_variables: create_hi, create_Gam, create_csi

### -----  external variables involved in the dynamic ---- ###
const Eig_vals, Res_p = get_poles()
const Eig_vals_k2α = cat(Eig_vals,Eig_vals,dims=2)
const R_k2α = cat(Res_p,Res_p,dims=2) ;
const hi_αmk,hi_αmk1,hi_αmk2 = create_hi( Eig_vals_k2α = Eig_vals_k2α)
const Gam_greater_αmik, Gam_lesser_αmik = create_Gam(hi_αmk=hi_αmk
                                    ,hi_αmk1=hi_αmk1
                                    ,hi_αmk2=hi_αmk2,R_k2α=R_k2α)
const csi_aikα = create_csi() ;
#println("the constant values were created")

end