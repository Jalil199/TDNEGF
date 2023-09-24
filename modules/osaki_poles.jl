module osaki_poles
export get_poles

### Libraries
using LinearAlgebra
include("parameters.jl")
using .parameters

function get_poles(local_dim = n_channels*N_poles)
    """This function calculates the poles and residues of the Ozaki decomposotion of 
    the fermi energy(Note that there are not difference between right and left poles)
    """
    #:qlocal_dim = n_channels*N_poles + 1
    #Mat = zeros(local_dim,local_dim)
    diag = [1/(2*sqrt(4*x^2-1)) for x in 1:local_dim-1 ]
    ### necesary matrix to compute the Osaki poles and residues 
    Mat = diagm(-1 => diag) + diagm(1 => diag)
    ### engein values of the function 
    Eig_vals, Mat = eigen(Mat)
    ### residues 
    Res = Mat[1,:].^2 ./ (4 .*Eig_vals.^2)
    ### filtering the positive values (only the upper poles in the complex plane are needed)
    Eig_vals_p = [] # positive eigenvalues
    Res_p = []
    for i in 1:local_dim
        if Eig_vals[i]>0.
            #println(Eig_vals[i],"  " ,i )
            push!(Eig_vals_p, Eig_vals[i])
            push!(Res_p, Res[i])
        end
    end
    return Eig_vals_p, Res_p
end

end

