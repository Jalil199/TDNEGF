module TDNEGF

    using LinearAlgebra
    using DifferentialEquations
    using DelimitedFiles
    using StaticArrays
    ⊗(A,B) = kron(A,B) ;
    include("types.jl")
    # if only this element is included thus
    # the function are called as 
    # TDNEGF.SelfEnergySquare.function
    include("selfenergy.jl")  
    include("hamiltonians.jl")
    include("eom_tdnegf.jl")
    include("observables.jl")
    include("poles.jl")
    
    ### main function to export from types.jl
    export ModelParamsTDNEGF, DynamicalVariables, pointer
    ### Submodule of self energy for square lattice
    # Bring functions to the name space of module 
    using .SelfEnergySquare: build_Σᴸ_nλ, build_Σᴳ_nλ,
                             build_χ_nλ, build_ξ_an
    # Export functios of self energy without join the submodule
    export build_Σᴸ_nλ, build_Σᴳ_nλ, build_χ_nλ,build_ξ_an
    # functions from hamiltonians.jl"
    export build_H_ab, update_H_e!
    # functions from eom_tdnegf.jl"
    export eom_tdnegf!
    # functions from observables.jl"
    export ObservablesTDNEGF, obs_s_i!, obs_σ_i!, obs_n_i!,obs_Ixα!, get_sub
    # functions from poles.jl"
    export load_poles_square
     
    
end # end module TDNEGF

