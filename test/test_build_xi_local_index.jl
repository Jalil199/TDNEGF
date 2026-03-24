using Test
using TDNEGF

@testset "build_ξ_local_index maps local rows into global coupling" begin
    Ns = 6
    local_index = [2, 5]
    U_local = ComplexF64[1 + 1im 2 - 1im; 3 + 0im 4 + 2im]

    ξ_an = build_ξ_local_index(Ns, local_index, U_local)

    @test size(ξ_an) == (Ns, size(U_local, 2))
    @test ξ_an[local_index, :] == U_local

    zero_rows = setdiff(1:Ns, local_index)
    @test ξ_an[zero_rows, :] == zeros(ComplexF64, length(zero_rows), size(U_local, 2))
end
