using Test
using TDNEGF

@testset "build_ξ_an and build_ξ_local_index are equivalent" begin
    Nx, Ny = 2, 3
    Nσ, N_orb = 2, 1
    Nloc = Nσ * N_orb
    Ns = Nx * Ny * Nloc

    xcol = 1
    y_coup = 1:Ny

    ξ_old = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = xcol, y_coup = y_coup)

    local_index = Int[]
    for y in y_coup
        i = (xcol - 1) * Ny + y
        append!(local_index, ((i - 1) * Nloc + 1):(i * Nloc))
    end

    Nc = Ny * Nloc
    U_local = zeros(ComplexF64, length(local_index), Nc)

    for (k, row) in enumerate(local_index)
        α = (row - 1) % Nloc + 1
        y = ((row - 1) ÷ Nloc) % Ny + 1
        for ny_mode in 1:Ny
            amp = sqrt(2 / (Ny + 1)) * sin(ny_mode * y * pi / (Ny + 1))
            n = (ny_mode - 1) * Nloc + α
            U_local[k, n] = amp
        end
    end

    ξ_new = build_ξ_local_index(Ns, local_index, U_local)

    @test size(ξ_old) == size(ξ_new)
    @test ξ_old == ξ_new
end
