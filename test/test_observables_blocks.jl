using Test
using TDNEGF
using Random

@testset "Observable equivalence between rectangular and block pointers" begin
    p_rect, p_blocks, u_rect, u_blocks = build_equivalent_rhs_setup_multiblock()
    p_blocks_obs = ExperimentalBlockRHSParams(copy(p_rect.H_ab), p_blocks.blocks, copy(p_blocks.Δ_blocks), p_rect)

    dv_rect = TDNEGF.pointer(u_rect, p_rect)
    dv_blocks = pointer_blocks(u_blocks, p_blocks_obs.dims_ρ_ab, p_blocks_obs.aux_layout)

    obs_rect = ObservablesTDNEGF(p_rect; N_tmax = 1, N_leads = p_rect.Nα)
    obs_blocks = ObservablesTDNEGF(p_rect; N_tmax = 1, N_leads = length(p_blocks.blocks))
    obs_rect.idx = 1
    obs_blocks.idx = 1

    obs_n_i!(dv_rect, p_rect, obs_rect)
    obs_n_i!(dv_blocks, p_rect, obs_blocks)

    obs_σ_i!(dv_rect, p_rect, obs_rect)
    obs_σ_i!(dv_blocks, p_rect, obs_blocks)

    obs_Ixα!(dv_rect, p_rect, obs_rect)
    obs_Ixα!(dv_blocks, p_blocks_obs, obs_blocks)

    @test obs_rect.n_i[:, 1] ≈ obs_blocks.n_i[:, 1] rtol = 1e-12 atol = 1e-12
    @test obs_rect.σx_i[:, :, 1] ≈ obs_blocks.σx_i[:, :, 1] rtol = 1e-12 atol = 1e-12
    @test obs_rect.Iα[:, 1] ≈ obs_blocks.Iα[:, 1] rtol = 1e-12 atol = 1e-12
    @test obs_rect.Iαx[:, :, 1] ≈ obs_blocks.Iαx[:, :, 1] rtol = 1e-12 atol = 1e-12

    # Backward-compatible adapter should preserve values exactly.
    obs_blocks_compat = ObservablesTDNEGF(p_rect; N_tmax = 1, N_leads = length(p_blocks.blocks))
    obs_blocks_compat.idx = 1
    obs_Ixα!(dv_blocks, p_blocks, p_rect, obs_blocks_compat)
    @test obs_blocks_compat.Iα[:, 1] ≈ obs_blocks.Iα[:, 1] rtol = 1e-12 atol = 1e-12
    @test obs_blocks_compat.Iαx[:, :, 1] ≈ obs_blocks.Iαx[:, :, 1] rtol = 1e-12 atol = 1e-12

    @test_throws ArgumentError obs_Ixα!(dv_blocks, p_blocks, obs_blocks)
end
