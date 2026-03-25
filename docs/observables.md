# Observables pipeline

## Inputs to observables
Observable routines consume:
- a state pointer from either backend:
  - `dv::DynamicalVariables` via `pointer(ut, p)` (legacy rectangular),
  - `ptr::HeterogeneousAuxPointers` via `pointer_blocks(ut, dims_ρ_ab, aux_layout)` (block-based),
- `p::ModelParamsTDNEGF` (dimensions, Pauli matrices, channel vectors, buffers),
- `obs::ObservablesTDNEGF` (preallocated output container plus active time index `obs.idx`).

Primary electronic input is the equal-time reduced density matrix `ρ_ab`. Lead-current observables additionally use `Ψ` and `ξ` through `Π_abα` reconstruction.

## Observable container and output shapes
`ObservablesTDNEGF` allocates:
- `t::Vector{T}` with shape `(N_tmax,)`.
- `n_i::Matrix{T}` with shape `(N_sites, N_tmax)`.
- `σx_i::Array{T,3}` with shape `(N_sites, 3, N_tmax)` for electronic spin density components `(x,y,z)`.
- `sx_i::Array{T,3}` with shape `(N_sites, 3, N_tmax)` for classical spin components `(x,y,z)`.
- `Iα::Matrix{T}` with shape `(N_leads, N_tmax)` for lead charge current.
- `Iαx::Array{T,3}` with shape `(N_leads, 3, N_tmax)` for lead spin-current components `(x,y,z)`.

## How each observable is computed

### Local charge density `obs_n_i!` (backend-independent)
For each site block `r`, take local density submatrix `ρ[r,r]` and sum diagonal real parts:

- `n_i(site,t) = Σ_{a in site dof} Re[ρ_aa]`.

### Local electronic spin density `obs_σ_i!` (backend-independent)
For each site-local block, compute real traces with Pauli matrices:

- `σ_{x,y,z}(site,t) = Re Tr[σ_{x,y,z} ρ_loc]`.

### Classical spin density `obs_s_i!`
Copies classical spin vectors `S[a,b]::SVector{3}` into `obs.sx_i[:, :, t]`.

### Lead/block charge-spin currents `obs_Ixα!`
1. Reconstruct `Π_abα` via `cal_Π_abα` from `Ψ` and `ξ`.
2. Charge current per channel index:
   - `Iα = 2 Re Tr[Π_abα]`.
3. Spin current components per channel index:
   - `Iαx,y,z = 2 Re Tr[σ_{x,y,z} Π_loc]` accumulated over site blocks.

For the legacy path, index `α` is the rectangular lead index.
For the block path, index `α` follows `p_blocks.blocks` order and is therefore
block-structured (it only equals physical-lead indexing when blocks are defined
one-per-lead). The minimum geometry/spin metadata is stored directly in
`ExperimentalBlockRHSParams` when it is constructed with
`ExperimentalBlockRHSParams(H_ab, blocks, p_model)`.

## Data-flow placement
Typical usage after propagation:
1. iterate over `(it, ut)` in solution snapshots,
2. set `obs.idx = it`,
3. build either `dv = pointer(ut, p)` or `ptr = pointer_blocks(ut, dims_ρ_ab, aux_layout)`,
4. call:
   - local observables: `obs_n_i!`, `obs_σ_i!` with either pointer type,
   - current observables:
     - legacy: `obs_Ixα!(dv, p, obs)`,
     - block: `obs_Ixα!(ptr, p_blocks, obs)` if `p_blocks` was built with
       model metadata, or fallback `obs_Ixα!(ptr, p_blocks, p, obs)`.

This post-processing flow is demonstrated in the main square-lattice example.
