# TDNEGF architecture (current implementation)

## Scope
This document describes the **current** solver in this repository as implemented in `src/`, focused on the present **one-time TDNEGF ODE formulation**.

## Overall workflow (one-time formulation)
At a high level, the current flow is:

1. Build and allocate a `ModelParamsTDNEGF` object containing model sizes, static inputs (Hamiltonian and embedding coefficients), workspace arrays, and the flattened state vector `u`.
2. Fill static quantities (`H_ab`, lead-channel couplings `ξ_anα`, pole-expanded self-energy data `Σᴸ_nλα`, `Σᴳ_nλα`, `χ_nλα`, and derived arrays such as `Γ_nλα` and conjugates).
3. Define an `ODEProblem(eom_tdnegf!, p.u, tspan, p)`.
4. Integrate with DifferentialEquations.jl (example: `solve(prob, Vern7(), ...)`).
5. Post-process each saved state with `pointer(ut, p)` and observable routines.

This is a **single-time** state propagation (`u(t)`), not a two-time contour-grid storage/propagation strategy.

## Role of `global_params`
In the current `src/` API there is **no active `global_params` struct/module**; the primary parameter container is `ModelParamsTDNEGF` in `src/types.jl`.

A `global_parameters.jl` file exists under `legacy/`, but that is outside the current package path used by `src/TDNEGF.jl`.

## Core data structures used in propagation

### `ModelParamsTDNEGF`
`ModelParamsTDNEGF` centralizes:
- Lattice/channel dimensions (`Nx`, `Ny`, `Ns`, `Nc`, `Nα`, `N_λ1`, `N_λ2`, ...).
- Tensor shapes and flattened index ranges for packing/unpacking the ODE state.
- Static matrices/tensors: `H_ab`, `ξ_anα`, `Σᴸ_nλα`, `Σᴳ_nλα`, `χ_nλα`, `Γ_nλα`, conjugates, and lead shifts `Δ_α`.
- Dynamical allocations and workspace buffers used by the EOM for in-place updates.

### Flattened state vector + views
The ODE solver evolves one flattened complex vector `u::Vector{ComplexF64}`. The helper `pointer(u, p)` returns typed views as:
- `ρ_ab` (density matrix),
- `Ψ_anλα` (auxiliary vectors),
- split `Ω` blocks (`Ω_nλ1α_nλ1α`, `Ω_nλ1α_nλ2α`, `Ω_nλ2α_nλ1α`).

This avoids reallocations and keeps solver interoperability with DifferentialEquations.jl.

### `DynamicalVariables`
`DynamicalVariables` is a lightweight view container returned by `pointer`, used both in propagation (`eom_tdnegf!`) and in observable calculations.

## Interaction with Distributed workers
No explicit `Distributed`/multi-process worker orchestration appears in current `src/` or the main example workflow. Parallelism in practice is currently thread/BLAS-level where available, but there is no dedicated worker partitioning logic in the solver API.
