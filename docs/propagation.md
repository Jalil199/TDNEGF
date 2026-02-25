# Propagation workflow (current one-time solver)

## Current propagation function
The core RHS routine is:

- `eom_tdnegf!(du, u, p::ModelParamsTDNEGF, t)` in `src/eom_tdnegf.jl`.

This function is passed directly to `ODEProblem` and integrated by DifferentialEquations.jl.

## High-level time-stepping logic
Time stepping itself (step size control, stage evaluations, tolerances) is delegated to the chosen DifferentialEquations.jl integrator (example uses `Vern7()`).

At each RHS evaluation, `eom_tdnegf!` performs in-place updates of `du` from `u` via these conceptual blocks:

1. **Unpack flattened state** with `pointer` into `ρ`, `Ψ`, and split `Ω` tensors.
2. **Density-matrix block**:
   - Build summed auxiliaries `Ψ_anα = Σ_λ Ψ_anλα`.
   - Build `Π_abα = Ψ_anα ξ^T`, then `Π_ab = Σ_α Π_abα`.
   - Compute `dρ_ab = -i(Hρ - Hρ†) + (Π + Π†)`.
3. **Auxiliary-vector block (local terms)**:
   - Compute matrix products `HΨ` and `ρξ`.
   - Apply pole/lead coefficients (`χ`, `Σᴸ`, `Γ`, `Δ`) to fill `dΨ`.
4. **Auxiliary-vector block (Ω coupling terms)**:
   - Add `-i Ω·ξ`-type contributions using split Ω tensors.
5. **Ω block**:
   - Update each Ω partition using dot products between channel vectors and Ψ components plus coefficient combinations.

All operations are in-place and reuse preallocated buffers stored in `p`.

## Where a future `PropagationMethod` hook would plug in
There is currently a **single hardcoded propagation RHS** (`eom_tdnegf!`) and no `PropagationMethod` abstraction/type dispatch layer in `src/`.

A future hook for alternatives (e.g., GKBA) would naturally sit at the RHS selection boundary, i.e. where `ODEProblem` currently receives `eom_tdnegf!`. In practice, this could be introduced as:
- a method-dispatch wrapper that chooses among RHS implementations, or
- multiple RHS constructors sharing the same parameter/state layout when possible.

## Confirmation: this is not a two-time solver
The present implementation evolves a one-time ODE state `u(t)` with auxiliary variables (`ρ`, `Ψ`, `Ω`) and does **not** maintain a two-time Green’s function grid `G(t,t')` as solver state.
