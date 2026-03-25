# TDNEGF-Hybrid

TDNEGF-Hybrid is a Julia package for time-dependent quantum transport in open systems.
In practice, you evolve a one-time TDNEGF ODE system for the reduced density matrix
and auxiliary variables, then post-process observables from that trajectory.

## Features

- Two solver backends in one package:
  - block-based heterogeneous auxiliary backend (`eom_tdnegf_blocks!`) for current/new workflows,
  - legacy rectangular auxiliary backend (`eom_tdnegf!`) for compatibility and cross-checks.
- Pole-based embedding self-energy workflow for square-lattice lead setups.
- In-place ODE propagation with DifferentialEquations.jl.
- Observable utilities for charge density, electronic spin density, and lead/block charge-spin currents.
- Ready-to-run examples and notebooks demonstrating the current block-based workflow.

## Current architecture

At the moment, the repository ships with **two auxiliary-solver backends**:

- **Block-based heterogeneous auxiliary backend (recommended for new work)**
  - RHS: `eom_tdnegf_blocks!`
  - Parameters: `ExperimentalBlockRHSParams`
  - Self-energy structure: `SelfEnergyBlock`
  - This is the path used by the main examples and notebooks today.

- **Legacy rectangular auxiliary backend (stable compatibility path)**
  - RHS: `eom_tdnegf!`
  - Parameters: `ModelParamsTDNEGF`
  - Kept for continuity with older workflows and for cross-checking.

> TL;DR: if you are starting now, use the block backend first.
> The type name still includes `Experimental...`, but this is the workflow actively demonstrated in the repo.

## Core roles

- **`SelfEnergyBlock`**
  - Stores **static structural block data** for one auxiliary block: block sizes/splits (`Nc`, `N_λ1`, `N_λ2`) and static tensors (`ΣL_nλ`, `ΣG_nλ`, `χ_nλ`, `ξ_an`).

- **`ExperimentalBlockRHSParams`**
  - Stores solver-level state, preallocated workspaces, and **dynamic problem quantities** used during propagation.
  - This includes dynamic data like the active Hamiltonian `H_ab` and per-block energy shifts `Δ_blocks`.

- **`eom_tdnegf_blocks!`**
  - In-place ODE RHS for the block backend.
  - Uses block/pair layout metadata to evolve `ρ_ab` plus heterogeneous `Ψ`/`Ω` sectors without converting to the old rectangular auxiliary layout.

## Design split 

- Put **structural/static** block metadata in `SelfEnergyBlock`.
- Put **dynamic, problem-level** quantities (`H_ab`, `Δ_blocks`, and runtime caches) in `ExperimentalBlockRHSParams`.

This split makes sweeps and time-dependent updates simpler: you can modify
dynamic terms without rebuilding block definitions.

## Physical quantities and formalism 

The propagated state
follows the same TDNEGF one-time structure in both backends:

- `ρ_ab(t)`: reduced single-particle density matrix in the device basis.
- `Ψ` and `Ω`: auxiliary objects from the pole expansion of embedding
  self-energies, which close the equations as a first-order ODE system.

At a high level, the RHS combines:
- evolution through `H_ab`,
- open-boundary/lead effects through self-energy coefficients
  (`ΣL`, `ΣG`, `χ`, `ξ`),
- and bias-dependent shifts through `Δ` (legacy `Δ_α`, block `Δ_blocks`).

In short: **same physical TDNEGF formalism, different auxiliary data layout**
(rectangular legacy vs heterogeneous blocks).

## Observables and post-processing status

- **Backend-independent observables** (effectively shared):
  - Local charge density and local electronic spin density come from `ρ_ab`,
    so they are backend-independent in practice.

- **Current/spin-current observables in the block backend**:
  - `obs.Iα` / `obs.Iαx` are indexed by **auxiliary block order** (`p_blocks.blocks`) unless you explicitly aggregate/relabel blocks to match physical-lead grouping.

- **Compatibility note**:
  - Legacy and block current-observable paths both exist.
  - Prefer the block-native current path when using the block solver, but compatibility overloads are still available during migration.

## Installation

### Julia version
- Recommended: Julia `1.10+`.

### Add package
From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/<your-org-or-user>/TDNEGF.git")
```

Or for local development:

```julia
using Pkg
Pkg.develop(path="/path/to/TDNEGF")
```

## Quick start 

If you are new to the repo, start with the **block backend** workflow:

```julia
using TDNEGF
using DifferentialEquations

# 1) Build your model Hamiltonian + lead ingredients (see example for full setup)
p_model = ModelParamsTDNEGF(Nx=6, Ny=2, Nσ=2, N_orb=1, Nα=2, N_λ1=49, N_λ2=20)
H_ab = build_H_ab(; Nx=p_model.Nx, Ny=p_model.Ny, Nσ=p_model.Nσ, N_orb=p_model.N_orb,
                  γ=1.0, γso=0.5 + 0im)
p_model.H_ab .= H_ab
p_model.H0_ab .= H_ab

# 2) Build static blocks + dynamic shifts
#    (Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL/ξ_anR built as in examples/01_...)
left_block  = SelfEnergyBlock(:left,  p_model.Nc, p_model.N_λ1, p_model.N_λ2, Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL)
right_block = SelfEnergyBlock(:right, p_model.Nc, p_model.N_λ1, p_model.N_λ2, Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anR)
blocks = [left_block, right_block]
Δ_blocks = ComplexF64[+0.5 + 0im, -0.5 + 0im]

# 3) Build RHS params and initial state
p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, Δ_blocks, p_model)
u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)

# 4) Solve with the block RHS
prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, 10.0), p_blocks)
sol  = solve(prob, Vern7(); reltol=1e-6, abstol=1e-8, dense=false)
```

For legacy reproduction/validation, the rectangular path is still available via
`ModelParamsTDNEGF` + `eom_tdnegf!`.

## Usage: which backend should I pick?

- Choose **block backend** (`eom_tdnegf_blocks!`) for new simulations and current examples.
- Choose **legacy backend** (`eom_tdnegf!`) when reproducing older scripts/results or validating against previous rectangular-auxiliary behavior.

## Current examples and notebooks

If you want to copy a working workflow quickly, start here:

- Script: `examples/01_two_terminal_square_lattice.jl`
- Notebooks:
  - `examples/notebooks/01_two_terminal_square_lattice.ipynb`
  - `examples/notebooks/02_spin_dynamics_and_electrons.ipynb`

These examples show the end-to-end block workflow:
block construction, `ExperimentalBlockRHSParams`, solving with `eom_tdnegf_blocks!`,
and post-processing with `pointer_blocks(...)`.

## Project structure

- `src/` — package implementation (`types.jl`, `eom_tdnegf.jl`, `observables.jl`, etc.).
- `examples/` — runnable scripts and notebooks.
- `docs/` — lightweight developer docs:
  - [`docs/architecture.md`](docs/architecture.md)
  - [`docs/propagation.md`](docs/propagation.md)
  - [`docs/observables.md`](docs/observables.md)
- `legacy/` — legacy notebooks/scripts kept for reference.

## Citation
- Placeholder: add preferred citation information here if/when a `CITATION` file is added.

## License
- Placeholder: add license information here if/when a `LICENSE` file is added.
