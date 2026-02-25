# TDNEGF-Hybrid

TDNEGF-Hybrid is a Julia package for time-dependent quantum transport in open systems using a **one-time TDNEGF propagation** formulated as a large ODE system. It evolves the reduced density matrix and auxiliary variables (`ρ`, `Ψ`, `Ω`) and supports post-processing into charge/spin observables.

## Features
- One-time TDNEGF ODE propagation (`eom_tdnegf!`) using DifferentialEquations.jl.
- Pole-based embedding self-energy workflow for square-lattice lead setups.
- In-place state packing/unpacking via `ModelParamsTDNEGF` + `pointer` views.
- Observable utilities for local charge density, local spin density, and lead charge/spin currents.
- Ready-to-run example for a two-terminal square-lattice device in `examples/`.

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

## Quickstart

A minimal run (small system) follows the same pattern as `examples/01_two_terminal_square_lattice.jl`:

```julia
using TDNEGF
using DifferentialEquations

# 1) Define dimensions / key pole parameters
p = ModelParamsTDNEGF(Nx=6, Ny=2, Nσ=2, N_orb=1, Nα=2, N_λ1=49, N_λ2=20)

# 2) Build and assign static model data (see example for full setup)
Rλ, zλ = load_poles_square(p.N_λ1, p.N_λ2)
p.H_ab .= build_H_ab(Nx=p.Nx, Ny=p.Ny, Nσ=p.Nσ, N_orb=p.N_orb, γ=1.0, γso=0.5 + 0im)
p.H0_ab .= p.H_ab

# 3) Build lead/self-energy quantities and assign into p (as in examples/01_...)
#    p.Σᴸ_nλα, p.Σᴳ_nλα, p.χ_nλα, p.Γ_nλα, p.ξ_anα, p.Δ_α, and conjugate buffers

# 4) Integrate
prob = ODEProblem(eom_tdnegf!, p.u, (0.0, 10.0), p)
sol  = solve(prob, Vern7(); reltol=1e-6, abstol=1e-8, dense=false)
```

Key parameters you will usually set first:
- Geometry/spin/orbital sizes: `Nx`, `Ny`, `Nσ`, `N_orb`.
- Lead count: `Nα`.
- Pole counts: `N_λ1`, `N_λ2`.
- Time span and solver tolerances in `solve(...)`.

## Output / Observables
Main outputs are:
- `sol.t`, `sol.u` from DifferentialEquations.jl.
- Each state `sol.u[it]` can be viewed with `dv = pointer(sol.u[it], p)`.
- `ObservablesTDNEGF` stores time series arrays such as:
  - `n_i` (site charge density),
  - `σx_i` (site electron spin density components),
  - `Iα` (lead charge current),
  - `Iαx` (lead spin current components).

See the example loop in `examples/01_two_terminal_square_lattice.jl` for how observables are filled and written to `examples/data/two_terminal_square_lattice_observables.jl2`.

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
