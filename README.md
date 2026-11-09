# TDNEGF-Hybrid

**Time-dependent nonequilibrium Green's functions for open quantum systems, in Julia.**

TDNEGF-Hybrid simulates electronic transport through a finite device coupled to
semi-infinite leads, with arbitrary time dependence in the device Hamiltonian
and in the lead energies. It is intended for problems in which the transient
response is essential: switching on a bias, light pulses, spin and charge pumping,
or electrons coupled self-consistently to classical magnetization
dynamics [3].

For noninteracting electrons the method is exact up to the accuracy of the
pole expansion, and it does not require
two-time Green's functions. Expanding the lead self-energies in poles turns the
Kadanoff–Baym equations into a closed set of first-order ODEs for single-time
quantities, which are integrated with the solvers of DifferentialEquations.jl.

## Physical model

The system consists of a device region described by a tight-binding Hamiltonian
$`H(t)`$, with orbital, spin, and site degrees of freedom $`a, b`$, coupled to
leads $`\alpha`$. Each lead couples to the device through a set of channels $`n`$
with coupling vectors $`\xi_{n\alpha}`$ and can be rigidly shifted in energy by
$`\Delta_\alpha(t)`$, which is how a (possibly time-dependent) bias is applied.

The lesser and greater lead self-energies are approximated by a finite sum of
exponentials in $`t - t'`$. Two kinds of poles $`\lambda`$ enter that sum:

- **spectral poles**, which come from fitting the lead density of states
  (for example a Lorentzian fit with 31 poles, or a 49-pole semicircular fit
  obtained with the minimal pole method [4, 5]), and
- **Fermi poles**, which come from a Padé expansion of the Fermi function
  at inverse temperature $`\beta`$.

Each pole carries a complex frequency $`\chi_{n\lambda\alpha}`$ and residues
$`\Sigma^{<}_{n\lambda\alpha}`$ and $`\Sigma^{>}_{n\lambda\alpha}`$, with
$`\Gamma_{n\lambda\alpha} = i(\Sigma^{>}_{n\lambda\alpha} - \Sigma^{<}_{n\lambda\alpha})`$.

## Equations of motion

With $`\hbar = 1`$, the propagated state is the single-particle density matrix
$`\rho(t)`$ of the device together with two sets of auxiliary modes: vectors
$`\Psi_{n\lambda\alpha}(t)`$ and matrices $`\Omega_{n\lambda\alpha,\,n'\lambda'\alpha'}(t)`$.
They obey [1, 2]

```math
\dot\rho = -i[H,\rho] + \left(\Pi + \Pi^\dagger\right),
\qquad
\Pi = \sum_{n\lambda\alpha} \Psi_{n\lambda\alpha}\, \xi_{n\alpha}^{T},
```

```math
\dot\Psi_{n\lambda\alpha} = -iH\Psi_{n\lambda\alpha}
 + i\left(\chi^{*}_{n\lambda\alpha} + \Delta_\alpha\right)\Psi_{n\lambda\alpha}
 + i\,\Sigma^{<*}_{n\lambda\alpha}\,\xi_{n\alpha}
 - \Gamma^{*}_{n\lambda\alpha}\,\rho\,\xi_{n\alpha}
 - i\sum_{n'\lambda'\alpha'} \Omega_{n\lambda\alpha,\,n'\lambda'\alpha'}\,\xi_{n'\alpha'},
```

```math
\dot\Omega_{n\lambda\alpha,\,n'\lambda'\alpha'} =
 -i\,\Gamma_{n'\lambda'\alpha'}\,\xi_{n'\alpha'}^{\dagger}\Psi_{n\lambda\alpha}
 -i\,\Gamma^{*}_{n\lambda\alpha}\,\Psi_{n'\lambda'\alpha'}^{\dagger}\xi_{n\alpha}
 -i\left(\chi_{n'\lambda'\alpha'} + \Delta_{\alpha'} - \chi^{*}_{n\lambda\alpha} - \Delta_\alpha\right)
 \Omega_{n\lambda\alpha,\,n'\lambda'\alpha'} .
```

Only the spectral–spectral, spectral–Fermi, and Fermi–spectral sectors of
$`\Omega`$ are propagated.
Non-Hermitian baths can optionally be included through a phenomenological
$`-\frac{1}{2}\{\Gamma_{\mathrm{bath}},\rho\}`$ term in $`\dot\rho`$.

## Observables

All observables follow from the state at time $`t`$:

| Quantity | Expression | Function |
|---|---|---|
| Site charge density | $`n_i = \sum_{a\in i}\mathrm{Re}\,\rho_{aa}`$ | `obs_n_i!` |
| Site spin density | $`\langle\sigma^{x,y,z}_i\rangle = \mathrm{Re}\,\mathrm{Tr}[\sigma^{x,y,z}\rho_{ii}]`$ | `obs_σ_i!` |
| Charge current of lead $`\alpha`$ | $`I_\alpha = 2\mathrm{Re}\,\mathrm{Tr}\,\Pi_\alpha`$ | `obs_Ixα!` |
| Spin current of lead $`\alpha`$ | $`I^{x,y,z}_\alpha = 2\mathrm{Re}\,\mathrm{Tr}[\sigma^{x,y,z}\Pi_\alpha]`$ | `obs_Ixα!` |

## Conventions

- Units: $`\hbar = 1`$. Energies are in units of the hopping $`\gamma`$ and times in
  units of $`\hbar/\gamma`$.
- **Bias.** A bias must be applied exactly once: either through the chemical
  potentials $`\mu_\alpha`$ in the self-energy and Fermi-pole coefficients, or
  through a common $`\mu = E_F`$ together with lead shifts
  $`\Delta_\alpha = \pm V/2`$. Combining both doubles the effective bias.
- In the block backend, currents are indexed by auxiliary block
  (`p_blocks.blocks`). They match physical leads only when each lead is a single
  block.

## Installation

Requires Julia 1.11 or newer.

```julia
using Pkg
Pkg.add(url="https://github.com/Jalil199/TDNEGF.git")
```

For development from a local clone, use `Pkg.develop(path="/path/to/TDNEGF")`.

## Quick start

The script below is self-contained. It drives a clean two-terminal wire with a
small DC bias, propagates it to the steady state, and compares the resulting
conductance with the Landauer formula. At $`E_F = 0`$ the wire has four open
channels (two transverse modes × two spins), so both values should be
$`4\,e^2/h`$.

```julia
using TDNEGF
using DifferentialEquations

# Device: clean 4×2 square-lattice wire, spin-1/2, one orbital per site
Nx, Ny, Nσ, N_orb = 4, 2, 2, 1
N_λ1, N_λ2 = 49, 20          # spectral poles (semicircle fit) + Padé Fermi poles
β  = 40.0                    # inverse temperature (1/γ)
E_F, δV = 0.0, 0.01          # Fermi energy and small DC bias

p_model = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
p_model.H_ab .= build_H_ab(; Nx, Ny, Nσ, N_orb, γ=1.0, γso=0.0 + 0im)

# Leads: one self-energy block per lead, each at its own chemical potential
Rλ, zλ = load_poles_square(N_λ1, N_λ2)
function lead_block(name, μ, xcol)
    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β, γ=1.0, μ)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β, γ=1.0, μ)
    χ  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β, γ=1.0, μ)
    ξ  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol, y_coup=1:Ny)
    return SelfEnergyBlock(name, p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ)
end
blocks   = [lead_block(:left, E_F + δV/2, 1), lead_block(:right, E_F - δV/2, Nx)]
Δ_blocks = ComplexF64[0, 0]  # bias already in μ_α; apply it only once

# Propagate from the empty device until the current is stationary
p = ExperimentalBlockRHSParams(p_model.H_ab, blocks, Δ_blocks, p_model)
u0 = zeros(ComplexF64, p.dims_ρ_ab[1]^2 + p.aux_layout.total_size)
sol = solve(ODEProblem(eom_tdnegf_blocks!, u0, (0.0, 200.0), p), Vern7();
            reltol=1e-8, abstol=1e-10, saveat=1.0)

# Observables along the trajectory
obs = ObservablesTDNEGF(p_model; N_tmax=length(sol.t), N_leads=length(blocks))
for (it, ut) in enumerate(sol.u)
    obs.idx = it
    ptr = pointer_blocks(ut, p.dims_ρ_ab, p.aux_layout)
    obs_n_i!(ptr, p_model, obs)
    obs_Ixα!(ptr, p, obs)
end

# Steady state: conductance in units of e²/h, compared with Landauer
G_tdnegf = 2π * obs.Iα[1, end] / δV
G_landauer = 2π * landauer_conductance(E_F; δV, β, Nx, Ny, Nσ, N_orb)
println("G (TDNEGF)   = ", round(G_tdnegf; digits=4), " e²/h")
println("G (Landauer) = ", round(G_landauer; digits=4), " e²/h")
```

Expected output:

```
G (TDNEGF)   = 4.0 e²/h
G (Landauer) = 4.0 e²/h
```

The first run takes a few minutes, most of it spent on compilation.

### Parameters

| Symbol | Meaning |
|---|---|
| `Nx`, `Ny` | Device length and width in lattice sites |
| `Nσ` | Spin components (2 for spin-1/2) |
| `N_orb` | Orbitals per site |
| `Nα` | Number of leads |
| `γ` | Nearest-neighbour hopping; sets the energy unit |
| `γso` | Rashba spin–orbit hopping |
| `β` | Inverse temperature of the leads, in units of $`1/\gamma`$ |
| `μ` | Chemical potential of a lead |
| `E_F`, `δV` | Fermi energy and bias; the leads sit at $`E_F \pm \delta V/2`$ |
| `N_λ1` | Spectral poles of the lead self-energy (49: semicircle fit, 31: Lorentzian fit) |
| `N_λ2` | Padé poles of the Fermi function |
| `xcol`, `y_coup` | Device column and rows to which a lead is attached |
| `Δ_blocks` | Rigid energy shift of each lead block (for AC or time-dependent bias) |
| `reltol`, `abstol` | ODE solver tolerances |

### Outputs

| Field | Meaning |
|---|---|
| `obs.t` | Saved times, in units of $`\hbar/\gamma`$ |
| `obs.n_i[i, t]` | Charge density on site `i` |
| `obs.σx_i[i, k, t]` | Electronic spin density on site `i`, component `k` = x, y, z |
| `obs.Iα[α, t]` | Charge current of lead (block) `α`, in units of $`e\gamma/\hbar`$ |
| `obs.Iαx[α, k, t]` | Spin current of lead (block) `α`, component `k` |

For spin densities, call `obs_σ_i!(ptr, p_model, obs)` inside the loop.
Because $`H`$ and $`\Delta_\alpha`$ live in the parameter object, time-dependent
drives are implemented by updating `p.H_ab` or `p.Δ_blocks` (for example, in a
callback) without rebuilding the self-energy blocks. More complete workflows,
including spin–orbit coupling and coupling to classical spins, are in
`examples/`.

## Implementation

There are two interchangeable implementations of the equations above:

- **Block backend** (`eom_tdnegf_blocks!`, `ExperimentalBlockRHSParams`,
  `SelfEnergyBlock`). This is the recommended one. Each block can have its own
  number of channels and poles. Static pole data lives in `SelfEnergyBlock`, and
  dynamic quantities ($`H`$, $`\Delta`$) and workspaces live in the parameter object.
  The `Experimental` prefix is historical.
- **Legacy backend** (`eom_tdnegf!`, `ModelParamsTDNEGF`). It uses a rectangular
  layout with the same number of channels and poles for every lead. It is kept
  for reproducing earlier results and for cross-validation.

Both are in-place right-hand sides that work on preallocated buffers.

## Pole tables for other leads

The spectral poles are read from `data/z_<name>_N<N>.txt` and
`data/R_<name>_N<N>.txt`. Two scripts in `tools/` generate them for any lead
density of states: the built-in semicircle, a two-column file `(ω, A(ω))`, or a
NumPy expression in `w`.

```bash
# Minimal pole method [4, 5] (needs MiniPole:
#   pip install git+https://github.com/Green-Phys/MiniPole.git kneed)
python tools/generate_poles.py semicircle --M 25
python tools/generate_poles.py table my_dos.txt --M 40 --name MyLead

# Sum of Lorentzians with non-negative weights (numpy/scipy only)
python tools/fit_lorentzians.py table my_dos.txt --K 40 --name MyLead
```

```julia
Rλ, zλ = load_poles_files("data/z_MyLead_N40.txt", "data/R_MyLead_N40.txt")
```

The fitted $`\Gamma(\omega)`$ must be non-negative on the whole real axis.
Otherwise the self-energy is not retarded, and some modes grow slowly instead
of decaying, which shows up at long times. A sum of Lorentzians satisfies this
by construction. A minimal-pole fit is much more accurate for the same number
of poles, but its small fit error oscillates around zero wherever
$`A(\omega)`$ vanishes. `generate_poles.py` checks the fit on a grid that
resolves every pole and, if needed, adds one broad positive Lorentzian (one
extra pole, with a height of the order of the fit error) that makes
$`\Gamma \ge 0`$ everywhere.

## Validation

- **Stationary reference.** `src/stationary.jl` implements the Landauer
  transmission of the lead–device–lead system, which serves as the reference
  for the long-time TDNEGF current. Its quantized steps are checked in
  `test/test_standard_landauer.jl`.
- **Backend consistency.** The block and legacy backends give the same
  right-hand side and the same short-time trajectories to within $`10^{-12}`$
  (`test/test_eom_tdnegf_blocks.jl`).
- **Legacy benchmark.** `examples/notebooks/05_legacy_n49_benchmark.ipynb`
  compares the current implementation with the original code.

Run the test suite with `julia --project -e 'using Pkg; Pkg.test()'`.

## Examples

| | |
|---|---|
| `notebooks/00_current_workflow_guide.ipynb` | Walkthrough of the full workflow |
| `01_two_terminal_square_lattice` (`.jl` / `.ipynb`) | Biased two-terminal wire with spin–orbit coupling |
| `notebooks/02_spin_dynamics_and_electrons.ipynb` | Electrons coupled to classical spin (LLG) dynamics via Sunny.jl [6] |
| `notebooks/03_nonhermitian_top_bath_square_lattice.ipynb` | Additional non-Hermitian bath |
| `04_blocks_1d_twochannel_benchmark` (`.jl` / `.ipynb`) | 1D two-channel benchmark |
| `notebooks/05_legacy_n49_benchmark.ipynb` | Comparison with the legacy code |

All paths are relative to `examples/`.

## Repository layout

```
src/         package source (types, Hamiltonians, self-energies, EOM, observables, stationary limit)
examples/    scripts and notebooks
data/        pole tables for the lead self-energies
tools/       pole-table generators (minimal pole method, Lorentzian fit)
test/        unit tests and validation notebooks
docs/        developer notes: architecture, propagation, observables
legacy/      original code, kept for reference
```

## References

**Auxiliary-mode TDNEGF formalism**

1. A. Croy and U. Saalmann, *Propagation scheme for nonequilibrium dynamics of
   electron transport in nanoscale devices*, Phys. Rev. B **80**, 245311 (2009).
   [doi:10.1103/PhysRevB.80.245311](https://doi.org/10.1103/PhysRevB.80.245311)
2. B. S. Popescu and A. Croy, *Efficient auxiliary-mode approach for
   time-dependent nanoelectronics*, New J. Phys. **18**, 093044 (2016).
   [doi:10.1088/1367-2630/18/9/093044](https://doi.org/10.1088/1367-2630/18/9/093044)

**Quantum–classical hybrid TDNEGF + LLG**

3. M. D. Petrović, B. S. Popescu, U. Bajpai, P. Plecháč, and B. K. Nikolić,
   *Spin and charge pumping by a steady or pulse-current-driven magnetic domain
   wall: A self-consistent multiscale time-dependent quantum-classical hybrid
   approach*, Phys. Rev. Applied **10**, 054038 (2018).
   [doi:10.1103/PhysRevApplied.10.054038](https://doi.org/10.1103/PhysRevApplied.10.054038)

**Minimal pole representation of the lead self-energies**

4. L. Zhang and E. Gull, *Minimal pole representation and controlled analytic
   continuation of Matsubara response functions*, Phys. Rev. B **110**, 035154
   (2024). [arXiv:2312.10576](https://arxiv.org/abs/2312.10576)
5. L. Zhang, A. Erpenbeck, Y. Yu, and E. Gull, *Minimal pole representation for
   spectral functions*, J. Chem. Phys. (2025).
   [doi:10.1063/5.0273763](https://doi.org/10.1063/5.0273763),
   [arXiv:2504.01163](https://arxiv.org/abs/2504.01163)

**Classical spin dynamics**

6. D. Dahlbom *et al.*, *Sunny.jl: A Julia package for spin dynamics*,
   J. Open Source Softw. **10**, 8138 (2025).
   [doi:10.21105/joss.08138](https://doi.org/10.21105/joss.08138)

## Citation and license

Citation information and a license will be added.
