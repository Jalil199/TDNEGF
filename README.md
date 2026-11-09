# TDNEGF-Hybrid

**Time-dependent nonequilibrium Green's functions for open quantum systems, in Julia.**

TDNEGF-Hybrid simulates electronic transport through a finite device coupled to
semi-infinite leads, with arbitrary time dependence in the device Hamiltonian
and in the lead energies. It is intended for problems in which the transient
response is essential: switching on a bias, light pulses, spin and charge pumping,
or electrons coupled self-consistently to classical magnetization
dynamics [3].

The method is exact for noninteracting electrons and does not require
two-time Green's functions. Expanding the lead self-energies in poles turns the
Kadanoff–Baym equations into a closed set of first-order ODEs for single-time
quantities, which are integrated with the solvers of DifferentialEquations.jl.

## Physical model

The system consists of a device region described by a tight-binding Hamiltonian
$H(t)$, with orbital, spin, and site degrees of freedom $a, b$, coupled to
leads $\alpha$. Each lead couples to the device through a set of channels $n$
with coupling vectors $\xi_{n\alpha}$ and can be rigidly shifted in energy by
$\Delta_\alpha(t)$, which is how a (possibly time-dependent) bias is applied.

The lesser and greater lead self-energies are approximated by a finite sum of
exponentials in $t - t'$. Two kinds of poles $\lambda$ enter that sum:

- **spectral poles**, which come from fitting the lead density of states
  (for example a Lorentzian fit with 31 poles, or a 49-pole semicircular fit
  obtained with the minimal pole method [4, 5]), and
- **Fermi poles**, which come from a Padé/Ozaki expansion of the Fermi function
  at inverse temperature $\beta$.

Each pole carries a complex frequency $\chi_{n\lambda\alpha}$ and residues
$\Sigma^{<}_{n\lambda\alpha}$ and $\Sigma^{>}_{n\lambda\alpha}$, with
$\Gamma_{n\lambda\alpha} = i(\Sigma^{>}_{n\lambda\alpha} - \Sigma^{<}_{n\lambda\alpha})$.

## Equations of motion

With $\hbar = 1$, the propagated state is the single-particle density matrix
$\rho(t)$ of the device together with two sets of auxiliary modes: vectors
$\Psi_{n\lambda\alpha}(t)$ and matrices $\Omega_{n\lambda\alpha,\,n'\lambda'\alpha'}(t)$.
They obey [1, 2]

$$
\dot\rho = -i[H,\rho] + \left(\Pi + \Pi^\dagger\right),
\qquad
\Pi = \sum_{n\lambda\alpha} \Psi_{n\lambda\alpha}\, \xi_{n\alpha}^{T},
$$

$$
\dot\Psi_{n\lambda\alpha} = -iH\Psi_{n\lambda\alpha}
 + i\left(\chi^{*}_{n\lambda\alpha} + \Delta_\alpha\right)\Psi_{n\lambda\alpha}
 + i\,\Sigma^{<*}_{n\lambda\alpha}\,\xi_{n\alpha}
 - \Gamma^{*}_{n\lambda\alpha}\,\rho\,\xi_{n\alpha}
 - i\sum_{n'\lambda'\alpha'} \Omega_{n\lambda\alpha,\,n'\lambda'\alpha'}\,\xi_{n'\alpha'},
$$

$$
\dot\Omega_{n\lambda\alpha,\,n'\lambda'\alpha'} =
 -i\,\Gamma_{n'\lambda'\alpha'}\,\xi_{n'\alpha'}^{\dagger}\Psi_{n\lambda\alpha}
 -i\,\Gamma^{*}_{n\lambda\alpha}\,\Psi_{n'\lambda'\alpha'}^{\dagger}\xi_{n\alpha}
 -i\left(\chi_{n'\lambda'\alpha'} + \Delta_{\alpha'} - \chi^{*}_{n\lambda\alpha} - \Delta_\alpha\right)
 \Omega_{n\lambda\alpha,\,n'\lambda'\alpha'} .
$$

Only the spectral–spectral, spectral–Fermi, and Fermi–spectral sectors of
$\Omega$ are propagated, since the Fermi–Fermi block does not contribute.
Non-Hermitian baths can optionally be included through a phenomenological
$-\tfrac12\{\Gamma_{\text{bath}},\rho\}$ term in $\dot\rho$.

## Observables

All observables follow from the state at time $t$:

| Quantity | Expression | Function |
|---|---|---|
| Site charge density | $n_i = \sum_{a\in i}\operatorname{Re}\rho_{aa}$ | `obs_n_i!` |
| Site spin density | $\langle\sigma^{x,y,z}_i\rangle = \operatorname{Re}\operatorname{Tr}[\sigma^{x,y,z}\rho_{ii}]$ | `obs_σ_i!` |
| Charge current into lead $\alpha$ | $I_\alpha = 2\operatorname{Re}\operatorname{Tr}\,\Pi_\alpha$ | `obs_Ixα!` |
| Spin current into lead $\alpha$ | $I^{x,y,z}_\alpha = 2\operatorname{Re}\operatorname{Tr}[\sigma^{x,y,z}\Pi_\alpha]$ | `obs_Ixα!` |

## Conventions

- Units: $\hbar = 1$. Energies are in units of the hopping $\gamma$ and times in
  units of $\hbar/\gamma$.
- **Bias.** A bias must be applied exactly once: either through the chemical
  potentials $\mu_\alpha$ in the self-energy and Fermi-pole coefficients, or
  through a common $\mu = E_F$ together with lead shifts
  $\Delta_\alpha = \pm V/2$. Combining both doubles the effective bias.
- In the block backend, currents are indexed by auxiliary block
  (`p_blocks.blocks`). They match physical leads only when each lead is a single
  block.

## Installation

Requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url="https://github.com/Jalil199/TDNEGF.git")
```

For development from a local clone, use `Pkg.develop(path="/path/to/TDNEGF")`.

## Quick start

This is a two-terminal square-lattice wire with Rashba spin–orbit coupling and a
bias across the leads. Building the pole tensors takes a few more lines; see
`examples/01_two_terminal_square_lattice.jl` for the complete setup.

```julia
using TDNEGF
using DifferentialEquations

# Device: 6×2 square lattice, spin-1/2, one orbital per site
p_model = ModelParamsTDNEGF(Nx=6, Ny=2, Nσ=2, N_orb=1, Nα=2, N_λ1=49, N_λ2=20)
H_ab = build_H_ab(; Nx=p_model.Nx, Ny=p_model.Ny, Nσ=p_model.Nσ, N_orb=p_model.N_orb,
                  γ=1.0, γso=0.5 + 0im)
p_model.H_ab .= H_ab
p_model.H0_ab .= H_ab

# Leads: one self-energy block each, shifted by ±V/2
#   (Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL, ξ_anR built as in examples/01_...)
left  = SelfEnergyBlock(:left,  p_model.Nc, p_model.N_λ1, p_model.N_λ2, Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anL)
right = SelfEnergyBlock(:right, p_model.Nc, p_model.N_λ1, p_model.N_λ2, Σᴸ_nλ, Σᴳ_nλ, χ_nλ, ξ_anR)
Δ_blocks = ComplexF64[+0.5, -0.5]

# Propagate from the empty state
p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, [left, right], Δ_blocks, p_model)
u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)

prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, 10.0), p_blocks)
sol  = solve(prob, Vern7(); reltol=1e-6, abstol=1e-8, dense=false)
```

Observables are evaluated by obtaining views of $\rho$, $\Psi$, and $\Omega$
for each saved state with `pointer_blocks(...)` and passing them to the
`obs_*` routines.

Because $H$ and $\Delta_\alpha$ live in the parameter object, time-dependent
drives are implemented by updating `p_blocks.H_ab` or `p_blocks.Δ_blocks`
(for example, in a callback) without rebuilding the self-energy blocks.

## Implementation

There are two interchangeable implementations of the equations above:

- **Block backend** (`eom_tdnegf_blocks!`, `ExperimentalBlockRHSParams`,
  `SelfEnergyBlock`). This is the recommended one. Each block can have its own
  number of channels and poles. Static pole data lives in `SelfEnergyBlock`, and
  dynamic quantities ($H$, $\Delta$) and workspaces live in the parameter object.
  The `Experimental` prefix is historical.
- **Legacy backend** (`eom_tdnegf!`, `ModelParamsTDNEGF`). It uses a rectangular
  layout with the same number of channels and poles for every lead. It is kept
  for reproducing earlier results and for cross-validation.

Both are in-place right-hand sides that work on preallocated buffers.

## Validation

- **Stationary limit.** For a clean wire under small DC bias, the long-time
  current reproduces the Landauer conductance, including quantized steps as
  $E_F$ crosses subband edges (`test/test_standard_landauer.jl`,
  `examples/conductance_verification.jl`).
- **Cross-checks.** The block and legacy backends give the same right-hand side
  and the same short-time trajectories to $10^{-12}$ (`test/test_eom_tdnegf_blocks.jl`).
  The legacy N49 benchmark is reproduced in
  `examples/notebooks/05_legacy_n49_benchmark.ipynb`.

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
test/        unit tests and validation notebooks
docs/        developer notes: architecture, propagation, observables
benchmarks/  performance comparisons
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
