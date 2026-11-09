# Block RHS benchmarks

Two comparisons are recorded here:

1. **Contiguous Omega traversal** (`compare_rhs_versions.jl`, commit
   `f8c378d`, now on `main`) against the previous `main` RHS. Results in
   `results/contiguous_rhs/`. See the first section below.
2. **Threaded RHS candidate** (`compare_threaded_rhs.jl`, commit `e5cd7fe`,
   not merged) against `22e885c`. Results in `results/threaded_rhs_t*.csv`.
   This script uses the workspace fields of the old `ExperimentalBlockRHSParams`
   and only runs from a checkout of `22e885c`.

## 1. Contiguous Omega traversal (f8c378d)

`compare_rhs_versions.jl` times `eom_tdnegf_blocks!` for whichever TDNEGF
version `--project` points to, and saves the RHS outputs and final solve states
so two versions can be compared afterwards:

```bash
git worktree add ../tdnegf-old 31a91cb
for T in 1 4 8; do
  julia --project=../tdnegf-old --threads=$T benchmarks/compare_rhs_versions.jl run old OUT 10
  julia --project=.             --threads=$T benchmarks/compare_rhs_versions.jl run new OUT 10
done
julia benchmarks/compare_rhs_versions.jl compare OUT old new
```

Cases and solver settings are the same as in section 2 below (49 spectral and
20 Fermi poles, two leads, spin 1/2, Vern7 on t = 0..0.5, rtol 1e-6, atol 1e-8),
with an extra 20x4 solve. Runs alternated old/new for each thread count on the
same shared machine (Julia 1.12.4, 20 logical CPUs available, load average
about 98), so absolute times are noisy; the ratios between paired runs are the
meaningful numbers.

### Results

Median time in ms, old / new (speedup = old ÷ new).

| Case | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| RHS 6x2 | 7.2 / 3.4 (2.1) | 8.3 / 2.7 (3.1) | 5.0 / 2.8 (1.8) |
| RHS 10x2 | 8.7 / 5.0 (1.8) | 10.3 / 3.8 (2.7) | 9.8 / 4.0 (2.4) |
| RHS 50x2 | 24.3 / 18.5 (1.3) | 24.2 / 16.8 (1.4) | 25.0 / 16.9 (1.5) |
| RHS 20x4 | 155.8 / 40.1 (3.9) | 138.3 / 32.8 (4.2) | 135.9 / 43.9 (3.1) |
| Solve 50x2 | 4807 / 3066 (1.6) | 4169 / 3263 (1.3) | 3747 / 3269 (1.1) |
| Solve 20x4 | 17842 / 7810 (2.3) | 14818 / 8324 (1.8) | 15378 / 7163 (2.1) |

- **Same results.** RHS outputs agree to a relative 4e-17 and final solve states
  to 6e-16; the solver takes the same number of RHS evaluations (92 for 50x2,
  82 for 20x4).
- **Allocations.** 96 bytes per call in every case without threading. The
  threaded Omega sectors (20x4 with 4 and 8 threads) allocate 14–28 kB per call
  for the task spawns.
- **Where the gain comes from.** Almost all of it is single-threaded: Omega is
  now traversed in memory order and the Psi/Omega contractions use gemm on
  preallocated matrices. Threading the Omega sectors adds little on this loaded
  machine (20x4: 40 → 33 ms with 4 threads, and 44 ms with 8). It is enabled
  only above `thread_min_Ω` elements per sector, so small systems never pay for it.
- **Full solve.** 1.6x (50x2) and 2.3x (20x4) faster with one thread. The solve
  speedup is smaller than the RHS speedup because Vern7's own vector operations
  on the state (length 344 368 for 50x2) are unchanged.

## 2. Threaded RHS candidate (e5cd7fe)

`compare_threaded_rhs.jl` compares the working tree's block RHS with the RHS
from commit `e5cd7fed39320f4b23dd98f414c9be2e4d6a5ea6`. Only the candidate
function is loaded into memory under a different name. Both implementations
use the current package types, helpers, dependencies and example setup.
No branch is merged and no production source is overwritten.

Run from the repository root:

```bash
julia --project=. --startup-file=no --threads=4 benchmarks/compare_threaded_rhs.jl benchmarks/results/threaded_rhs_t4.csv 15 5
```

Arguments after the output path are the number of RHS timing samples and the
number of short-solve samples. Set the latter to zero to skip the solver.
Run thread configurations sequentially to avoid competition between benchmarks.

### Method

- The setup function is parsed from `examples/01_two_terminal_square_lattice.jl`;
  the example's top-level simulation, saving and plotting are not executed.
- Cases: 6x2, 10x2, 50x2 and 20x4, each with two spin components, two leads,
  49 spectral poles and 20 Fermi poles, using the example's remaining defaults.
- Julia thread count is selected at process startup; BLAS always uses one thread.
- Both functions are warmed up. Each timing sample batches evaluations to target
  at least 40 ms, using the same batch size for both functions.
- Timing order alternates between main-first and candidate-first. GC remains
  enabled during timing; an incremental collection precedes each sample.
- The RHS input is a fixed seeded nonzero complex state. This exercises all
  auxiliary sectors, but is not itself a physical initial density matrix.
- Correctness is checked on both a zero state and that nonzero state, with ten
  repeated candidate evaluations each, at rtol=1e-12 and atol=1e-12.
- The full-solver comparison uses the physical zero initial state on the 50x2
  case, Vern7, t=0..0.5, rtol=1e-6, atol=1e-8, and final-state-only saving.
  Both solves are warmed up; final states and RHS evaluation counts are checked.

CSV times are milliseconds per RHS call or per complete short solve. `bytes`
means allocated bytes, not peak resident memory. RHS allocation counts are
measured separately; solve allocation counts are not measured (NaN). Error
columns compare candidate against main and are repeated on both variant rows.
`exact_equal` reports array value equality, not an assertion about bit patterns.

### Measurement environment

Measurements on 2026-11-09 used Julia 1.12.4 and an AMD EPYC 7662 system
(128 logical CPUs reported, two NUMA nodes). Full runs used the same CPU
affinity set: `6,13,19,21,28,37,40,44`, all on NUMA node 0. To reproduce that
affinity on this machine, prefix the command with
`taskset -c 6,13,19,21,28,37,40,44`.

The machine was shared and busy (load average about 102 at the start). Treat
small timing differences as inconclusive. Compare paired versions within the
same run, and inspect the minimum and p90 as well as the median. These results
do not establish scaling with threaded BLAS or with multiple concurrent solves.

Baseline commit: `22e885c9e4a94961b1c2cd613715971baa7ab23e`.
Candidate commit: `e5cd7fed39320f4b23dd98f414c9be2e4d6a5ea6`.

### Results

Median time in ms, main / candidate (speedup = main ÷ candidate; above 1 means
the candidate is faster). RHS rows are per call; the solve row is the complete
short solve.

| Case | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| RHS 6x2 | 12.1 / 13.8 (0.88) | 13.4 / 11.7 (1.15) | 12.2 / 11.0 (1.11) |
| RHS 10x2 | 9.3 / 11.9 (0.78) | 12.7 / 12.6 (1.01) | 12.1 / 12.2 (0.99) |
| RHS 50x2 | 26.1 / 26.8 (0.97) | 29.6 / 29.5 (1.00) | 27.0 / 57.0 (0.47) |
| RHS 20x4 | 168.2 / 187.5 (0.90) | 190.9 / 78.5 (2.43) | 177.6 / 94.7 (1.88) |
| Solve 50x2 | 4156 / 4271 (0.97) | 5117 / 5368 (0.95) | 4272 / 6790 (0.63) |

Allocations per RHS call: main 96 bytes (2 allocations) in every case; the
candidate allocates 0.4–4.2 MB (1758–7450 allocations), growing with the thread
count and the system size, even with a single thread. Numerical agreement:
max abs error about 1e-16, relative error about 4e-17, in every case.

### Conclusions

- The candidate only pays off on the 20x4 case (Nc = 8), where the Omega sector
  dominates the cost: 2.4x faster with 4 threads, 1.9x with 8.
- For Ny = 2 (Nc = 4), including the 50x2 case of the main example, it is
  neutral or slower, and the full solve is never faster. With 8 threads, 50x2
  becomes about 2x slower.
- With one thread the candidate is 3–22% slower, which points to the
  per-call allocation of thread-local buffers as overhead.
- The machine was heavily loaded (some inconsistencies, such as 6x2 being slower
  than 10x2, are noise), so only the large effects are reliable.

Recommendation (followed in section 1): do not merge the commit as is. If
threading is pursued, preallocate the thread-local buffers in
`ExperimentalBlockRHSParams`, thread only above a size threshold, and measure
the full solve on a wide-lead case (for example 20x4), which this benchmark
did not include.
