#!/usr/bin/env julia
# Times eom_tdnegf_blocks! for whichever TDNEGF version --project points to.
# Run once per version (e.g. a `git worktree` of the baseline and the working
# tree), then compare the CSVs and the saved states with `compare` mode.
#
# Usage:
#   julia --project=<tree> --threads=N compare_rhs_versions.jl run LABEL OUTDIR [SAMPLES=10]
#   julia compare_rhs_versions.jl compare OUTDIR LABEL_A LABEL_B
using LinearAlgebra
using Random
using Statistics
using Printf
using Serialization

if ARGS[1] == "run"
    using TDNEGF, DifferentialEquations
end

const CASES = ((6, 2), (10, 2), (50, 2), (20, 4))
const SOLVE_CASES = ((50, 2), (20, 4))

function find_function(node, name)
    node isa Expr || return nothing
    if node.head == :function && node.args[1] isa Expr && node.args[1].head == :call && node.args[1].args[1] == name
        return deepcopy(node)
    end
    for child in node.args
        found = find_function(child, name)
        found === nothing || return found
    end
    return nothing
end

function run_version(label, outdir, samples)
    BLAS.set_num_threads(1)
    root = dirname(dirname(pathof(TDNEGF)))
    # Parse only the example's builder; never execute its solve/plot/output code.
    setup = find_function(Meta.parseall(read(joinpath(root, "examples", "01_two_terminal_square_lattice.jl"), String)), :init_params_blocks)
    Core.eval(Main, setup)
    rhs! = TDNEGF.eom_tdnegf_blocks!
    mkpath(outdir)
    states = Dict{String,Any}()
    open(joinpath(outdir, "$(label)_t$(Threads.nthreads()).csv"), "w") do io
        println(io, "kind,label,threads,nx,ny,batch,samples,min_ms,median_ms,p90_ms,bytes")
        for (nx, ny) in CASES
            _, p, u0 = Base.invokelatest(init_params_blocks; Nx=nx, Ny=ny)
            u = randn(MersenneTwister(20261109), ComplexF64, length(u0)) .* 0.01
            du = similar(u)
            rhs!(du, u, p, 0.0)
            states["rhs_$(nx)x$(ny)"] = copy(du)
            once = @elapsed rhs!(du, u, p, 0.0)
            batch = clamp(ceil(Int, 0.04 / once), 1, 1000)
            times = Float64[]
            for _ in 1:samples
                GC.gc(false)
                push!(times, @elapsed(for _ in 1:batch; rhs!(du, u, p, 0.0); end) / batch)
            end
            bytes = @allocated rhs!(du, u, p, 0.0)
            row = ("rhs", label, Threads.nthreads(), nx, ny, batch, samples,
                   1e3minimum(times), 1e3median(times), 1e3quantile(times, 0.9), bytes)
            println(io, join(row, ',')); flush(io)
            @printf("rhs   %2dx%d  median=%8.2f ms  bytes=%d\n", nx, ny, 1e3median(times), bytes)
        end
        for (nx, ny) in SOLVE_CASES
            _, p, u0 = Base.invokelatest(init_params_blocks; Nx=nx, Ny=ny)
            prob = ODEProblem(rhs!, u0, (0.0, 0.5), p)
            solve_once() = solve(prob, Vern7(); reltol=1e-6, abstol=1e-8,
                                 save_everystep=false, save_start=false, dense=false)
            sol = solve_once()
            states["solve_$(nx)x$(ny)"] = copy(sol.u[end])
            states["nf_$(nx)x$(ny)"] = sol.stats.nf
            n = max(1, samples ÷ 2)
            times = Float64[]
            for _ in 1:n
                GC.gc(false)
                push!(times, @elapsed solve_once())
            end
            row = ("solve", label, Threads.nthreads(), nx, ny, 1, n,
                   1e3minimum(times), 1e3median(times), 1e3quantile(times, 0.9), NaN)
            println(io, join(row, ',')); flush(io)
            @printf("solve %2dx%d  median=%8.1f ms  nf=%d\n", nx, ny, 1e3median(times), sol.stats.nf)
        end
    end
    serialize(joinpath(outdir, "$(label)_t$(Threads.nthreads())_states.jls"), states)
end

function compare_states(outdir, a, b)
    for file in sort(readdir(outdir))
        m = match(Regex("^$(a)_t(\\d+)_states\\.jls\$"), file)
        m === nothing && continue
        other = joinpath(outdir, "$(b)_t$(m[1])_states.jls")
        isfile(other) || continue
        A, B = deserialize(joinpath(outdir, file)), deserialize(other)
        println("threads=", m[1])
        for key in sort(collect(keys(A)))
            if startswith(key, "nf_")
                println("  $key: $(A[key]) vs $(B[key])")
            else
                @printf("  %-12s rel.err = %.2e\n", key, norm(A[key] - B[key]) / norm(A[key]))
            end
        end
    end
end

if ARGS[1] == "run"
    run_version(ARGS[2], abspath(ARGS[3]), length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10)
elseif ARGS[1] == "compare"
    compare_states(abspath(ARGS[2]), ARGS[3], ARGS[4])
end
