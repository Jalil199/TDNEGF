#!/usr/bin/env julia
# Compares the RHS of commit e5cd7fe with that of 22e885c. It relies on the
# workspace fields of the old ExperimentalBlockRHSParams, so run it from a
# checkout of 22e885c with --project=<that checkout> --threads=N.
# Usage: compare_threaded_rhs.jl OUTPUT.csv [SAMPLES=15] [SOLVE_SAMPLES=5]
using TDNEGF
using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using Printf

const ROOT = dirname(@__DIR__)
const CANDIDATE = "e5cd7fed39320f4b23dd98f414c9be2e4d6a5ea6"
const OUTPUT = abspath(ARGS[1])
const SAMPLES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 15
const SOLVE_SAMPLES = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
BLAS.set_num_threads(1)

function find_function(node, name)
    node isa Expr || return nothing
    if node.head == :function
        signature = node.args[1]
        if signature isa Expr && signature.head == :call && signature.args[1] == name
            return deepcopy(node)
        end
    end
    for child in node.args
        found = find_function(child, name)
        found === nothing || return found
    end
    return nothing
end

# Parse only the builder: never execute the example's solve/plot/output code.
setup_ast = find_function(Meta.parseall(read(joinpath(ROOT, "examples", "01_two_terminal_square_lattice.jl"), String)), :init_params_blocks)
setup_ast === nothing && error("Example setup function was not found")
Core.eval(@__MODULE__, setup_ast)

# Share all types, helpers and dependencies; change only the RHS implementation.
candidate_source = read(`git -C $ROOT show $(CANDIDATE):src/eom_tdnegf.jl`, String)
candidate_ast = find_function(Meta.parseall(candidate_source), :eom_tdnegf_blocks!)
candidate_ast === nothing && error("Candidate RHS was not found")
candidate_ast.args[1].args[1] = :eom_tdnegf_blocks_candidate!
Core.eval(TDNEGF, candidate_ast)
const SERIAL = TDNEGF.eom_tdnegf_blocks!
const THREADED = TDNEGF.eom_tdnegf_blocks_candidate!

function rhs_batch!(rhs!, du, u, p, repetitions)
    for _ in 1:repetitions
        rhs!(du, u, p, 0.0)
    end
end

function check_rhs!(p, q, u)
    original = copy(u)
    a, b = similar(u), similar(u)
    SERIAL(a, u, p, 0.0)
    max_abs = 0.0
    max_rel = 0.0
    exact = true
    for _ in 1:10
        THREADED(b, u, q, 0.0)
        delta = maximum(abs, a .- b)
        relative = norm(a .- b) / max(norm(a), eps())
        max_abs = max(max_abs, delta)
        max_rel = max(max_rel, relative)
        exact &= a == b
        @assert isapprox(a, b; rtol=1e-12, atol=1e-12)
    end
    @assert u == original "RHS modified its input state"
    return max_abs, max_rel, exact
end

function emit(io, kind, nx, ny, variant, repeats, times, bytes, allocations, abs_error, rel_error, exact)
    row = (kind, Threads.nthreads(), BLAS.get_num_threads(), nx, ny,
           variant, repeats, length(times), minimum(times) * 1e3,
           median(times) * 1e3, quantile(times, 0.9) * 1e3,
           median(bytes), allocations, abs_error, rel_error, exact)
    println(io, join(row, ','))
    flush(io)
    @printf("%s %dx%d %s threads=%d median=%.4f ms bytes=%.0f exact=%s\n",
            kind, nx, ny, variant, Threads.nthreads(), median(times)*1e3,
            median(bytes), string(exact))
    flush(stdout)
end

function bench_rhs(io, nx, ny)
    _, p, zero_state = init_params_blocks(; Nx=nx, Ny=ny)
    q = deepcopy(p)
    rng = MersenneTwister(20261109)
    check_rhs!(p, q, zero_state)
    u = randn(rng, ComplexF64, length(zero_state)) .* 0.01
    abs_error, rel_error, exact = check_rhs!(p, q, u)
    du = similar(u)
    funcs = (SERIAL, THREADED)
    params = (p, q)
    trials = Float64[]
    for k in 1:2
        rhs_batch!(funcs[k], du, u, params[k], 5)
        push!(trials, @elapsed rhs_batch!(funcs[k], du, u, params[k], 5))
    end
    repeats = clamp(ceil(Int, 0.04 / (minimum(trials) / 5)), 1, 1000)
    times = (Float64[], Float64[])
    bytes = (Float64[], Float64[])
    for sample in 1:SAMPLES
        # Alternate order to reduce bias from load, cache and CPU frequency drift.
        for k in (isodd(sample) ? (1, 2) : (2, 1))
            GC.gc(false)
            measurement = @timed rhs_batch!(funcs[k], du, u, params[k], repeats)
            push!(times[k], measurement.time / repeats)
            push!(bytes[k], measurement.bytes / repeats)
        end
    end
    for k in 1:2
        allocations = @allocations rhs_batch!(funcs[k], du, u, params[k], repeats)
        emit(io, "rhs", nx, ny, ("main", "candidate")[k], repeats,
             times[k], bytes[k], allocations / repeats, abs_error, rel_error, exact)
    end
end

function short_solve(rhs!, initial, p)
    problem = ODEProblem(rhs!, initial, (0.0, 0.5), p)
    sol = solve(problem, Vern7(); reltol=1e-6, abstol=1e-8,
                save_everystep=false, save_start=false, dense=false)
    @assert string(sol.retcode) == "Success"
    return sol
end

function bench_solve(io)
    nx, ny = 50, 2
    _, p, initial = init_params_blocks(; Nx=nx, Ny=ny)
    q = deepcopy(p)
    a = short_solve(SERIAL, initial, p)
    b = short_solve(THREADED, initial, q)
    @assert a.t == b.t
    @assert a.destats.nf == b.destats.nf
    @assert isapprox(a.u[end], b.u[end]; rtol=1e-11, atol=1e-12)
    abs_error = maximum(abs, a.u[end] .- b.u[end])
    rel_error = norm(a.u[end] .- b.u[end]) / max(norm(a.u[end]), eps())
    exact = a.u[end] == b.u[end]
    println("Short solve RHS evaluations: ", a.destats.nf)
    times = (Float64[], Float64[])
    bytes = (Float64[], Float64[])
    funcs, params = (SERIAL, THREADED), (p, q)
    for sample in 1:SOLVE_SAMPLES
        for k in (isodd(sample) ? (1, 2) : (2, 1))
            GC.gc(false)
            measurement = @timed short_solve(funcs[k], initial, params[k])
            push!(times[k], measurement.time)
            push!(bytes[k], measurement.bytes)
        end
    end
    for k in 1:2
        emit(io, "solve", nx, ny, ("main", "candidate")[k], 1,
             times[k], bytes[k], NaN, abs_error, rel_error, exact)
    end
end

function main()
    println("Julia ", VERSION, "; threads=", Threads.nthreads(), "; BLAS=", BLAS.get_num_threads())
    println("Baseline: ", strip(read(`git -C $ROOT rev-parse HEAD`, String)))
    println("Candidate RHS: ", CANDIDATE)
    mkpath(dirname(OUTPUT))
    open(OUTPUT, "w") do io
        println(io, "kind,threads,blas_threads,nx,ny,variant,batch,samples,min_ms,median_ms,p90_ms,bytes,allocations,max_abs_error,relative_error,exact_equal")
        for (nx, ny) in ((6, 2), (10, 2), (50, 2), (20, 4))
            bench_rhs(io, nx, ny)
        end
        SOLVE_SAMPLES > 0 && bench_solve(io)
    end
end

main()
