#!/usr/bin/env julia
#=
run_legacy_hbar1.jl
====================
Igual que run_legacy_bench.jl pero con hbar=1 en el codigo legacy.
Objetivo: verificar si hbar=1 en el viejo da lo mismo que el codigo nuevo.

IMPORTANTE: antes de correr esto, global_parameters.jl ya tiene hbar=1.
Revertir despues con hbar = 0.658211928e0.

Guarda en examples/data/:
  legacy_hbar1_curr.txt
  legacy_hbar1_cden.txt
=#

using DelimitedFiles

const LEGACY_BASE = "/home/jalil/jalil_codes/Project_Exciton/TDNEGF_exciton_B"
const OUT_DIR     = joinpath(dirname(@__DIR__), "examples", "data")

mkpath(OUT_DIR)

const NX     = 10
const NY     = 2
const N_DOTS = NX * 4
const T_END  = 50.0
const T_STEP = 1.0

cd(LEGACY_BASE) do
    include(joinpath(LEGACY_BASE, "main_parallel.jl"))

    Base.invokelatest(main;
        nx      = NX,
        ny      = NY,
        n       = N_DOTS,
        n_sites = N_DOTS,
        tc1     = -1.0,
        tc2     = -1.0,
        tc      = -0.5,
        tv      = 0.6,
        Delta   = 6.0,
        t0      = -1.0,
        j_sd    = 0.0,
        run_llg = false,
        U0      = 0.0,
        A_max   = 0.0,
        Temp    = 300.0,
        N_poles = 30,
        t_end   = T_END,
        t_step  = T_STEP,
        t_0     = 0.0,
        curr    = true,
        cden    = true,
        scurr   = false,
        sden_eq = false,
        sden_neq= false,
        rho     = false,
        sclas   = false,
        bcurrs  = false,
        name    = "legacy_hbar1",
    )
end

src_curr = joinpath(LEGACY_BASE, "data", "cc_legacy_hbar1_jl.txt")
src_cden = joinpath(LEGACY_BASE, "data", "cden_legacy_hbar1_jl.txt")

dst_curr = joinpath(OUT_DIR, "legacy_hbar1_curr.txt")
dst_cden = joinpath(OUT_DIR, "legacy_hbar1_cden.txt")

if isfile(src_curr)
    cp(src_curr, dst_curr; force=true)
    println("✓ Corriente (hbar=1) guardada en: $dst_curr")
else
    @warn "No se encontró: $src_curr"
end

if isfile(src_cden)
    cp(src_cden, dst_cden; force=true)
    println("✓ Densidad (hbar=1) guardada en:  $dst_cden")
else
    @warn "No se encontró: $src_cden"
end

println("Listo. Comparar legacy_hbar1_curr.txt con benchmark_legacy_n49.jld2.")
