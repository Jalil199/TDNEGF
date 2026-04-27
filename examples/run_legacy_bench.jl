#!/usr/bin/env julia
#=
run_legacy_bench.jl
====================
Corre el código legacy TDNEGF_exciton_B con parámetros mínimos
(sin LLG, sin luz, sin bias) para generar datos de referencia
para el benchmark N49.

Guarda en examples/data/:
  legacy_bench_curr.txt  — corriente [I_L  I_R] por fila temporal
  legacy_bench_cden.txt  — densidad de carga por sitio, por fila temporal

Uso:
  julia examples/run_legacy_bench.jl

IMPORTANTE: este script corre el código viejo tal cual,
con nx=10, ny=2, n=40, sin modificaciones.
Los parámetros deben coincidir exactamente con los del notebook 05.
=#

using DelimitedFiles

const LEGACY_BASE = "/home/jalil/jalil_codes/Project_Exciton/TDNEGF_exciton_B"
const OUT_DIR     = joinpath(dirname(@__DIR__), "examples", "data")

mkpath(OUT_DIR)

# ─── Parámetros congelados (idénticos al notebook 05) ──────────────────────
const NX     = 10
const NY     = 2
const N_DOTS = NX * 4        # = 40 (notación legacy: n)
const T_END  = 200.0
const T_STEP = 1.0           # paso del integrador legacy (cada 10 pasos originales de 0.1)

# ─── Cambiar al directorio legacy para que los includes funcionen ───────────
cd(LEGACY_BASE) do
    include(joinpath(LEGACY_BASE, "main_parallel.jl"))   # carga todos los módulos del legacy

    Base.invokelatest(main;
        # Geometría
        nx      = NX,
        ny      = NY,
        n       = N_DOTS,
        n_sites = N_DOTS,
        # Hamiltoniano
        tc1     = -1.0,
        tc2     = -1.0,
        tc      = -0.5,
        tv      = 0.6,
        Delta   = 6.0,
        t0      = -1.0,
        # LLG y luz: off
        j_sd    = 0.0,
        run_llg = false,
        U0      = 0.0,
        A_max   = 0.0,
        # Temperatura
        Temp    = 300.0,
        N_poles = 30,
        # Tiempo
        t_end   = T_END,
        t_step  = T_STEP,
        t_0     = 0.0,
        # Observables
        curr    = true,
        cden    = true,
        scurr   = false,
        sden_eq = false,
        sden_neq= false,
        rho     = false,
        sclas   = false,
        bcurrs  = false,
        # Nombre del run
        name    = "legacy_bench",
    )
end

# Copiar los datos del directorio legacy al directorio de output
src_curr = joinpath(LEGACY_BASE, "data", "cc_legacy_bench_jl.txt")
src_cden = joinpath(LEGACY_BASE, "data", "cden_legacy_bench_jl.txt")

dst_curr = joinpath(OUT_DIR, "legacy_bench_curr.txt")
dst_cden = joinpath(OUT_DIR, "legacy_bench_cden.txt")

if isfile(src_curr)
    cp(src_curr, dst_curr; force=true)
    println("✓ Corriente guardada en: $dst_curr")
else
    @warn "No se encontró el archivo de corriente legacy: $src_curr"
end

if isfile(src_cden)
    cp(src_cden, dst_cden; force=true)
    println("✓ Densidad guardada en:  $dst_cden")
else
    @warn "No se encontró el archivo de densidad legacy: $src_cden"
end

println("Listo. Abrir el notebook 05 para comparar con N49.")
