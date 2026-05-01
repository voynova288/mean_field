#!/usr/bin/env julia

using Printf

if length(ARGS) != 2
    println("Usage: julia export_b0_parameter_reference_from_julia.jl <tbg_repo_root> <outfile>")
    exit(1)
end

repo_root = abspath(ARGS[1])
outfile = abspath(ARGS[2])

include(joinpath(repo_root, "B0", "libs", "Parameters_mod.jl"))

thetas = [1.20, 1.28]

open(outfile, "w") do io
    println(io, join([
        "theta_deg",
        "dtheta_rad",
        "vf",
        "w0",
        "w1",
        "strain",
        "alpha",
        "kb",
        "g1_re",
        "g1_im",
        "g2_re",
        "g2_im",
        "a1_re",
        "a1_im",
        "a2_re",
        "a2_im",
        "theta12",
        "kt_re",
        "kt_im",
        "kb_re",
        "kb_im",
    ], '\t'))
    for theta_deg in thetas
        params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
        initParamsWithStrain(params)
        @printf(
            io,
            "%.2f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
            theta_deg,
            params.dθ,
            params.vf,
            params.w0,
            params.w1,
            params.ϵ,
            params.α,
            params.kb,
            real(params.g1),
            imag(params.g1),
            real(params.g2),
            imag(params.g2),
            real(params.a1),
            imag(params.a1),
            real(params.a2),
            imag(params.a2),
            params.θ12,
            real(params.Kt),
            imag(params.Kt),
            real(params.Kb),
            imag(params.Kb),
        )
    end
end
