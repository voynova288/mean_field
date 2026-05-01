using JLD2
using LinearAlgebra
using Printf
using Random

const TBG_REPO_ROOT = "/data/home/ziyuzhu/TBG_HartreeFock"
const MF_REPO_ROOT = "/data/home/ziyuzhu/Mean_Field"
include(joinpath(TBG_REPO_ROOT, "B0", "libs", "HF_mod.jl"))

function parse_runtime_summary(benchmark_id::String)
    path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "cases", benchmark_id, "runtime_summary.txt")
    entries = Dict{String, String}()
    for line in eachline(path)
        occursin('=', line) || continue
        key, value = split(line, '='; limit=2)
        entries[strip(key)] = strip(value)
    end
    return (
        theta_deg = parse(Float64, entries["theta_deg"]),
        nu = parse(Float64, entries["nu"]),
        init_mode = entries["init_mode"],
        seed = parse(Int, entries["seed"]),
        lk = parse(Int, entries["lk"]),
        lg = parse(Int, entries["lg"]),
    )
end

function build_params(theta_deg::Float64)
    params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
    initParamsWithStrain(params)
    return params
end

function main()
    benchmark_id = length(ARGS) >= 1 ? ARGS[1] : "theta_120_nu_-2_ivc_ground"
    lk_override = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing
    settings = parse_runtime_summary(benchmark_id)
    theta_tag = replace(@sprintf("%.2f", settings.theta_deg), "." => "")
    lk = lk_override === nothing ? settings.lk : lk_override
    bm_path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "bm_inputs", "bm_theta_$(theta_tag)_lk$(lk)_lg$(settings.lg).jld2")
    out_name = lk == settings.lk ? @sprintf("initial_density_%s_seed_%03d.tsv", settings.init_mode, settings.seed) :
                                   @sprintf("initial_density_%s_seed_%03d_lk%d.tsv", settings.init_mode, settings.seed, lk)
    out_path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "cases", benchmark_id, out_name)

    params = build_params(settings.theta_deg)
    latt = Lattice()
    initLattice(latt, params; lk=lk)

    Random.seed!(settings.seed)
    hf = HartreeFock()
    hf.params = params
    hf.latt = latt
    hf.ν = settings.nu
    hf.precision = 1e-5
    hf.fname = bm_path
    jldopen(hf.fname, "r") do file
        hf.ns, hf.nη, hf.nb = file["ns"], file["nη"], file["nb"]
        hf.nt = hf.ns * hf.nη * hf.nb
    end
    hf.P = zeros(ComplexF64, hf.nt, hf.nt, hf.latt.nk)
    hf.H = zeros(ComplexF64, size(hf.P))
    hf.Σz = zeros(ComplexF64, size(hf.P))
    hf.ϵk = zeros(Float64, hf.nt, latt.nk)
    hf.σzτz = zeros(Float64, hf.nt, latt.nk)
    hf.V0 = CoulombUnit(hf.params)
    BM_info(hf)
    init_P(hf; _Init=settings.init_mode)

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        println(io, "# benchmark_id=$(benchmark_id)")
        println(io, "# theta_deg=$(settings.theta_deg)")
        println(io, "# init_mode=$(settings.init_mode)")
        println(io, "# seed=$(settings.seed)")
        println(io, "# nt=$(hf.nt)")
        println(io, "# nk=$(hf.latt.nk)")
        for ik in 1:hf.latt.nk, row in 1:hf.nt, col in 1:hf.nt
            val = hf.P[row, col, ik]
            println(io, @sprintf("%d\t%d\t%d\t%.17e\t%.17e", ik - 1, row - 1, col - 1, real(val), imag(val)))
        end
    end

    println("benchmark_id=$(benchmark_id)")
    println("output_path=$(out_path)")
    println("nt=$(hf.nt)")
    println("nk=$(hf.latt.nk)")
    println("lk=$(lk)")
end

main()
