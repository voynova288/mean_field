using JLD2
using LinearAlgebra
using Printf

const TBG_REPO_ROOT = "/data/home/ziyuzhu/TBG_HartreeFock"
const MF_REPO_ROOT = "/data/home/ziyuzhu/Mean_Field"
include(joinpath(TBG_REPO_ROOT, "B0", "libs", "BM_mod.jl"))

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
        lk = parse(Int, entries["lk"]),
        lg = parse(Int, entries["lg"]),
    )
end

function build_params(theta_deg::Float64)
    params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
    initParamsWithStrain(params)
    return params
end

function theta_tag(theta_deg::Float64)
    return @sprintf("%03d", round(Int, theta_deg * 100))
end

function write_complex_tensor4(path::String, tensor::Array{ComplexF64, 4}; metadata::Vector{Pair{String, String}}=Pair{String, String}[])
    mkpath(dirname(path))
    open(path, "w") do io
        for (key, value) in metadata
            println(io, "# $(key)=$(value)")
        end
        println(io, "# n0=$(size(tensor, 1))")
        println(io, "# n1=$(size(tensor, 2))")
        println(io, "# n2=$(size(tensor, 3))")
        println(io, "# n3=$(size(tensor, 4))")
        for i0 in 1:size(tensor, 1), i1 in 1:size(tensor, 2), i2 in 1:size(tensor, 3), i3 in 1:size(tensor, 4)
            value = tensor[i0, i1, i2, i3]
            println(io, @sprintf("%d\t%d\t%d\t%d\t%.17e\t%.17e", i0 - 1, i1 - 1, i2 - 1, i3 - 1, real(value), imag(value)))
        end
    end
    return path
end

function main()
    benchmark_id = get(ENV, "BENCHMARK_ID", length(ARGS) >= 1 ? ARGS[1] : "theta_120_nu_-2_ivc_ground")
    lk_override = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing
    lg_override = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
    settings = parse_runtime_summary(benchmark_id)
    lk = lk_override === nothing ? settings.lk : lk_override
    lg = lg_override === nothing ? settings.lg : lg_override
    params = build_params(settings.theta_deg)
    latt = Lattice()
    initLattice(latt, params; lk=lk)

    scratch = tempname() * ".jld2"
    bm = HBM()
    initHBM(bm, latt, params; lg=lg, _σrotation=true, _calculate_overlap=false, fname=scratch)

    tag = theta_tag(settings.theta_deg)
    out_path = joinpath(
        MF_REPO_ROOT,
        "benchmarks",
        "b0",
        "bm_inputs",
        "bm_theta_$(tag)_lk$(lk)_lg$(lg)_uk_reference.tsv",
    )
    summary_path = joinpath(
        MF_REPO_ROOT,
        "benchmarks",
        "b0",
        "bm_inputs",
        "bm_theta_$(tag)_lk$(lk)_lg$(lg)_uk_reference_summary.txt",
    )

    metadata = Pair{String, String}[
        "benchmark_id" => benchmark_id,
        "theta_deg" => @sprintf("%.2f", settings.theta_deg),
        "lk" => string(lk),
        "lg" => string(lg),
        "source" => "julia_bm_grid_uk_reference",
        "sigma_rotation" => "true",
    ]
    write_complex_tensor4(out_path, bm.Uk; metadata=metadata)

    open(summary_path, "w") do io
        println(io, "benchmark_id=$(benchmark_id)")
        println(io, "theta_deg=$(settings.theta_deg)")
        println(io, "lk=$(lk)")
        println(io, "lg=$(lg)")
        println(io, "uk_shape=$(join(size(bm.Uk), ','))")
        println(io, "max_abs=$(maximum(abs.(bm.Uk)))")
        println(io, "fro_norm=$(norm(bm.Uk))")
    end

    rm(scratch; force=true)
    println("benchmark_id=$(benchmark_id)")
    println("uk_reference_path=$(out_path)")
    println("summary_path=$(summary_path)")
end

main()
