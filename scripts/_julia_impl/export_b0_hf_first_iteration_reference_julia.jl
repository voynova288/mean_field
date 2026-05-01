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

function write_complex_stack(path::String, stack::Array{ComplexF64, 3}; metadata::Vector{Pair{String, String}}=Pair{String, String}[])
    mkpath(dirname(path))
    open(path, "w") do io
        for (key, value) in metadata
            println(io, "# $(key)=$(value)")
        end
        println(io, "# nrow=$(size(stack, 1))")
        println(io, "# ncol=$(size(stack, 2))")
        println(io, "# nk=$(size(stack, 3))")
        for ik in 1:size(stack, 3), row in 1:size(stack, 1), col in 1:size(stack, 2)
            value = stack[row, col, ik]
            println(io, @sprintf("%d\t%d\t%d\t%.17e\t%.17e", ik - 1, row - 1, col - 1, real(value), imag(value)))
        end
    end
    return path
end

function load_initial_density_override!(hf::HartreeFock, path::String)
    fill!(hf.P, 0.0 + 0.0im)
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        fields = split(stripped, '\t')
        length(fields) == 5 || error("Expected 5 tab-separated fields, got $(length(fields)) in $(path)")
        ik = parse(Int, fields[1]) + 1
        row = parse(Int, fields[2]) + 1
        col = parse(Int, fields[3]) + 1
        real_part = parse(Float64, fields[4])
        imag_part = parse(Float64, fields[5])
        hf.P[row, col, ik] = complex(real_part, imag_part)
    end
    return nothing
end

function main()
    benchmark_id = length(ARGS) >= 1 ? ARGS[1] : "theta_120_nu_-2_ivc_ground"
    settings = parse_runtime_summary(benchmark_id)
    theta_tag = replace(@sprintf("%.2f", settings.theta_deg), "." => "")
    bm_path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "bm_inputs", "bm_theta_$(theta_tag)_lk$(settings.lk)_lg$(settings.lg).jld2")

    case_dir = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "cases", benchmark_id)
    init_override_path = joinpath(
        case_dir,
        @sprintf("initial_density_%s_seed_%03d.tsv", settings.init_mode, settings.seed),
    )
    interaction_out = joinpath(
        case_dir,
        @sprintf("reference_first_iteration_interaction_%s_seed_%03d.tsv", settings.init_mode, settings.seed),
    )
    hamiltonian_out = joinpath(
        case_dir,
        @sprintf("reference_first_iteration_hamiltonian_%s_seed_%03d.tsv", settings.init_mode, settings.seed),
    )
    density_out = joinpath(
        case_dir,
        @sprintf("reference_first_iteration_density_%s_seed_%03d.tsv", settings.init_mode, settings.seed),
    )
    summary_out = joinpath(
        case_dir,
        @sprintf("reference_first_iteration_summary_%s_seed_%03d.txt", settings.init_mode, settings.seed),
    )

    params = build_params(settings.theta_deg)
    latt = Lattice()
    initLattice(latt, params; lk=settings.lk)

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
    if isfile(init_override_path)
        load_initial_density_override!(hf, init_override_path)
    end
    hf.Λ = zeros(ComplexF64, hf.nt * latt.nk, hf.nt * latt.nk)
    ηs = ["η0", "η1", "η2", "η3"]
    σs = ["s0", "s1", "s2", "s3"]
    ns = ["n0", "n1", "n2", "n3"]
    hf.Δstr = [ns[i] * ηs[j] * σs[k] for i in 1:4 for j in 1:4 for k in 1:4]
    hf.Δ = zeros(Float64, size(hf.Δstr))

    hf.H .= hf.H0
    initial_density = copy(hf.P)
    add_HartreeFock(hf; β=1.0)
    interaction_h = hf.H .- hf.H0
    norm_convergence, λ = update_P(hf; Δ=0.0)
    updated_density = copy(hf.P)

    metadata = Pair{String, String}[
        "benchmark_id" => benchmark_id,
        "theta_deg" => @sprintf("%.2f", settings.theta_deg),
        "nu" => string(settings.nu),
        "init_mode" => settings.init_mode,
        "seed" => @sprintf("%03d", settings.seed),
        "lk" => string(settings.lk),
        "lg" => string(settings.lg),
        "source" => "julia_b0_first_iteration_reference",
        "initial_density_override" => string(isfile(init_override_path)),
    ]
    write_complex_stack(interaction_out, interaction_h; metadata=metadata)
    write_complex_stack(hamiltonian_out, hf.H; metadata=metadata)
    write_complex_stack(density_out, updated_density; metadata=metadata)

    open(summary_out, "w") do io
        println(io, "benchmark_id=$(benchmark_id)")
        println(io, "theta_deg=$(settings.theta_deg)")
        println(io, "nu=$(settings.nu)")
        println(io, "init_mode=$(settings.init_mode)")
        println(io, "seed=$(settings.seed)")
        println(io, "lk=$(settings.lk)")
        println(io, "lg=$(settings.lg)")
        println(io, "bm_path=$(bm_path)")
        println(io, "initial_density_override=$(isfile(init_override_path))")
        println(io, "mu=$(hf.μ)")
        println(io, "oda_lambda=$(λ)")
        println(io, "norm_convergence=$(norm_convergence)")
        println(io, "interaction_fro=$(norm(interaction_h))")
        println(io, "interaction_max_abs=$(maximum(abs.(interaction_h)))")
        println(io, "hamiltonian_fro=$(norm(hf.H))")
        println(io, "hamiltonian_max_abs=$(maximum(abs.(hf.H)))")
        println(io, "updated_density_fro=$(norm(updated_density))")
        println(io, "updated_density_max_abs=$(maximum(abs.(updated_density)))")
        println(io, "delta_density_fro=$(norm(updated_density .- initial_density))")
        println(io, "delta_density_max_abs=$(maximum(abs.(updated_density .- initial_density)))")
        println(io, "lowest_eigs_k0=" * join([@sprintf("%.12f", x) for x in hf.ϵk[1:8, 1]], ","))
    end

    println("benchmark_id=$(benchmark_id)")
    println("interaction_path=$(interaction_out)")
    println("hamiltonian_path=$(hamiltonian_out)")
    println("density_path=$(density_out)")
    println("summary_path=$(summary_out)")
    println("initial_density_override=$(isfile(init_override_path))")
    println("mu=$(hf.μ)")
    println("oda_lambda=$(λ)")
    println("norm_convergence=$(norm_convergence)")
end

main()
