using JLD2
using LinearAlgebra
using Printf
using Random
using Statistics

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

function offdiag_flavor_norm(hf::HartreeFock)
    sectors = flavor_block_indices(hf)
    mask = falses(hf.nt, hf.nt)
    for inds in sectors
        mask[inds, inds] .= true
    end
    total = 0.0
    for ik in 1:size(hf.P, 3)
        block = copy(view(hf.P, :, :, ik))
        block[mask] .= 0.0 + 0.0im
        total += sum(abs2, block)
    end
    return sqrt(total)
end

function restricted_gap_estimate(hf::HartreeFock)
    νnorm = round(Int, (hf.ν + 4) / 8 * length(hf.ϵk))
    sorted = sort(hf.ϵk[:])
    if νnorm <= 0 || νnorm >= length(sorted)
        return NaN
    end
    return sorted[νnorm + 1] - sorted[νnorm]
end

function occupied_sigma_mean(hf::HartreeFock)
    νnorm = round(Int, (hf.ν + 4) / 8 * length(hf.ϵk))
    order = sortperm(hf.ϵk[:])[1:νnorm]
    return mean(hf.σzτz[order])
end

function build_projector_update!(hf::HartreeFock)
    νnorm = round(Int, (hf.ν + 4) / 8 * size(hf.H, 1) * size(hf.H, 3))
    vecs = similar(hf.H)
    for ik in 1:size(hf.H, 3)
        hf.ϵk[:, ik], vecs[:, :, ik] = eigen(Hermitian(view(hf.H, :, :, ik)))
        hf.σzτz[:, ik] = real(diag(view(vecs, :, :, ik)' * view(hf.Σz, :, :, ik) * view(vecs, :, :, ik)))
    end

    iϵ_sorted = sortperm(hf.ϵk[:])
    iϵ_occupied = iϵ_sorted[1:νnorm]
    iband_occupied = (iϵ_occupied .- 1) .% size(hf.ϵk, 1) .+ 1
    ik_occupied = (iϵ_occupied .- 1) .÷ size(hf.ϵk, 1) .+ 1

    hf.μ = find_chemicalpotential(hf.ϵk[:], (hf.ν + 4) / 8)
    hf.Δ .= calculate_valley_spin_band_order_parameters(hf)

    P_new = zeros(ComplexF64, size(hf.P))
    for ik in 1:size(hf.P, 3)
        occupied_vecs = vecs[:, iband_occupied[ik_occupied .== ik], ik]
        P_new[:, :, ik] = conj(occupied_vecs) * transpose(occupied_vecs) - 0.5 * I
    end
    return P_new
end

function main()
    benchmark_id = length(ARGS) >= 1 ? ARGS[1] : "theta_120_nu_-2_ivc_ground"
    iteration_target = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    iteration_target >= 1 || error("Expected iteration_target >= 1, got $(iteration_target)")

    settings = parse_runtime_summary(benchmark_id)
    theta_tag = replace(@sprintf("%.2f", settings.theta_deg), "." => "")
    bm_path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "bm_inputs", "bm_theta_$(theta_tag)_lk$(settings.lk)_lg$(settings.lg).jld2")
    isfile(bm_path) || error("Missing BM overlap metadata: $(bm_path)")

    case_dir = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "cases", benchmark_id)
    stem = @sprintf("reference_iteration_%03d", iteration_target)
    init_override_path = joinpath(case_dir, @sprintf("initial_density_%s_seed_%03d.tsv", settings.init_mode, settings.seed))
    input_density_out = joinpath(case_dir, @sprintf("%s_input_density_%s_seed_%03d.tsv", stem, settings.init_mode, settings.seed))
    interaction_out = joinpath(case_dir, @sprintf("%s_interaction_%s_seed_%03d.tsv", stem, settings.init_mode, settings.seed))
    hamiltonian_out = joinpath(case_dir, @sprintf("%s_hamiltonian_%s_seed_%03d.tsv", stem, settings.init_mode, settings.seed))
    updated_density_out = joinpath(case_dir, @sprintf("%s_updated_density_%s_seed_%03d.tsv", stem, settings.init_mode, settings.seed))
    summary_out = joinpath(case_dir, @sprintf("%s_summary_%s_seed_%03d.txt", stem, settings.init_mode, settings.seed))

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

    snapshot_input_density = Array{ComplexF64, 3}(undef, 0, 0, 0)
    snapshot_interaction = Array{ComplexF64, 3}(undef, 0, 0, 0)
    snapshot_hamiltonian = Array{ComplexF64, 3}(undef, 0, 0, 0)
    snapshot_updated_density = Array{ComplexF64, 3}(undef, 0, 0, 0)
    snapshot_lambda = NaN
    snapshot_norm_raw = NaN
    snapshot_norm_mixed = NaN
    snapshot_energy = NaN

    for iteration in 1:iteration_target
        hf.H .= hf.H0
        input_density = copy(hf.P)
        add_HartreeFock(hf; β=1.0)
        interaction_h = hf.H .- hf.H0
        energy = compute_HF_energy(interaction_h, hf.H0, hf.P)
        P_new = build_projector_update!(hf)
        λ = oda_parametrization(hf, P_new .- hf.P; β=1.0)
        mixed_P = λ .* P_new .+ (1 - λ) .* hf.P
        norm_raw = calculate_norm_convergence(P_new, hf.P)
        norm_mixed = calculate_norm_convergence(mixed_P, hf.P)

        if iteration == iteration_target
            snapshot_input_density = input_density
            snapshot_interaction = copy(interaction_h)
            snapshot_hamiltonian = copy(hf.H)
            snapshot_updated_density = copy(mixed_P)
            snapshot_lambda = λ
            snapshot_norm_raw = norm_raw
            snapshot_norm_mixed = norm_mixed
            snapshot_energy = energy
        end

        hf.P .= mixed_P
    end

    metadata = Pair{String, String}[
        "benchmark_id" => benchmark_id,
        "iteration" => string(iteration_target),
        "theta_deg" => @sprintf("%.2f", settings.theta_deg),
        "nu" => string(settings.nu),
        "init_mode" => settings.init_mode,
        "seed" => @sprintf("%03d", settings.seed),
        "lk" => string(settings.lk),
        "lg" => string(settings.lg),
        "source" => "julia_b0_iteration_snapshot",
        "initial_density_override" => string(isfile(init_override_path)),
    ]
    write_complex_stack(input_density_out, snapshot_input_density; metadata=metadata)
    write_complex_stack(interaction_out, snapshot_interaction; metadata=metadata)
    write_complex_stack(hamiltonian_out, snapshot_hamiltonian; metadata=metadata)
    write_complex_stack(updated_density_out, snapshot_updated_density; metadata=metadata)

    open(summary_out, "w") do io
        println(io, "benchmark_id=$(benchmark_id)")
        println(io, "iteration=$(iteration_target)")
        println(io, "theta_deg=$(settings.theta_deg)")
        println(io, "nu=$(settings.nu)")
        println(io, "init_mode=$(settings.init_mode)")
        println(io, "seed=$(settings.seed)")
        println(io, "lk=$(settings.lk)")
        println(io, "lg=$(settings.lg)")
        println(io, "bm_path=$(bm_path)")
        println(io, "initial_density_override=$(isfile(init_override_path))")
        println(io, "energy=$(snapshot_energy)")
        println(io, "mu=$(hf.μ)")
        println(io, "oda_lambda=$(snapshot_lambda)")
        println(io, "norm_raw=$(snapshot_norm_raw)")
        println(io, "norm_mixed=$(snapshot_norm_mixed)")
        println(io, "offdiag_flavor=$(offdiag_flavor_norm(hf))")
        println(io, "gap=$(restricted_gap_estimate(hf))")
        println(io, "occupied_sigma_mean=$(occupied_sigma_mean(hf))")
        println(io, "input_density_fro=$(norm(snapshot_input_density))")
        println(io, "updated_density_fro=$(norm(snapshot_updated_density))")
        println(io, "delta_density_fro=$(norm(snapshot_updated_density .- snapshot_input_density))")
        println(io, "interaction_fro=$(norm(snapshot_interaction))")
        println(io, "hamiltonian_fro=$(norm(snapshot_hamiltonian))")
        println(io, "lowest_eigs_k0=" * join([@sprintf("%.12f", x) for x in hf.ϵk[1:8, 1]], ","))
    end

    println("benchmark_id=$(benchmark_id)")
    println("iteration=$(iteration_target)")
    println("input_density_path=$(input_density_out)")
    println("interaction_path=$(interaction_out)")
    println("hamiltonian_path=$(hamiltonian_out)")
    println("updated_density_path=$(updated_density_out)")
    println("summary_path=$(summary_out)")
    println("initial_density_override=$(isfile(init_override_path))")
    println("mu=$(hf.μ)")
    println("oda_lambda=$(snapshot_lambda)")
    println("norm_raw=$(snapshot_norm_raw)")
    println("norm_mixed=$(snapshot_norm_mixed)")
end

main()
