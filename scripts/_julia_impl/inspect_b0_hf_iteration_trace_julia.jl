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

function restricted_filling(hf::HartreeFock)
    trace_sum = 0.0
    for ik in 1:size(hf.P, 3)
        trace_sum += real(tr(view(hf.P, :, :, ik)))
    end
    return trace_sum / size(hf.P, 3)
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

function build_projector_update(hf::HartreeFock)
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
    max_iter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    settings = parse_runtime_summary(benchmark_id)

    theta_tag = replace(@sprintf("%.2f", settings.theta_deg), "." => "")
    bm_path = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "bm_inputs", "bm_theta_$(theta_tag)_lk$(settings.lk)_lg$(settings.lg).jld2")
    init_override_path = joinpath(
        MF_REPO_ROOT,
        "benchmarks",
        "b0",
        "cases",
        benchmark_id,
        @sprintf("initial_density_%s_seed_%03d.tsv", settings.init_mode, settings.seed),
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

    println("benchmark_id=$(benchmark_id)")
    println("mode=full")
    println("theta_deg=$(settings.theta_deg)")
    println("nu=$(settings.nu)")
    println("init_mode=$(settings.init_mode)")
    println("seed=$(settings.seed)")
    println("lk=$(settings.lk)")
    println("lg=$(settings.lg)")
    println("nk=$(hf.latt.nk)")
    println("initial_density_override=$(isfile(init_override_path))")
    println(@sprintf("iter=0 filling=%.12f offdiag_flavor=%.12e", restricted_filling(hf), offdiag_flavor_norm(hf)))

    for iteration in 1:max_iter
        previous_P = copy(hf.P)
        hf.H .= hf.H0
        add_HartreeFock(hf; β=1.0)
        energy = compute_HF_energy(hf.H .- hf.H0, hf.H0, hf.P)
        P_new = build_projector_update(hf)
        λ = oda_parametrization(hf, P_new .- hf.P; β=1.0)
        mixed_P = λ .* P_new .+ (1 - λ) .* hf.P
        norm_raw = calculate_norm_convergence(P_new, hf.P)
        norm_mixed = calculate_norm_convergence(mixed_P, hf.P)
        hf.P .= mixed_P

        println(
            @sprintf(
                "iter=%d energy=%.12f mu=%.12f lambda=%.12f norm_raw=%.12e norm_mixed=%.12e offdiag_flavor=%.12e gap=%.12f occupied_sigma_mean=%.12f filling=%.12f",
                iteration,
                energy,
                hf.μ,
                λ,
                norm_raw,
                norm_mixed,
                offdiag_flavor_norm(hf),
                restricted_gap_estimate(hf),
                occupied_sigma_mean(hf),
                restricted_filling(hf),
            ),
        )
    end
end

main()
