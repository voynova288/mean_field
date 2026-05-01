using JLD2
using LinearAlgebra
using Printf
using Random

const TBG_REPO_ROOT = "/data/home/ziyuzhu/TBG_HartreeFock"
include(joinpath(TBG_REPO_ROOT, "B0", "libs", "HF_mod.jl"))

function build_params(theta_deg::Float64)
    params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
    initParamsWithStrain(params)
    return params
end

function matrix_norms(stack::Array{ComplexF64,3})
    offdiag = copy(stack)
    for ik in 1:size(stack, 3)
        offdiag[:, :, ik] .-= Diagonal(diag(view(stack, :, :, ik)))
    end
    return (
        fro_norm = norm(stack),
        offdiag_total_norm = norm(offdiag),
        max_abs = maximum(abs.(stack)),
        max_abs_offdiag = maximum(abs.(offdiag)),
    )
end

function print_matrix_norms(label::String, stack::Array{ComplexF64,3})
    norms = matrix_norms(stack)
    println(
        @sprintf(
            "%s: fro=%.6e offdiag_total=%.6e max_abs=%.6e max_abs_offdiag=%.6e",
            label,
            norms.fro_norm,
            norms.offdiag_total_norm,
            norms.max_abs,
            norms.max_abs_offdiag,
        ),
    )
end

function offdiag_flavor_norm(hf::HartreeFock)
    sectors = flavor_block_indices(hf)
    mask = falses(hf.nt, hf.nt)
    for inds in sectors
        mask[inds, inds] .= true
    end
    total = 0.0
    for ik in 1:hf.latt.nk
        block = copy(view(hf.P, :, :, ik))
        block[mask] .= 0.0 + 0.0im
        total += sum(abs2, block)
    end
    return sqrt(total)
end

function hex_shell_contains(params::Params, G::ComplexF64)
    G0 = abs(3 * params.g1 + 3 * params.g2) * 1.00001
    return abs(G) < G0 * cos(pi / 6) / abs(cos(mod(angle(G), pi / 3) - pi / 6))
end

function interaction_for_shift!(hf::HartreeFock, m::Int, n::Int)
    Gs = load(hf.fname, "Gs")
    lG = load(hf.fname, "lG")
    Glabels = (-(lG - 1) ÷ 2):((lG - 1) ÷ 2)
    ig = findfirst(i -> Glabels[(i - 1) % lG + 1] == m && Glabels[(i - 1) ÷ lG + 1] == n, 1:(lG^2))
    ig === nothing && error("Shift ($(m),$(n)) not found")

    jldopen(hf.fname, "r") do file
        hf.Λ .= file["$(m)_$(n)"]
    end
    tmp_Λ = reshape(hf.Λ, hf.nt, hf.latt.nk, hf.nt, hf.latt.nk)
    Lm = sqrt(abs(hf.params.a1) * abs(hf.params.a2))
    kvec = reshape(hf.latt.kvec, :)
    G = Gs[ig]

    trPG = 0.0 + 0.0im
    for ik in 1:hf.latt.nk
        trPG += tr(view(hf.P, :, :, ik) * conj(view(tmp_Λ, :, ik, :, ik)))
    end

    fock = zeros(ComplexF64, hf.nt, hf.nt, hf.latt.nk)
    for ik in 1:hf.latt.nk
        tmp_Fock = zeros(ComplexF64, hf.nt, hf.nt)
        for ip in 1:hf.latt.nk
            coeff = hf.V0 * V(kvec[ip] - kvec[ik] + G, Lm) / hf.latt.nk
            tmp_Fock .+= coeff .* (view(tmp_Λ, :, ik, :, ip) * transpose(view(hf.P, :, :, ip)) * view(tmp_Λ, :, ik, :, ip)')
        end
        fock[:, :, ik] .= tmp_Fock
    end

    return (
        G = G,
        overlap_fro = norm(hf.Λ),
        overlap_max_abs = maximum(abs.(hf.Λ)),
        hartree_coeff = hf.V0 * V(G, Lm) / hf.latt.nk,
        tr_pg_abs = abs(trPG),
        fock_fro = norm(fock),
        fock_max_abs = maximum(abs.(fock)),
    )
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
    theta_deg = 1.20
    ν = -2.0
    init_mode = "sp"
    seed = 1
    lk = 19
    lg = 9
    bm_path = "/data/home/ziyuzhu/Mean_Field/benchmarks/b0/bm_inputs/bm_theta_120_lk19_lg9.jld2"
    init_override_path = "/data/home/ziyuzhu/Mean_Field/benchmarks/b0/cases/theta_120_nu_-2_ivc_ground/initial_density_sp_seed_001.tsv"

    params = build_params(theta_deg)
    latt = Lattice()
    initLattice(latt, params; lk=lk)

    Random.seed!(seed)
    hf = HartreeFock()
    hf.params = params
    hf.latt = latt
    hf.ν = ν
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
    init_P(hf; _Init=init_mode)
    if isfile(init_override_path)
        load_initial_density_override!(hf, init_override_path)
    end
    hf.Λ = zeros(ComplexF64, hf.nt * latt.nk, hf.nt * latt.nk)
    ηs = ["η0", "η1", "η2", "η3"]
    σs = ["s0", "s1", "s2", "s3"]
    ns = ["n0", "n1", "n2", "n3"]
    hf.Δstr = [ns[i] * ηs[j] * σs[k] for i in 1:4 for j in 1:4 for k in 1:4]
    hf.Δ = zeros(Float64, size(hf.Δstr))

    println("theta_deg=$(theta_deg)")
    println("nu=$(ν)")
    println("init_mode=$(init_mode)")
    println("seed=$(seed)")
    println("lk=$(lk)")
    println("lg=$(lg)")
    println("nk=$(hf.latt.nk)")
    println("initial_density_override=$(isfile(init_override_path))")
    println(@sprintf("initial_offdiag_flavor_norm=%.6e", offdiag_flavor_norm(hf)))
    print_matrix_norms("density_initial", hf.P)

    for shift in ((0, 0), (1, 0), (0, 1))
        info = interaction_for_shift!(hf, shift[1], shift[2])
        println(
            @sprintf(
                "shift=(%d, %d) g_abs=%.6e in_shell=%s overlap_fro=%.6e overlap_max_abs=%.6e hartree_coeff=%.6e tr_pg_abs=%.6e fock_fro=%.6e fock_max_abs=%.6e",
                shift[1],
                shift[2],
                abs(info.G),
                string(hex_shell_contains(hf.params, info.G)),
                info.overlap_fro,
                info.overlap_max_abs,
                info.hartree_coeff,
                info.tr_pg_abs,
                info.fock_fro,
                info.fock_max_abs,
            ),
        )
    end

    previous_P = copy(hf.P)
    hf.H .= hf.H0
    add_HartreeFock(hf; β=1.0)
    print_matrix_norms("interaction_h", hf.H .- hf.H0)
    print_matrix_norms("hamiltonian_total", hf.H)

    norm_convergence, λ = update_P(hf; Δ=0.0)
    println(@sprintf("updated_mu=%.12f", hf.μ))
    println(@sprintf("updated_offdiag_flavor_norm=%.6e", offdiag_flavor_norm(hf)))
    print_matrix_norms("density_new", hf.P)
    print_matrix_norms("delta_density", hf.P .- previous_P)
    println(@sprintf("delta_density_fro=%.6e", norm(hf.P .- previous_P)))
    println(@sprintf("delta_density_max_abs=%.6e", maximum(abs.(hf.P .- previous_P))))
    println(@sprintf("oda_lambda=%.12f", λ))
    println(@sprintf("norm_convergence=%.6e", norm_convergence))
    println(@sprintf("occupied_sigma_mean=%.6e", mean(hf.σzτz[hf.ϵk .<= hf.μ])))
    println("lowest_eigs_k0=" * join([@sprintf("%.12f", x) for x in hf.ϵk[1:8, 1]], ","))
end

main()
