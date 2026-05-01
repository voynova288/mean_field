using JLD2
using LinearAlgebra
using Printf
using Random

const TBG_REPO_ROOT = "/data/home/ziyuzhu/TBG_HartreeFock"
const MF_REPO_ROOT = "/data/home/ziyuzhu/Mean_Field"
include(joinpath(TBG_REPO_ROOT, "B0", "libs", "HF_mod.jl"))

function load_initial_density_override!(hf::HartreeFock, path::String)
    fill!(hf.P, 0.0 + 0.0im)
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        fields = split(stripped, '\t')
        ik = parse(Int, fields[1]) + 1
        row = parse(Int, fields[2]) + 1
        col = parse(Int, fields[3]) + 1
        real_part = parse(Float64, fields[4])
        imag_part = parse(Float64, fields[5])
        hf.P[row, col, ik] = complex(real_part, imag_part)
    end
    return nothing
end

function build_params(theta_deg::Float64)
    params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
    initParamsWithStrain(params)
    return params
end

function main()
    benchmark_id = length(ARGS) >= 1 ? ARGS[1] : "theta_120_nu_-2_ivc_ground"
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

    Gs = load(hf.fname, "Gs")
    lG = load(hf.fname, "lG")
    Glabels = (-(lG - 1) ÷ 2):((lG - 1) ÷ 2)
    Lm = sqrt(abs(hf.params.a1) * abs(hf.params.a2))
    tmp_Λ = reshape(hf.Λ, hf.nt, hf.latt.nk, hf.nt, hf.latt.nk)
    kvec = reshape(hf.latt.kvec, :)

    println("m\tn\ting_shell\thartree_coeff\ttr_pg_real\ttr_pg_imag\ttr_pg_abs\tfock_fro\tfock_max_abs")
    for ig in 1:(lG^2)
        m, n = Glabels[(ig - 1) % lG + 1], Glabels[(ig - 1) ÷ lG + 1]
        G = Gs[ig]
        jldopen(hf.fname, "r") do file
            hf.Λ .= file["$(m)_$(n)"]
        end

        G0 = abs(3 * hf.params.g1 + 3 * hf.params.g2) * 1.00001
        in_shell = abs(G) < G0 * cos(pi / 6) / abs(cos(mod(angle(G), pi / 3) - pi / 6))

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

        println(
            @sprintf(
                "%d\t%d\t%s\t%.16e\t%.16e\t%.16e\t%.16e\t%.16e\t%.16e",
                m,
                n,
                string(in_shell),
                hf.V0 * V(G, Lm) / hf.latt.nk,
                real(trPG),
                imag(trPG),
                abs(trPG),
                norm(fock),
                maximum(abs.(fock)),
            ),
        )
    end
end

main()
