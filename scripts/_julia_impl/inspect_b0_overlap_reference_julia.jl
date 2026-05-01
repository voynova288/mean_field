using JLD2
using LinearAlgebra
using Printf

const TBG_REPO_ROOT = "/data/home/ziyuzhu/TBG_HartreeFock"
const MF_REPO_ROOT = "/data/home/ziyuzhu/Mean_Field"
const BENCHMARK_ROOT = joinpath(MF_REPO_ROOT, "benchmarks", "b0", "bm_inputs", "unstrained_path")

include(joinpath(TBG_REPO_ROOT, "B0", "libs", "BM_mod.jl"))

function build_params(theta_deg::Float64)
    params = Params(ϵ=0.0, Da=0.0, dθ=theta_deg * π / 180, w1=110.0, w0=77.0, vf=2482.0)
    initParamsWithStrain(params)
    return params
end

function build_kpath_from_nodes(nodes::Vector{ComplexF64}, labels::Vector{String}, points_per_segment::Int)
    @assert points_per_segment > 0
    length(nodes) >= 2 || error("At least two k-path nodes are required")

    kvec = ComplexF64[nodes[1]]
    kdist = Float64[0.0]
    node_indices = Int[1]

    for (start_k, end_k) in zip(nodes[1:end-1], nodes[2:end])
        dk = (end_k - start_k) / points_per_segment
        for istep in 1:points_per_segment
            push!(kvec, start_k + istep * dk)
            push!(kdist, kdist[end] + abs(dk))
        end
        push!(node_indices, length(kvec))
    end

    return kvec, kdist, labels, node_indices
end

function parse_reference_nodes(path::String)
    labels = String[]
    nodes = ComplexF64[]
    node_indices = Int[]
    open(path, "r") do io
        readline(io)
        for raw in eachline(io)
            fields = split(raw, '\t')
            length(fields) >= 5 || error("Malformed reference node row: $(raw)")
            push!(labels, fields[1])
            push!(node_indices, parse(Int, fields[2]))
            push!(nodes, parse(Float64, fields[4]) + 1im * parse(Float64, fields[5]))
        end
    end
    return labels, nodes, node_indices
end

function build_reference_kpath(reference_nodes_path::String)
    labels, nodes, node_indices = parse_reference_nodes(reference_nodes_path)
    length(node_indices) >= 2 || error("Reference node file must contain at least two nodes")
    segment_lengths = diff(node_indices)
    all(length == segment_lengths[1] for length in segment_lengths) || error("Reference node spacing is not uniform: $(segment_lengths)")
    points_per_segment = segment_lengths[1]
    return build_kpath_from_nodes(nodes, labels, points_per_segment)
end

function extract_compact_k_overlap(bm::HBM)
    tmpΛ = reshape(bm.Λ, bm.ns, bm.nη, bm.nb * bm.latt.nk, bm.ns, bm.nη, bm.nb * bm.latt.nk)
    return Array(tmpΛ[1, 1, :, 1, 1, :])
end

function summarize_overlap(overlap::AbstractMatrix{ComplexF64})
    mid = size(overlap, 1) ÷ 2 + 1
    return Dict(
        "fro_norm" => norm(overlap),
        "trace_real" => real(tr(overlap)),
        "trace_imag" => imag(tr(overlap)),
        "entry_11_real" => real(overlap[1, 1]),
        "entry_11_imag" => imag(overlap[1, 1]),
        "entry_mid_real" => real(overlap[mid, mid]),
        "entry_mid_imag" => imag(overlap[mid, mid]),
    )
end

function main()
    length(ARGS) == 4 || error("Usage: julia inspect_b0_overlap_reference_julia.jl <theta_deg> <lattice_kind> <m> <n>")
    theta_deg = parse(Float64, ARGS[1])
    lattice_kind = ARGS[2]
    m = parse(Int, ARGS[3])
    n = parse(Int, ARGS[4])
    lattice_kind == "path" || error("Only path overlap is supported in this diagnostic.")

    theta_code = round(Int, 100 * theta_deg)
    node_path = joinpath(BENCHMARK_ROOT, @sprintf("theta_%03d_unstrained_path_nodes.tsv", theta_code))
    kvec, _, _, _ = build_reference_kpath(node_path)

    params = build_params(theta_deg)
    latt = initLatticeWithKvec(kvec)
    bm = HBM()
    scratch = tempname() * ".jld2"
    initHBM(bm, latt, params; lg=9, _σrotation=true, _calculate_overlap=false, fname=scratch)
    bm.Λ = zeros(ComplexF64, bm.nt * bm.latt.nk, bm.nt * bm.latt.nk)
    calculate_overlap(bm, m, n)

    diag = summarize_overlap(extract_compact_k_overlap(bm))
    println(
        @sprintf(
            "julia\ttheta=%.2f\tlattice=%s\tG=(%d,%d)\tfro_norm=%.16e\ttrace_real=%.16e\ttrace_imag=%.16e\tentry_11=(%.16e,%.16e)\tentry_mid=(%.16e,%.16e)",
            theta_deg,
            lattice_kind,
            m,
            n,
            diag["fro_norm"],
            diag["trace_real"],
            diag["trace_imag"],
            diag["entry_11_real"],
            diag["entry_11_imag"],
            diag["entry_mid_real"],
            diag["entry_mid_imag"],
        ),
    )
end

main()
