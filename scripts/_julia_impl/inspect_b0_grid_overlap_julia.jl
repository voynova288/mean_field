using JLD2
using LinearAlgebra
using Printf

function summarize(name::String, matrix::AbstractMatrix{ComplexF64})
    mid = size(matrix, 1) ÷ 2 + 1
    return @sprintf(
        "%s\tfro_norm=%.16e\tmax_abs=%.16e\ttrace_real=%.16e\ttrace_imag=%.16e\tentry_11=(%.16e,%.16e)\tentry_mid=(%.16e,%.16e)",
        name,
        norm(matrix),
        maximum(abs.(matrix)),
        real(tr(matrix)),
        imag(tr(matrix)),
        real(matrix[1, 1]),
        imag(matrix[1, 1]),
        real(matrix[mid, mid]),
        imag(matrix[mid, mid]),
    )
end

function main()
    path = length(ARGS) >= 1 ? ARGS[1] : "/data/home/ziyuzhu/Mean_Field/benchmarks/b0/bm_inputs/bm_theta_120_lk19_lg9.jld2"
    jldopen(path, "r") do file
        ns = Int(file["ns"])
        nη = Int(file["nη"])
        nb = Int(file["nb"])
        println(@sprintf("julia_grid\tpath=%s\tns=%d\tneta=%d\tnb=%d", path, ns, nη, nb))
        for shift in ("0_0", "1_0", "0_1")
            full = Array{ComplexF64}(file[shift])
            nk = size(full, 1) ÷ (ns * nη * nb)
            compact = Array(reshape(full, ns, nη, nb * nk, ns, nη, nb * nk)[1, 1, :, 1, 1, :])
            println(summarize("julia_full\tG=$(shift)", full))
            println(summarize("julia_compact\tG=$(shift)", compact))
        end
    end
end

main()
