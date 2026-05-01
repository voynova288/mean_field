using JLD2
using Printf

function write_density(path::String, density::Array{ComplexF64, 3}; metadata::Vector{Pair{String, String}}=Pair{String, String}[])
    mkpath(dirname(path))
    open(path, "w") do io
        for (key, value) in metadata
            println(io, "# $(key)=$(value)")
        end
        println(io, "# nrow=$(size(density, 1))")
        println(io, "# ncol=$(size(density, 2))")
        println(io, "# nk=$(size(density, 3))")
        for ik in 1:size(density, 3), row in 1:size(density, 1), col in 1:size(density, 2)
            value = density[row, col, ik]
            println(io, @sprintf("%d\t%d\t%d\t%.17e\t%.17e", ik - 1, row - 1, col - 1, real(value), imag(value)))
        end
    end
    return path
end

function jld2_scalar_string(file, key::String)
    haskey(file, key) || return ""
    return string(file[key])
end

function main()
    length(ARGS) >= 2 || error("Usage: julia scripts/mean_field_tools.jl export_b0_hf_state_density_julia <state.jld2> <output.tsv>")
    state_path = abspath(ARGS[1])
    output_path = abspath(ARGS[2])
    isfile(state_path) || error("Missing HF state JLD2: $(state_path)")

    jldopen(state_path, "r") do file
        hf = file["hf"]
        density = Array{ComplexF64, 3}(hf.P)
        metadata = Pair{String, String}[
            "source" => "julia_b0_hf_state_density",
            "state_path" => state_path,
            "theta_deg" => jld2_scalar_string(file, "theta_deg"),
            "nu" => jld2_scalar_string(file, "nu"),
            "lk" => jld2_scalar_string(file, "lk"),
            "lg" => jld2_scalar_string(file, "lg"),
            "init_mode" => jld2_scalar_string(file, "init_mode"),
            "seed" => jld2_scalar_string(file, "seed"),
            "converged" => jld2_scalar_string(file, "converged"),
            "exit_reason" => jld2_scalar_string(file, "exit_reason"),
            "mu" => string(hf.μ),
        ]
        write_density(output_path, density; metadata=metadata)
        lk_text = jld2_scalar_string(file, "lk")
        exit_reason_text = jld2_scalar_string(file, "exit_reason")
        converged_text = jld2_scalar_string(file, "converged")
        println("state_path=$(state_path)")
        println("output_path=$(output_path)")
        println("nt=$(size(density, 1))")
        println("nk=$(size(density, 3))")
        println("lk=$(lk_text)")
        println("exit_reason=$(exit_reason_text)")
        println("converged=$(converged_text)")
        println("mu=$(hf.μ)")
    end
end

main()
