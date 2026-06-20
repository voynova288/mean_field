#!/usr/bin/env julia

const IMPL_DIR = joinpath(@__DIR__, "_julia_impl")

function normalize_command(text::String)
    return replace(endswith(text, ".jl") ? text[1:end-3] : text, "-" => "_")
end

const COMMANDS = Dict(
    "export_b0_bm_grid_uk_reference_julia" => "export_b0_bm_grid_uk_reference_julia.jl",
    "export_b0_hf_first_iteration_reference_julia" => "export_b0_hf_first_iteration_reference_julia.jl",
    "export_b0_hf_initial_density_julia" => "export_b0_hf_initial_density_julia.jl",
    "export_b0_hf_iteration_snapshot_julia" => "export_b0_hf_iteration_snapshot_julia.jl",
    "export_b0_hf_state_density_julia" => "export_b0_hf_state_density_julia.jl",
    "export_b0_parameter_reference_from_julia" => "export_b0_parameter_reference_from_julia.jl",
)

function print_help()
    println("Usage: julia scripts/mean_field_tools.jl <command> [args...]")
    println()
    println("Commands:")
    for name in sort!(collect(keys(COMMANDS)))
        println("  ", name)
    end
end

if isempty(ARGS) || ARGS[1] in ("help", "--help", "-h") || normalize_command(ARGS[1]) == "help"
    print_help()
    exit(0)
end

command = normalize_command(ARGS[1])
haskey(COMMANDS, command) || error("Unknown command: $(ARGS[1])")

rest = copy(ARGS[2:end])
empty!(ARGS)
append!(ARGS, rest)
include(joinpath(IMPL_DIR, COMMANDS[command]))
