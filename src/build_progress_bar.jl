module BuildProgressBar

using ProgressMeter


struct CMakeMakefileStyle end
struct CMakeNinjaStyle end

# Capture '[ %42]' for Makefiles, '[5/23]' for Ninja
const PROGRESS_STYLES = Dict(
    CMakeMakefileStyle() => r"^\[\s*(\d+)%\]",
    CMakeNinjaStyle() => r"^\[(\d+)/(\d+)\]"
)


function extract_progress(match, ::CMakeMakefileStyle)
    current_progress = parse(Int, match[1])
    total_jobs = 100  # Output a percentage
    return total_jobs, current_progress
end


function extract_progress(match, ::CMakeNinjaStyle)
    current_progress = parse(Int, match[1])
    total_jobs = parse(Int, match[2])
    return total_jobs, current_progress
end


function next_line_end_pos(str, i)
    i > length(str) && return length(str)
    pos = findnext(str, "\n", i)
    pos === nothing && return length(str)
    return first(pos) - 1
end


# Taken from the ProgressMeter.jl docs
is_logging(io) = isinteractive() || isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")


function progress_callback(progress_bar, total_jobs, current_job)
    total_jobs == 0 && return nothing
    if progress_bar === nothing
        barlen = min(ProgressMeter.tty_width("Build: ", stderr, false), 50)
        progress_bar = Progress(total_jobs; desc="Build: ", barlen, enabled=is_logging(stderr))
    end
    if current_job == total_jobs
        finish!(progress_bar)
    else
        update!(progress_bar, current_job)
    end
    return progress_bar
end


function track_build_progress(build_cmd; stdout_pipe=nothing)
    build_output = IOBuffer()
    build_process = run(pipeline(build_cmd, stdout=build_output); wait=false)

    format_style = nothing

    total_jobs = 0
    current_progress = 0
    progress_state = progress_callback(nothing, total_jobs, current_progress)

    while process_running(build_process)
        sleep(0.05)
        lines = String(take!(build_output))
        isempty(lines) && continue

        stdout_pipe !== nothing && print(stdout_pipe, lines)

        line_start = 1
        line_end = next_line_end_pos(lines, 1)
        while line_start <= line_end
            line = SubString(lines, line_start, line_end)

            if startswith(line, '[')
                if format_style === nothing
                    for (style, regex) in PROGRESS_STYLES
                        m = match(regex, line)
                        if m !== nothing
                            format_style = style
                            break
                        end
                    end
                else
                    m = match(PROGRESS_STYLES[format_style], line)
                end

                m === nothing && continue

                total_jobs, current_progress = extract_progress(m, format_style)
                progress_state = progress_callback(progress_state, total_jobs, current_progress)
            end

            line_start = line_end + 1
            line_end = next_line_end_pos(lines, line_start)
        end
    end

    progress_callback(progress_state, total_jobs, total_jobs)

    if !success(build_process)
        error("failed process: ", build_process)
    end
end

end