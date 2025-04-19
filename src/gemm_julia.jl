using LinearAlgebra, Random

const SCRUB_SIZE = 64_000_000  # 64 million floats ≈ 256 MB
const REP = 3  # or any number of repetitions

function cache_scrub()
    scrub = zeros(Float32, SCRUB_SIZE)
    # ensure it's touched and not optimized away
    s = sum(scrub)
    return s
end

function main(m::Int, k::Int, n::Int)
    A = rand(Float32, m, k)
    B = rand(Float32, k, n)
    C = zeros(Float32, m, n)

    for it in 1:REP
        fill!(C, 0.0f0)

        cache_scrub()

        BLAS.gemm!('N', 'N', 1.0f0, A, B, 0.0f0, C)
    end

    return C
end

# Run from command line
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 3
        println("Usage: julia gemm.jl m k n")
        exit(1)
    end
    m, k, n = parse.(Int, ARGS)
    main(m, k, n)
end

