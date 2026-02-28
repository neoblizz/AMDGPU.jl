using Test
using AMDGPU
using AMDGPU: ROCArray

@testset "Eager GC" begin
    AMDGPU.synchronize()
    GC.gc(true)
    sleep(0.1)

    free_gpu = AMDGPU.free()

    alloc_frac = 0.0025
    bytes_per_iter = 3 * free_gpu * alloc_frac
    iters = max(50, ceil(Int, 2 * free_gpu / bytes_per_iter))

    elem_bytes = sizeof(Float32)
    n = round(Int, sqrt(free_gpu * alloc_frac / elem_bytes))

    function stress_loop(iters, n)
        baseline = AMDGPU.used_memory()
        peak = baseline
        completed = 0
        for i in 1:iters
            try
                A = AMDGPU.rand(Float32, n, n)
                B = AMDGPU.rand(Float32, n, n)
                C = A .+ B
                AMDGPU.synchronize()
                peak = max(peak, AMDGPU.used_memory())
                completed += 1
            catch
                break
            end
        end
        return peak - baseline, completed
    end

    # --- Run with eager GC enabled (should keep memory bounded) ---

    AMDGPU.EAGER_GC[] = true
    GC.gc(true); AMDGPU.synchronize(); AMDGPU.reclaim(); sleep(0.5)

    eager_peak, eager_completed = stress_loop(iters, n)

    @info "Eager GC (n=$n, iters=$iters):" *
        "\n  Array size:  $(Base.format_bytes(n * n * elem_bytes))" *
        "\n  Peak used:   $(Base.format_bytes(eager_peak))" *
        "\n  Completed:   $eager_completed/$iters" *
        "\n  Free GPU:    $(Base.format_bytes(free_gpu))"

    @test eager_completed == iters

    # --- Run with eager GC disabled (memory should grow much higher) ---

    GC.gc(true); AMDGPU.synchronize(); AMDGPU.reclaim(); sleep(1.0)

    AMDGPU.EAGER_GC[] = false
    GC.gc(true); AMDGPU.synchronize(); AMDGPU.reclaim(); sleep(0.5)

    nogc_peak, nogc_completed = stress_loop(iters, n)

    @info "No eager GC (n=$n, iters=$iters):" *
        "\n  Array size:  $(Base.format_bytes(n * n * elem_bytes))" *
        "\n  Peak used:   $(Base.format_bytes(nogc_peak))" *
        "\n  Completed:   $nogc_completed/$iters"

    # Eager GC must use strictly less peak memory, or disabled must OOM first.
    @test eager_peak < nogc_peak || nogc_completed < iters

    AMDGPU.EAGER_GC[] = true
    GC.gc(true); AMDGPU.synchronize(); AMDGPU.reclaim()
end
