using Test
using AMDGPU
using AMDGPU: ROCArray

@testset "Eager GC" begin
    # Each iteration allocates large temporary GPU arrays that go out of scope.
    # Julia GC does not know about GPU memory pressure because the ROCArray
    # wrappers are tiny. With EAGER_GC enabled, maybe_collect() triggers GC
    # before each allocation when GPU pressure is high, keeping memory bounded.

    AMDGPU.synchronize()
    GC.gc(true)
    sleep(0.1)

    free_gpu = AMDGPU.free()

    # Each array is 0.25% of free memory, 3 per iteration.
    # Iteration count is chosen so total allocations reach 2x free memory.
    alloc_frac = 0.0025
    bytes_per_iter = 3 * free_gpu * alloc_frac
    iters = max(50, ceil(Int, 2 * free_gpu / bytes_per_iter))

    elem_bytes = sizeof(Float32)
    n = round(Int, sqrt(free_gpu * alloc_frac / elem_bytes))

    @testset "enabled" begin
        AMDGPU.EAGER_GC[] = true
        GC.gc(true)
        AMDGPU.synchronize()
        AMDGPU.reclaim()
        sleep(0.5)

        baseline = AMDGPU.used_memory()
        peak = baseline

        for i in 1:iters
            A = AMDGPU.rand(Float32, n, n)
            B = AMDGPU.rand(Float32, n, n)
            C = A .+ B
            AMDGPU.synchronize()
            peak = max(peak, AMDGPU.used_memory())
        end

        net_peak = peak - baseline

        @info "Eager GC (n=$n, iters=$iters):" *
            "\n  Array size:  $(Base.format_bytes(n * n * elem_bytes))" *
            "\n  Peak used:   $(Base.format_bytes(net_peak))" *
            "\n  Free GPU:    $(Base.format_bytes(free_gpu))"

        @test true
        # Peak memory stays well below free memory despite allocating 2x total.
        @test net_peak < 0.75 * free_gpu
    end

    GC.gc(true)
    AMDGPU.synchronize()
    AMDGPU.reclaim()
    sleep(1.0)

    @testset "disabled" begin
        AMDGPU.EAGER_GC[] = false
        GC.gc(true)
        AMDGPU.synchronize()
        AMDGPU.reclaim()
        sleep(0.5)

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
            catch e
                @info "EAGER_GC=false hit OOM at iteration $i/$iters"
                break
            end
        end

        net_peak = peak - baseline

        @info "No eager GC (n=$n, iters=$iters):" *
            "\n  Array size:  $(Base.format_bytes(n * n * elem_bytes))" *
            "\n  Peak used:   $(Base.format_bytes(net_peak))" *
            "\n  Completed:   $completed/$iters"

        # Without eager GC, memory grows until OOM or the last-resort
        # GC in alloc_or_retry kicks in at much higher pressure.
        @test net_peak > 0.75 * free_gpu || completed < iters
    end

    AMDGPU.EAGER_GC[] = true
    GC.gc(true)
    AMDGPU.synchronize()
    AMDGPU.reclaim()
end
