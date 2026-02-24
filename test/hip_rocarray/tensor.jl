@testset "hipTensor" begin

if !AMDGPU.functional(:hiptensor)
    @test_skip "hipTensor not available"
else

@testset "Version" begin
    v = AMDGPU.hipTensor.version()
    @test v isa Integer
    @test v > 0
end

@testset "Handle" begin
    h = AMDGPU.hipTensor.handle()
    @test h != C_NULL
end

@testset "Tensor Contraction" begin
    @testset "Matrix multiply via contraction $T" for T in (Float32, Float64)
        m, n, k = 4, 4, 4
        A = ROCArray(rand(T, m, k))
        B = ROCArray(rand(T, k, n))
        C = ROCArray(zeros(T, m, n))

        AMDGPU.hipTensor.contract!(
            one(T), A, [1, 2], B, [2, 3],
            zero(T), C, [1, 3])
        AMDGPU.synchronize()

        A_h = Array(A)
        B_h = Array(B)
        C_h = Array(C)
        @test C_h ≈ A_h * B_h atol=1e-3
    end

    @testset "High-level contract" begin
        T = Float32
        m, n, k = 8, 6, 4
        A = ROCArray(rand(T, m, k))
        B = ROCArray(rand(T, k, n))

        C = AMDGPU.hipTensor.contract(A, [1, 2], B, [2, 3], [1, 3])
        AMDGPU.synchronize()

        @test size(C) == (m, n)
        @test Array(C) ≈ Array(A) * Array(B) atol=1e-3
    end
end

@testset "Tensor Permutation" begin
    @testset "Permute 3D $T" for T in (Float32, Float64)
        A = ROCArray(rand(T, 3, 4, 5))
        B = ROCArray(zeros(T, 5, 3, 4))

        AMDGPU.hipTensor.permute!(
            one(T), A, [1, 2, 3], B, [3, 1, 2])
        AMDGPU.synchronize()

        @test Array(B) ≈ permutedims(Array(A), (3, 1, 2))
    end

    @testset "Permute 2D (transpose) $T" for T in (Float32,)
        A = ROCArray(rand(T, 6, 8))
        B = ROCArray(zeros(T, 8, 6))

        AMDGPU.hipTensor.permute!(
            one(T), A, [1, 2], B, [2, 1])
        AMDGPU.synchronize()

        @test Array(B) ≈ transpose(Array(A))
    end
end

@testset "Tensor Reduction" begin
    @testset "Sum reduction $T" for T in (Float32, Float64)
        A = ROCArray(rand(T, 4, 5, 6))
        C = ROCArray(zeros(T, 4, 6))

        AMDGPU.hipTensor.reduce!(
            one(T), A, [1, 2, 3],
            zero(T), C, [1, 3];
            opReduce=AMDGPU.hipTensor.HIPTENSOR_OP_ADD)
        AMDGPU.synchronize()

        expected = dropdims(sum(Array(A); dims=2); dims=2)
        @test Array(C) ≈ expected atol=1e-2
    end
end

@testset "Elementwise Binary" begin
    @testset "Add $T" for T in (Float32,)
        A = ROCArray(rand(T, 4, 5))
        C = ROCArray(rand(T, 4, 5))
        C_orig = copy(C)

        AMDGPU.hipTensor.elementwise_binary!(
            one(T), A, [1, 2],
            one(T), C, [1, 2];
            opAC=AMDGPU.hipTensor.HIPTENSOR_OP_ADD)
        AMDGPU.synchronize()

        @test Array(C) ≈ Array(A) .+ Array(C_orig) atol=1e-5
    end
end

@testset "Elementwise Trinary" begin
    @testset "Add three tensors $T" for T in (Float32,)
        A = ROCArray(rand(T, 4, 5))
        B = ROCArray(rand(T, 4, 5))
        C = ROCArray(rand(T, 4, 5))
        C_orig = copy(C)

        AMDGPU.hipTensor.elementwise_trinary!(
            one(T), A, [1, 2],
            one(T), B, [1, 2],
            one(T), C, [1, 2];
            opAB=AMDGPU.hipTensor.HIPTENSOR_OP_ADD,
            opABC=AMDGPU.hipTensor.HIPTENSOR_OP_ADD)
        AMDGPU.synchronize()

        @test Array(C) ≈ Array(A) .+ Array(B) .+ Array(C_orig) atol=1e-5
    end
end

end

end
