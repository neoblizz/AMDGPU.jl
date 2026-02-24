function _default_compute_type(::Type{T}) where T
    if T <: Union{Float16, BFloat16}
        return Float32
    elseif T <: ComplexF16
        return ComplexF32
    else
        return T
    end
end

function _scalar_ref(alpha, T)
    Ref{T}(T(alpha))
end

"""
    contract!(alpha, A, modesA, B, modesB, beta, C, modesC;
              opA=HIPTENSOR_OP_IDENTITY, opB=HIPTENSOR_OP_IDENTITY, opC=HIPTENSOR_OP_IDENTITY,
              compute_type=nothing, algo=HIPTENSOR_ALGO_DEFAULT,
              jit=HIPTENSOR_JIT_MODE_NONE)

Compute the tensor contraction `D = alpha * opA(A) * opB(B) + beta * opC(C)` where
C and D share the same descriptor and modes. The result is written into C.

Modes are specified as vectors of integers (labels for each dimension).
"""
function contract!(
    alpha::Number, A::ROCArray, modesA::Vector{<:Integer},
    B::ROCArray, modesB::Vector{<:Integer},
    beta::Number, C::ROCArray, modesC::Vector{<:Integer};
    opA::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opB::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opC::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    compute_type::Union{DataType, Nothing} = nothing,
    algo::hiptensorAlgo_t = HIPTENSOR_ALGO_DEFAULT,
    jit::hiptensorJitMode_t = HIPTENSOR_JIT_MODE_NONE,
)
    CT = something(compute_type, _default_compute_type(eltype(C)))
    h = handle()
    s = stream()

    descA = HIPTensorDescriptor(h, A)
    descB = HIPTensorDescriptor(h, B)
    descC = HIPTensorDescriptor(h, C)

    mA = Int32.(modesA)
    mB = Int32.(modesB)
    mC = Int32.(modesC)

    op_ref = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreateContraction(h, op_ref, descA.ptr, mA, opA,
                               descB.ptr, mB, opB,
                               descC.ptr, mC, opC,
                               descC.ptr, mC,
                               hiptensor_computetype(CT))

    pref_ref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(h, pref_ref, algo, jit)

    plan = HIPTensorPlan(h, op_ref[], pref_ref[])

    a_ref = _scalar_ref(alpha, CT)
    b_ref = _scalar_ref(beta, CT)

    hiptensorContract(h, plan.ptr, a_ref, A, B, b_ref, C, C,
                      plan.workspace, UInt64(length(plan.workspace)),
                      s.stream)

    hiptensorDestroyOperationDescriptor(op_ref[])
    hiptensorDestroyPlanPreference(pref_ref[])
    return C
end

"""
    permute!(alpha, A, modesA, B, modesB;
             opA=HIPTENSOR_OP_IDENTITY, compute_type=nothing)

Compute the tensor permutation `B = alpha * opA(A)`.
"""
function permute!(
    alpha::Number, A::ROCArray, modesA::Vector{<:Integer},
    B::ROCArray, modesB::Vector{<:Integer};
    opA::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    compute_type::Union{DataType, Nothing} = nothing,
)
    CT = something(compute_type, _default_compute_type(eltype(A)))
    h = handle()
    s = stream()

    descA = HIPTensorDescriptor(h, A)
    descB = HIPTensorDescriptor(h, B)

    mA = Int32.(modesA)
    mB = Int32.(modesB)

    op_ref = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreatePermutation(h, op_ref, descA.ptr, mA, opA,
                               descB.ptr, mB,
                               hiptensor_computetype(CT))

    pref_ref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(h, pref_ref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE)

    plan = HIPTensorPlan(h, op_ref[], pref_ref[])

    a_ref = _scalar_ref(alpha, CT)

    hiptensorPermute(h, plan.ptr, a_ref, A, B, s.stream)

    hiptensorDestroyOperationDescriptor(op_ref[])
    hiptensorDestroyPlanPreference(pref_ref[])
    return B
end

"""
    reduce!(alpha, A, modesA, beta, C, modesC;
            opA=HIPTENSOR_OP_IDENTITY, opC=HIPTENSOR_OP_IDENTITY,
            opReduce=HIPTENSOR_OP_ADD, compute_type=nothing)

Compute the tensor reduction `D = alpha * opReduce(opA(A)) + beta * opC(C)`.
The result is written into C.
"""
function reduce!(
    alpha::Number, A::ROCArray, modesA::Vector{<:Integer},
    beta::Number, C::ROCArray, modesC::Vector{<:Integer};
    opA::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opC::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opReduce::hiptensorOperator_t = HIPTENSOR_OP_ADD,
    compute_type::Union{DataType, Nothing} = nothing,
    algo::hiptensorAlgo_t = HIPTENSOR_ALGO_DEFAULT,
    jit::hiptensorJitMode_t = HIPTENSOR_JIT_MODE_NONE,
)
    CT = something(compute_type, _default_compute_type(eltype(C)))
    h = handle()
    s = stream()

    descA = HIPTensorDescriptor(h, A)
    descC = HIPTensorDescriptor(h, C)

    mA = Int32.(modesA)
    mC = Int32.(modesC)

    op_ref = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreateReduction(h, op_ref, descA.ptr, mA, opA,
                             descC.ptr, mC, opC,
                             descC.ptr, mC, opReduce,
                             hiptensor_computetype(CT))

    pref_ref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(h, pref_ref, algo, jit)

    plan = HIPTensorPlan(h, op_ref[], pref_ref[])

    a_ref = _scalar_ref(alpha, CT)
    b_ref = _scalar_ref(beta, CT)

    hiptensorReduce(h, plan.ptr, a_ref, A, b_ref, C, C,
                    plan.workspace, UInt64(length(plan.workspace)),
                    s.stream)

    hiptensorDestroyOperationDescriptor(op_ref[])
    hiptensorDestroyPlanPreference(pref_ref[])
    return C
end

"""
    elementwise_binary!(alpha, A, modesA, gamma, C, modesC;
                        opA=HIPTENSOR_OP_IDENTITY, opC=HIPTENSOR_OP_IDENTITY,
                        opAC=HIPTENSOR_OP_ADD, compute_type=nothing)

Compute elementwise `D = opAC(alpha * opA(A), gamma * opC(C))`.
The result is written into C (since D must share C's descriptor).
"""
function elementwise_binary!(
    alpha::Number, A::ROCArray, modesA::Vector{<:Integer},
    gamma::Number, C::ROCArray, modesC::Vector{<:Integer};
    opA::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opC::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opAC::hiptensorOperator_t = HIPTENSOR_OP_ADD,
    compute_type::Union{DataType, Nothing} = nothing,
    algo::hiptensorAlgo_t = HIPTENSOR_ALGO_DEFAULT,
    jit::hiptensorJitMode_t = HIPTENSOR_JIT_MODE_NONE,
)
    CT = something(compute_type, _default_compute_type(eltype(C)))
    h = handle()
    s = stream()

    descA = HIPTensorDescriptor(h, A)
    descC = HIPTensorDescriptor(h, C)

    mA = Int32.(modesA)
    mC = Int32.(modesC)

    op_ref = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreateElementwiseBinary(h, op_ref, descA.ptr, mA, opA,
                                     descC.ptr, mC, opC,
                                     descC.ptr, mC, opAC,
                                     hiptensor_computetype(CT))

    pref_ref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(h, pref_ref, algo, jit)

    plan = HIPTensorPlan(h, op_ref[], pref_ref[])

    a_ref = _scalar_ref(alpha, CT)
    g_ref = _scalar_ref(gamma, CT)

    hiptensorElementwiseBinaryExecute(h, plan.ptr, a_ref, A, g_ref, C, C, s.stream)

    hiptensorDestroyOperationDescriptor(op_ref[])
    hiptensorDestroyPlanPreference(pref_ref[])
    return C
end

"""
    elementwise_trinary!(alpha, A, modesA, beta, B, modesB, gamma, C, modesC;
                         opA=HIPTENSOR_OP_IDENTITY, opB=HIPTENSOR_OP_IDENTITY,
                         opC=HIPTENSOR_OP_IDENTITY, opAB=HIPTENSOR_OP_ADD,
                         opABC=HIPTENSOR_OP_ADD, compute_type=nothing)

Compute elementwise `D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`.
The result is written into C.
"""
function elementwise_trinary!(
    alpha::Number, A::ROCArray, modesA::Vector{<:Integer},
    beta::Number, B::ROCArray, modesB::Vector{<:Integer},
    gamma::Number, C::ROCArray, modesC::Vector{<:Integer};
    opA::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opB::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opC::hiptensorOperator_t = HIPTENSOR_OP_IDENTITY,
    opAB::hiptensorOperator_t = HIPTENSOR_OP_ADD,
    opABC::hiptensorOperator_t = HIPTENSOR_OP_ADD,
    compute_type::Union{DataType, Nothing} = nothing,
    algo::hiptensorAlgo_t = HIPTENSOR_ALGO_DEFAULT,
    jit::hiptensorJitMode_t = HIPTENSOR_JIT_MODE_NONE,
)
    CT = something(compute_type, _default_compute_type(eltype(C)))
    h = handle()
    s = stream()

    descA = HIPTensorDescriptor(h, A)
    descB = HIPTensorDescriptor(h, B)
    descC = HIPTensorDescriptor(h, C)

    mA = Int32.(modesA)
    mB = Int32.(modesB)
    mC = Int32.(modesC)

    op_ref = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreateElementwiseTrinary(h, op_ref,
                                      descA.ptr, mA, opA,
                                      descB.ptr, mB, opB,
                                      descC.ptr, mC, opC,
                                      descC.ptr, mC,
                                      opAB, opABC,
                                      hiptensor_computetype(CT))

    pref_ref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(h, pref_ref, algo, jit)

    plan = HIPTensorPlan(h, op_ref[], pref_ref[])

    a_ref = _scalar_ref(alpha, CT)
    b_ref = _scalar_ref(beta, CT)
    g_ref = _scalar_ref(gamma, CT)

    hiptensorElementwiseTrinaryExecute(h, plan.ptr, a_ref, A, b_ref, B, g_ref, C, C, s.stream)

    hiptensorDestroyOperationDescriptor(op_ref[])
    hiptensorDestroyPlanPreference(pref_ref[])
    return C
end
