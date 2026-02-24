const HIPTensorDataTypes = Dict{DataType, hiptensorDataType_t}(
    Float16          => HIPTENSOR_R_16F,
    Float32          => HIPTENSOR_R_32F,
    Float64          => HIPTENSOR_R_64F,
    ComplexF16       => HIPTENSOR_C_16F,
    ComplexF32       => HIPTENSOR_C_32F,
    ComplexF64       => HIPTENSOR_C_64F,
    Int8             => HIPTENSOR_R_8I,
    UInt8            => HIPTENSOR_R_8U,
    Int32            => HIPTENSOR_R_32I,
    UInt32           => HIPTENSOR_R_32U,
    BFloat16         => HIPTENSOR_R_16BF,
)

function hiptensor_datatype(T::DataType)
    haskey(HIPTensorDataTypes, T) || throw(ArgumentError("hipTensor does not support $T"))
    return HIPTensorDataTypes[T]
end

const HIPTensorComputeTypes = Dict{DataType, hiptensorComputeDescriptor_t}(
    Float16    => HIPTENSOR_COMPUTE_DESC_16F,
    Float32    => HIPTENSOR_COMPUTE_DESC_32F,
    Float64    => HIPTENSOR_COMPUTE_DESC_64F,
    BFloat16   => HIPTENSOR_COMPUTE_DESC_16BF,
    ComplexF32 => HIPTENSOR_COMPUTE_DESC_C32F,
    ComplexF64 => HIPTENSOR_COMPUTE_DESC_C64F,
)

function hiptensor_computetype(T::DataType)
    haskey(HIPTensorComputeTypes, T) || throw(ArgumentError("hipTensor does not support compute type $T"))
    return HIPTensorComputeTypes[T]
end

mutable struct HIPTensorDescriptor
    ptr::hiptensorTensorDescriptor_t

    function HIPTensorDescriptor(handle, A::ROCArray; alignment::UInt32 = UInt32(256))
        T = eltype(A)
        nmodes = UInt32(ndims(A))
        lens = Int64[size(A)...]
        strd = Int64[strides(A)...]
        desc_ref = Ref{hiptensorTensorDescriptor_t}()
        hiptensorCreateTensorDescriptor(
            handle, desc_ref, nmodes, lens, strd,
            hiptensor_datatype(T), alignment)
        obj = new(desc_ref[])
        finalizer(obj) do d
            hiptensorDestroyTensorDescriptor(d.ptr)
        end
        return obj
    end
end

Base.unsafe_convert(::Type{hiptensorTensorDescriptor_t}, d::HIPTensorDescriptor) = d.ptr

mutable struct HIPTensorPlan
    ptr::hiptensorPlan_t
    workspace::ROCVector{UInt8}

    function HIPTensorPlan(handle, desc, pref; workspace_pref = HIPTENSOR_WORKSPACE_DEFAULT)
        ws_ref = Ref{UInt64}(0)
        hiptensorEstimateWorkspaceSize(handle, desc, pref, workspace_pref, ws_ref)
        workspace = ROCVector{UInt8}(undef, ws_ref[])
        plan_ref = Ref{hiptensorPlan_t}()
        hiptensorCreatePlan(handle, plan_ref, desc, pref, ws_ref[])
        obj = new(plan_ref[], workspace)
        finalizer(obj) do p
            hiptensorDestroyPlan(p.ptr)
        end
        return obj
    end
end

Base.unsafe_convert(::Type{hiptensorPlan_t}, p::HIPTensorPlan) = p.ptr
