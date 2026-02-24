export HIPTensorError

@cenum hiptensorStatus_t::UInt32 begin
    HIPTENSOR_STATUS_SUCCESS = 0
    HIPTENSOR_STATUS_NOT_INITIALIZED = 1
    HIPTENSOR_STATUS_ALLOC_FAILED = 3
    HIPTENSOR_STATUS_INVALID_VALUE = 7
    HIPTENSOR_STATUS_ARCH_MISMATCH = 8
    HIPTENSOR_STATUS_EXECUTION_FAILED = 13
    HIPTENSOR_STATUS_INTERNAL_ERROR = 14
    HIPTENSOR_STATUS_NOT_SUPPORTED = 15
    HIPTENSOR_STATUS_CK_ERROR = 17
    HIPTENSOR_STATUS_HIP_ERROR = 18
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER = 20
    HIPTENSOR_STATUS_IO_ERROR = 21
end

struct HIPTensorError <: Exception
    code::hiptensorStatus_t
    msg::AbstractString
end

function Base.show(io::IO, err::HIPTensorError)
    print(io, "HIPTensorError(code $(err.code), $(err.msg))")
end

function HIPTensorError(code::hiptensorStatus_t)
    msg = status_message(code)
    return HIPTensorError(code, msg)
end

function status_message(status)
    str = hiptensorGetErrorString(status)
    str == C_NULL ? "unknown hipTensor error" : unsafe_string(str)
end

function check(status::hiptensorStatus_t)
    if status != HIPTENSOR_STATUS_SUCCESS
        throw(HIPTensorError(status))
    end
    return status
end
