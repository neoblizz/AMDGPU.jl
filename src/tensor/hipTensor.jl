module hipTensor

using ..AMDGPU
import AMDGPU: libhiptensor, AnyROCArray, ROCArray, ROCVector
import AMDGPU: HandleCache, HIP, library_state
import AMDGPU: @check, check
import .HIP: HIPContext, HIPStream, hipStream_t

using GPUArrays
using LinearAlgebra
using CEnum
using BFloat16s

include("error.jl")
include("libhiptensor.jl")
include("types.jl")
include("operations.jl")
include("interfaces.jl")

function create_handle()
    AMDGPU.functional(:hiptensor) || error("hipTensor is not available")
    handle_ref = Ref{hiptensorHandle_t}()
    @check hiptensorCreate(handle_ref)
    handle_ref[]
end

function destroy_handle!(handle)
    hiptensorDestroy(handle)
end

const IDLE_HANDLES = HandleCache{HIPContext, hiptensorHandle_t}()

function lib_state()
    return library_state(
        :hipTensor, hiptensorHandle_t, IDLE_HANDLES,
        create_handle, destroy_handle!,
        (nh, s) -> nothing)
end

handle() = lib_state().handle
stream() = lib_state().stream

function version()
    VersionNumber(2, 0, 0)
end

end
