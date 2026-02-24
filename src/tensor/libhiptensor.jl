@cenum hiptensorDataType_t::UInt32 begin
    HIPTENSOR_R_32F = 0
    HIPTENSOR_R_64F = 1
    HIPTENSOR_R_16F = 2
    HIPTENSOR_R_8I = 3
    HIPTENSOR_C_32F = 4
    HIPTENSOR_C_64F = 5
    HIPTENSOR_C_16F = 6
    HIPTENSOR_C_8I = 7
    HIPTENSOR_R_8U = 8
    HIPTENSOR_C_8U = 9
    HIPTENSOR_R_32I = 10
    HIPTENSOR_C_32I = 11
    HIPTENSOR_R_32U = 12
    HIPTENSOR_C_32U = 13
    HIPTENSOR_R_16BF = 14
    HIPTENSOR_C_16BF = 15
    HIPTENSOR_R_4I = 16
    HIPTENSOR_C_4I = 17
    HIPTENSOR_R_4U = 18
    HIPTENSOR_C_4U = 19
    HIPTENSOR_R_16I = 20
    HIPTENSOR_C_16I = 21
    HIPTENSOR_R_16U = 22
    HIPTENSOR_C_16U = 23
    HIPTENSOR_R_64I = 24
    HIPTENSOR_C_64I = 25
    HIPTENSOR_R_64U = 26
    HIPTENSOR_C_64U = 27
end

@cenum hiptensorComputeDescriptor_t::UInt32 begin
    HIPTENSOR_COMPUTE_DESC_32F = 4
    HIPTENSOR_COMPUTE_DESC_64F = 16
    HIPTENSOR_COMPUTE_DESC_16F = 1
    HIPTENSOR_COMPUTE_DESC_16BF = 1024
    HIPTENSOR_COMPUTE_DESC_C32F = 2048
    HIPTENSOR_COMPUTE_DESC_C64F = 4096
    HIPTENSOR_COMPUTE_DESC_NONE = 0
    HIPTENSOR_COMPUTE_DESC_8U = 64
    HIPTENSOR_COMPUTE_DESC_8I = 256
    HIPTENSOR_COMPUTE_DESC_32U = 128
    HIPTENSOR_COMPUTE_DESC_32I = 512
end

@cenum hiptensorOperator_t::UInt32 begin
    HIPTENSOR_OP_IDENTITY = 1
    HIPTENSOR_OP_SQRT = 2
    HIPTENSOR_OP_RELU = 8
    HIPTENSOR_OP_CONJ = 9
    HIPTENSOR_OP_RCP = 10
    HIPTENSOR_OP_SIGMOID = 11
    HIPTENSOR_OP_TANH = 12
    HIPTENSOR_OP_EXP = 22
    HIPTENSOR_OP_LOG = 23
    HIPTENSOR_OP_ABS = 24
    HIPTENSOR_OP_NEG = 25
    HIPTENSOR_OP_SIN = 26
    HIPTENSOR_OP_COS = 27
    HIPTENSOR_OP_TAN = 28
    HIPTENSOR_OP_SINH = 29
    HIPTENSOR_OP_COSH = 30
    HIPTENSOR_OP_ASIN = 31
    HIPTENSOR_OP_ACOS = 32
    HIPTENSOR_OP_ATAN = 33
    HIPTENSOR_OP_ASINH = 34
    HIPTENSOR_OP_ACOSH = 35
    HIPTENSOR_OP_ATANH = 36
    HIPTENSOR_OP_CEIL = 37
    HIPTENSOR_OP_FLOOR = 38
    HIPTENSOR_OP_ADD = 3
    HIPTENSOR_OP_MUL = 5
    HIPTENSOR_OP_MAX = 6
    HIPTENSOR_OP_MIN = 7
    HIPTENSOR_OP_UNKNOWN = 126
end

@cenum hiptensorAlgo_t::Int32 begin
    HIPTENSOR_ALGO_ACTOR_CRITIC = -8
    HIPTENSOR_ALGO_DEFAULT = -1
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6
end

@cenum hiptensorWorksizePreference_t::UInt32 begin
    HIPTENSOR_WORKSPACE_MIN = 1
    HIPTENSOR_WORKSPACE_DEFAULT = 2
    HIPTENSOR_WORKSPACE_MAX = 3
end

@cenum hiptensorLogLevel_t::UInt32 begin
    HIPTENSOR_LOG_LEVEL_OFF = 0
    HIPTENSOR_LOG_LEVEL_ERROR = 1
    HIPTENSOR_LOG_LEVEL_PERF_TRACE = 2
    HIPTENSOR_LOG_LEVEL_PERF_HINT = 4
    HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8
    HIPTENSOR_LOG_LEVEL_API_TRACE = 16
end

@cenum hiptensorOperationDescriptorAttribute_t::UInt32 begin
    HIPTENSOR_OPERATION_DESCRIPTOR_TAG = 0
    HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1
    HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2
    HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6
end

@cenum hiptensorPlanPreferenceAttribute_t::UInt32 begin
    HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0
    HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1
    HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2
    HIPTENSOR_PLAN_PREFERENCE_ALGO = 3
    HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4
    HIPTENSOR_PLAN_PREFERENCE_JIT = 5
end

@cenum hiptensorPlanAttribute_t::UInt32 begin
    HIPTENSOR_PLAN_REQUIRED_WORKSPACE = 0
end

@cenum hiptensorAutotuneMode_t::UInt32 begin
    HIPTENSOR_AUTOTUNE_MODE_NONE = 0
    HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1
end

@cenum hiptensorCacheMode_t::UInt32 begin
    HIPTENSOR_CACHE_MODE_NONE = 0
    HIPTENSOR_CACHE_MODE_PEDANTIC = 1
end

@cenum hiptensorJitMode_t::UInt32 begin
    HIPTENSOR_JIT_MODE_NONE = 0
    HIPTENSOR_JIT_MODE_DEFAULT = 1
end

mutable struct hiptensorOperationDescriptor end

const hiptensorOperationDescriptor_t = Ptr{hiptensorOperationDescriptor}

mutable struct hiptensorPlan end

const hiptensorPlan_t = Ptr{hiptensorPlan}

mutable struct hiptensorPlanPreference end

const hiptensorPlanPreference_t = Ptr{hiptensorPlanPreference}

mutable struct hiptensorHandle end

const hiptensorHandle_t = Ptr{hiptensorHandle}

mutable struct hiptensorTensorDescriptor end

const hiptensorTensorDescriptor_t = Ptr{hiptensorTensorDescriptor}

# typedef void ( * hiptensorLoggerCallback_t ) ( int32_t logContext , const char * funcName , const char * msg )
const hiptensorLoggerCallback_t = Ptr{Cvoid}

function hiptensorCreate(handle)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z15hiptensorCreatePP15hiptensorHandle(handle::Ptr{hiptensorHandle_t})::hiptensorStatus_t)
end

function hiptensorDestroy(handle)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z16hiptensorDestroyP15hiptensorHandle(handle::hiptensorHandle_t)::hiptensorStatus_t)
end

function hiptensorHandleResizePlanCache(handle, numEntries)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z30hiptensorHandleResizePlanCacheP15hiptensorHandlej(handle::hiptensorHandle_t,
                                                              numEntries::UInt32)::hiptensorStatus_t)
end

function hiptensorHandleWritePlanCacheToFile(handle, filename)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z35hiptensorHandleWritePlanCacheToFileP15hiptensorHandlePKc(handle::hiptensorHandle_t,
                                                                   filename::Ptr{Cchar})::hiptensorStatus_t)
end

function hiptensorHandleReadPlanCacheFromFile(handle, filename, numCachelinesRead)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z36hiptensorHandleReadPlanCacheFromFileP15hiptensorHandlePKcPj(handle::hiptensorHandle_t,
                                                                    filename::Ptr{Cchar},
                                                                    numCachelinesRead::Ptr{UInt32})::hiptensorStatus_t)
end

function hiptensorWriteKernelCacheToFile(handle, filename)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z31hiptensorWriteKernelCacheToFileP15hiptensorHandlePKc(handle::hiptensorHandle_t,
                                                               filename::Ptr{Cchar})::hiptensorStatus_t)
end

function hiptensorReadKernelCacheFromFile(handle, filename)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z32hiptensorReadKernelCacheFromFileP15hiptensorHandlePKc(handle::hiptensorHandle_t,
                                                                filename::Ptr{Cchar})::hiptensorStatus_t)
end

function hiptensorCreateTensorDescriptor(handle, desc, numModes, lens, strides, dataType,
                                         alignmentRequirement)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z31hiptensorCreateTensorDescriptorP15hiptensorHandlePP25hiptensorTensorDescriptorjPKlS5_19hiptensorDataType_tj(handle::hiptensorHandle_t,
                                                               desc::Ptr{hiptensorTensorDescriptor_t},
                                                               numModes::UInt32,
                                                               lens::Ptr{Int64},
                                                               strides::Ptr{Int64},
                                                               dataType::hiptensorDataType_t,
                                                               alignmentRequirement::UInt32)::hiptensorStatus_t)
end

function hiptensorDestroyTensorDescriptor(desc)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z32hiptensorDestroyTensorDescriptorP25hiptensorTensorDescriptor(desc::hiptensorTensorDescriptor_t)::hiptensorStatus_t)
end

function hiptensorCreateContraction(handle, desc, descA, modeA, opA, descB, modeB, opB,
                                    descC, modeC, opC, descD, modeD, descCompute)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z26hiptensorCreateContractionP15hiptensorHandlePP28hiptensorOperationDescriptorP25hiptensorTensorDescriptorPKi19hiptensorOperator_tS5_S7_S8_S5_S7_S8_S5_S7_28hiptensorComputeDescriptor_t(handle::hiptensorHandle_t,
                                                          desc::Ptr{hiptensorOperationDescriptor_t},
                                                          descA::hiptensorTensorDescriptor_t,
                                                          modeA::Ptr{Int32},
                                                          opA::hiptensorOperator_t,
                                                          descB::hiptensorTensorDescriptor_t,
                                                          modeB::Ptr{Int32},
                                                          opB::hiptensorOperator_t,
                                                          descC::hiptensorTensorDescriptor_t,
                                                          modeC::Ptr{Int32},
                                                          opC::hiptensorOperator_t,
                                                          descD::hiptensorTensorDescriptor_t,
                                                          modeD::Ptr{Int32},
                                                          descCompute::hiptensorComputeDescriptor_t)::hiptensorStatus_t)
end

function hiptensorDestroyOperationDescriptor(desc)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z35hiptensorDestroyOperationDescriptorP28hiptensorOperationDescriptor(desc::hiptensorOperationDescriptor_t)::hiptensorStatus_t)
end

function hiptensorOperationDescriptorSetAttribute(handle, desc, attr, buf, sizeInBytes)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z40hiptensorOperationDescriptorSetAttributeP15hiptensorHandleP28hiptensorOperationDescriptor39hiptensorOperationDescriptorAttribute_tPKvm(handle::hiptensorHandle_t,
                                                                        desc::hiptensorOperationDescriptor_t,
                                                                        attr::hiptensorOperationDescriptorAttribute_t,
                                                                        buf::Ptr{Cvoid},
                                                                        sizeInBytes::Csize_t)::hiptensorStatus_t)
end

function hiptensorOperationDescriptorGetAttribute(handle, desc, attr, buf, sizeInBytes)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z40hiptensorOperationDescriptorGetAttributeP15hiptensorHandleP28hiptensorOperationDescriptor39hiptensorOperationDescriptorAttribute_tPvm(handle::hiptensorHandle_t,
                                                                        desc::hiptensorOperationDescriptor_t,
                                                                        attr::hiptensorOperationDescriptorAttribute_t,
                                                                        buf::Ptr{Cvoid},
                                                                        sizeInBytes::Csize_t)::hiptensorStatus_t)
end

function hiptensorCreatePlanPreference(handle, pref, algo, jitMode)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z29hiptensorCreatePlanPreferenceP15hiptensorHandlePP23hiptensorPlanPreference15hiptensorAlgo_t18hiptensorJitMode_t(handle::hiptensorHandle_t,
                                                             pref::Ptr{hiptensorPlanPreference_t},
                                                             algo::hiptensorAlgo_t,
                                                             jitMode::hiptensorJitMode_t)::hiptensorStatus_t)
end

function hiptensorDestroyPlanPreference(pref)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z30hiptensorDestroyPlanPreferenceP23hiptensorPlanPreference(pref::hiptensorPlanPreference_t)::hiptensorStatus_t)
end

function hiptensorPlanPreferenceSetAttribute(handle, pref, attr, buf, sizeInBytes)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z35hiptensorPlanPreferenceSetAttributeP15hiptensorHandleP23hiptensorPlanPreference34hiptensorPlanPreferenceAttribute_tPKvm(handle::hiptensorHandle_t,
                                                                   pref::hiptensorPlanPreference_t,
                                                                   attr::hiptensorPlanPreferenceAttribute_t,
                                                                   buf::Ptr{Cvoid},
                                                                   sizeInBytes::Csize_t)::hiptensorStatus_t)
end

function hiptensorPlanGetAttribute(handle, plan, attr, buf, sizeInBytes)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z25hiptensorPlanGetAttributeP15hiptensorHandleP13hiptensorPlan24hiptensorPlanAttribute_tPvm(handle::hiptensorHandle_t,
                                                         plan::hiptensorPlan_t,
                                                         attr::hiptensorPlanAttribute_t,
                                                         buf::Ptr{Cvoid},
                                                         sizeInBytes::Csize_t)::hiptensorStatus_t)
end

function hiptensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref,
                                        workspaceSizeEstimate)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z30hiptensorEstimateWorkspaceSizeP15hiptensorHandleP28hiptensorOperationDescriptorP23hiptensorPlanPreference29hiptensorWorksizePreference_tPm(handle::hiptensorHandle_t,
                                                              desc::hiptensorOperationDescriptor_t,
                                                              planPref::hiptensorPlanPreference_t,
                                                              workspacePref::hiptensorWorksizePreference_t,
                                                              workspaceSizeEstimate::Ptr{UInt64})::hiptensorStatus_t)
end

function hiptensorCreatePermutation(handle, desc, descA, modeA, opA, descB, modeB,
                                    descCompute)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z26hiptensorCreatePermutationP15hiptensorHandlePP28hiptensorOperationDescriptorP25hiptensorTensorDescriptorPKi19hiptensorOperator_tS5_S7_28hiptensorComputeDescriptor_t(handle::hiptensorHandle_t,
                                                          desc::Ptr{hiptensorOperationDescriptor_t},
                                                          descA::hiptensorTensorDescriptor_t,
                                                          modeA::Ptr{Int32},
                                                          opA::hiptensorOperator_t,
                                                          descB::hiptensorTensorDescriptor_t,
                                                          modeB::Ptr{Int32},
                                                          descCompute::hiptensorComputeDescriptor_t)::hiptensorStatus_t)
end

function hiptensorCreatePlan(handle, plan, desc, pref, workspaceSizeLimit)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z19hiptensorCreatePlanP15hiptensorHandlePP13hiptensorPlanP28hiptensorOperationDescriptorP23hiptensorPlanPreferencem(handle::hiptensorHandle_t,
                                                   plan::Ptr{hiptensorPlan_t},
                                                   desc::hiptensorOperationDescriptor_t,
                                                   pref::hiptensorPlanPreference_t,
                                                   workspaceSizeLimit::UInt64)::hiptensorStatus_t)
end

function hiptensorDestroyPlan(plan)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z20hiptensorDestroyPlanP13hiptensorPlan(plan::hiptensorPlan_t)::hiptensorStatus_t)
end

function hiptensorContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize,
                           stream)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z17hiptensorContractP15hiptensorHandleP13hiptensorPlanPKvS4_S4_S4_S4_PvS5_mP12ihipStream_t(handle::hiptensorHandle_t,
                                                 plan::hiptensorPlan_t, alpha::Ptr{Cvoid},
                                                 A::Ptr{Cvoid}, B::Ptr{Cvoid},
                                                 beta::Ptr{Cvoid}, C::Ptr{Cvoid},
                                                 D::Ptr{Cvoid}, workspace::Ptr{Cvoid},
                                                 workspaceSize::UInt64,
                                                 stream::hipStream_t)::hiptensorStatus_t)
end

function hiptensorGetErrorString(error)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z23hiptensorGetErrorString17hiptensorStatus_t(error::hiptensorStatus_t)::Ptr{Cchar})
end

function hiptensorPermute(handle, plan, alpha, A, B, stream)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z16hiptensorPermuteP15hiptensorHandleP13hiptensorPlanPKvS4_PvP12ihipStream_t(handle::hiptensorHandle_t,
                                                plan::hiptensorPlan_t, alpha::Ptr{Cvoid},
                                                A::Ptr{Cvoid}, B::Ptr{Cvoid},
                                                stream::hipStream_t)::hiptensorStatus_t)
end

function hiptensorCreateElementwiseBinary(handle, desc, descA, modeA, opA, descC, modeC,
                                          opC, descD, modeD, opAC, descCompute)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z32hiptensorCreateElementwiseBinaryP15hiptensorHandlePP28hiptensorOperationDescriptorP25hiptensorTensorDescriptorPKi19hiptensorOperator_tS5_S7_S8_S5_S7_S8_28hiptensorComputeDescriptor_t(handle::hiptensorHandle_t,
                                                                desc::Ptr{hiptensorOperationDescriptor_t},
                                                                descA::hiptensorTensorDescriptor_t,
                                                                modeA::Ptr{Int32},
                                                                opA::hiptensorOperator_t,
                                                                descC::hiptensorTensorDescriptor_t,
                                                                modeC::Ptr{Int32},
                                                                opC::hiptensorOperator_t,
                                                                descD::hiptensorTensorDescriptor_t,
                                                                modeD::Ptr{Int32},
                                                                opAC::hiptensorOperator_t,
                                                                descCompute::hiptensorComputeDescriptor_t)::hiptensorStatus_t)
end

function hiptensorElementwiseBinaryExecute(handle, plan, alpha, A, gamma, C, D, stream)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z33hiptensorElementwiseBinaryExecuteP15hiptensorHandleP13hiptensorPlanPKvS4_S4_S4_PvP12ihipStream_t(handle::hiptensorHandle_t,
                                                                 plan::hiptensorPlan_t,
                                                                 alpha::Ptr{Cvoid},
                                                                 A::Ptr{Cvoid},
                                                                 gamma::Ptr{Cvoid},
                                                                 C::Ptr{Cvoid},
                                                                 D::Ptr{Cvoid},
                                                                 stream::hipStream_t)::hiptensorStatus_t)
end

function hiptensorCreateElementwiseTrinary(handle, desc, descA, modeA, opA, descB, modeB,
                                           opB, descC, modeC, opC, descD, modeD, opAB,
                                           opABC, descCompute)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z33hiptensorCreateElementwiseTrinaryP15hiptensorHandlePP28hiptensorOperationDescriptorP25hiptensorTensorDescriptorPKi19hiptensorOperator_tS5_S7_S8_S5_S7_S8_S5_S7_S8_S8_28hiptensorComputeDescriptor_t(handle::hiptensorHandle_t,
                                                                 desc::Ptr{hiptensorOperationDescriptor_t},
                                                                 descA::hiptensorTensorDescriptor_t,
                                                                 modeA::Ptr{Int32},
                                                                 opA::hiptensorOperator_t,
                                                                 descB::hiptensorTensorDescriptor_t,
                                                                 modeB::Ptr{Int32},
                                                                 opB::hiptensorOperator_t,
                                                                 descC::hiptensorTensorDescriptor_t,
                                                                 modeC::Ptr{Int32},
                                                                 opC::hiptensorOperator_t,
                                                                 descD::hiptensorTensorDescriptor_t,
                                                                 modeD::Ptr{Int32},
                                                                 opAB::hiptensorOperator_t,
                                                                 opABC::hiptensorOperator_t,
                                                                 descCompute::hiptensorComputeDescriptor_t)::hiptensorStatus_t)
end

function hiptensorElementwiseTrinaryExecute(handle, plan, alpha, A, beta, B, gamma, C, D,
                                            stream)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z34hiptensorElementwiseTrinaryExecuteP15hiptensorHandleP13hiptensorPlanPKvS4_S4_S4_S4_S4_PvP12ihipStream_t(handle::hiptensorHandle_t,
                                                                  plan::hiptensorPlan_t,
                                                                  alpha::Ptr{Cvoid},
                                                                  A::Ptr{Cvoid},
                                                                  beta::Ptr{Cvoid},
                                                                  B::Ptr{Cvoid},
                                                                  gamma::Ptr{Cvoid},
                                                                  C::Ptr{Cvoid},
                                                                  D::Ptr{Cvoid},
                                                                  stream::hipStream_t)::hiptensorStatus_t)
end

function hiptensorCreateReduction(handle, desc, descA, modeA, opA, descC, modeC, opC, descD,
                                  modeD, opReduce, descCompute)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z24hiptensorCreateReductionP15hiptensorHandlePP28hiptensorOperationDescriptorP25hiptensorTensorDescriptorPKi19hiptensorOperator_tS5_S7_S8_S5_S7_S8_28hiptensorComputeDescriptor_t(handle::hiptensorHandle_t,
                                                        desc::Ptr{hiptensorOperationDescriptor_t},
                                                        descA::hiptensorTensorDescriptor_t,
                                                        modeA::Ptr{Int32},
                                                        opA::hiptensorOperator_t,
                                                        descC::hiptensorTensorDescriptor_t,
                                                        modeC::Ptr{Int32},
                                                        opC::hiptensorOperator_t,
                                                        descD::hiptensorTensorDescriptor_t,
                                                        modeD::Ptr{Int32},
                                                        opReduce::hiptensorOperator_t,
                                                        descCompute::hiptensorComputeDescriptor_t)::hiptensorStatus_t)
end

function hiptensorReduce(handle, plan, alpha, A, beta, C, D, workspace, workspaceSize,
                         stream)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z15hiptensorReduceP15hiptensorHandleP13hiptensorPlanPKvS4_S4_S4_PvS5_mP12ihipStream_t(handle::hiptensorHandle_t,
                                               plan::hiptensorPlan_t, alpha::Ptr{Cvoid},
                                               A::Ptr{Cvoid}, beta::Ptr{Cvoid},
                                               C::Ptr{Cvoid}, D::Ptr{Cvoid},
                                               workspace::Ptr{Cvoid}, workspaceSize::UInt64,
                                               stream::hipStream_t)::hiptensorStatus_t)
end

function hiptensorLoggerSetCallback(callback)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z26hiptensorLoggerSetCallbackPFviPKcS0_E(callback::hiptensorLoggerCallback_t)::hiptensorStatus_t)
end

function hiptensorLoggerSetFile(file)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z22hiptensorLoggerSetFileP8_IO_FILE(file::Ptr{Cvoid})::hiptensorStatus_t)
end

function hiptensorLoggerOpenFile(logFile)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z23hiptensorLoggerOpenFilePKc(logFile::Ptr{Cchar})::hiptensorStatus_t)
end

function hiptensorLoggerSetLevel(level)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z23hiptensorLoggerSetLevel19hiptensorLogLevel_t(level::hiptensorLogLevel_t)::hiptensorStatus_t)
end

function hiptensorLoggerSetMask(mask)
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z22hiptensorLoggerSetMaski(mask::Int32)::hiptensorStatus_t)
end

# no prototype is found for this function at hiptensor.h:513:19, please use with caution
function hiptensorLoggerForceDisable()
    AMDGPU.prepare_state()
    @check @ccall(libhiptensor._Z27hiptensorLoggerForceDisablev()::hiptensorStatus_t)
end

# no prototype is found for this function at hiptensor.h:515:5, please use with caution
function hiptensorGetHiprtVersion()
    @ccall(libhiptensor.hiptensorGetHiprtVersion()::Cint)
end
