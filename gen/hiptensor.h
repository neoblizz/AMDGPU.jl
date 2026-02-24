#include <stddef.h>
#include <stdint.h>

typedef void* hipStream_t;
typedef struct FILE FILE;

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/



//! @brief hipTensor data types
typedef enum hiptensorDataType_t
{
    HIPTENSOR_R_32F  = 0,
    HIPTENSOR_R_64F  = 1,
    HIPTENSOR_R_16F  = 2,
    HIPTENSOR_R_8I   = 3,
    HIPTENSOR_C_32F  = 4,
    HIPTENSOR_C_64F  = 5,
    HIPTENSOR_C_16F  = 6,
    HIPTENSOR_C_8I   = 7,
    HIPTENSOR_R_8U   = 8,
    HIPTENSOR_C_8U   = 9,
    HIPTENSOR_R_32I  = 10,
    HIPTENSOR_C_32I  = 11,
    HIPTENSOR_R_32U  = 12,
    HIPTENSOR_C_32U  = 13,
    HIPTENSOR_R_16BF = 14,
    HIPTENSOR_C_16BF = 15,
    HIPTENSOR_R_4I   = 16,
    HIPTENSOR_C_4I   = 17,
    HIPTENSOR_R_4U   = 18,
    HIPTENSOR_C_4U   = 19,
    HIPTENSOR_R_16I  = 20,
    HIPTENSOR_C_16I  = 21,
    HIPTENSOR_R_16U  = 22,
    HIPTENSOR_C_16U  = 23,
    HIPTENSOR_R_64I  = 24,
    HIPTENSOR_C_64I  = 25,
    HIPTENSOR_R_64U  = 26,
    HIPTENSOR_C_64U  = 27,
} hiptensorDataType_t;

//! @brief hipTensor status type enumeration
//! @details The type is used to indicate the resulting status of hipTensor library function calls
typedef enum
{
    //! The operation is successful.
    HIPTENSOR_STATUS_SUCCESS = 0,
    //! The handle was not initialized.
    HIPTENSOR_STATUS_NOT_INITIALIZED = 1,
    //! Resource allocation failed inside the hipTensor library.
    HIPTENSOR_STATUS_ALLOC_FAILED = 3,
    //! Invalid value or parameter was passed to the function (indicates a user error).
    HIPTENSOR_STATUS_INVALID_VALUE = 7,
    //! Indicates that the target architecure is not supported, or the device is not ready.
    HIPTENSOR_STATUS_ARCH_MISMATCH = 8,
    //! Indicates the failure of a GPU program or a kernel, which can be caused by multiple reasons.
    HIPTENSOR_STATUS_EXECUTION_FAILED = 13,
    //! An internal error has occurred.
    HIPTENSOR_STATUS_INTERNAL_ERROR = 14,
    //! The requested operation is not supported.
    HIPTENSOR_STATUS_NOT_SUPPORTED = 15,
    //! A call to Composable Kernels did not succeed.
    HIPTENSOR_STATUS_CK_ERROR = 17,
    //! Unknown hipTensor error has occurred.
    HIPTENSOR_STATUS_HIP_ERROR = 18,
    //! The provided workspace was insufficient.
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    //! Indicates that the driver version is insufficient.
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    //! Indicates an error related to file I/O.
    HIPTENSOR_STATUS_IO_ERROR = 21,

} hiptensorStatus_t;

//! @brief hipTensor compute type enumeration
typedef enum
{
    //! Single precision floating point
    HIPTENSOR_COMPUTE_DESC_32F = (1U << 2U),
    //! Double precision floating point
    HIPTENSOR_COMPUTE_DESC_64F = (1U << 4U),
    //! Half precision floating point
    HIPTENSOR_COMPUTE_DESC_16F = (1U << 0U),
    //! Brain float half precision floating point
    HIPTENSOR_COMPUTE_DESC_16BF = (1U << 10U),
    //! Complex single precision floating point
    HIPTENSOR_COMPUTE_DESC_C32F = (1U << 11U),
    //! Complex double precision floating point
    HIPTENSOR_COMPUTE_DESC_C64F = (1U << 12U),
    //! No type
    HIPTENSOR_COMPUTE_DESC_NONE = 0,

    // @cond
    //! <Following types to be added (TBA)>
    HIPTENSOR_COMPUTE_DESC_8U  = (1U << 6U),
    HIPTENSOR_COMPUTE_DESC_8I  = (1U << 8U),
    HIPTENSOR_COMPUTE_DESC_32U = (1U << 7U),
    HIPTENSOR_COMPUTE_DESC_32I = (1U << 9U),
    // @endcond

} hiptensorComputeDescriptor_t;

//! @brief Element-wise operations
typedef enum
{
    HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
    HIPTENSOR_OP_SQRT     = 2, ///< Square root
    HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
    HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
    HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
    HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
    HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
    HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
    HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
    HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
    HIPTENSOR_OP_NEG      = 25, ///< Negation.
    HIPTENSOR_OP_SIN      = 26, ///< Sine.
    HIPTENSOR_OP_COS      = 27, ///< Cosine.
    HIPTENSOR_OP_TAN      = 28, ///< Tangent.
    HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
    HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
    HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
    HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
    HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
    HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
    HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
    HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
    HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
    HIPTENSOR_OP_FLOOR    = 38, ///< Floor.

    /* Binary */
    HIPTENSOR_OP_ADD = 3, ///< Addition of two elements
    HIPTENSOR_OP_MUL = 5, ///< Multiplication of two elements
    HIPTENSOR_OP_MAX = 6, ///< Maximum of two elements
    HIPTENSOR_OP_MIN = 7, ///< Minimum of two elements

    HIPTENSOR_OP_UNKNOWN = 126, ///< reserved for internal use only)
} hiptensorOperator_t;

//! @brief Tensor contraction kernel selection algorithm
typedef enum
{
    //! Uses novel actor-critic selection model
    HIPTENSOR_ALGO_ACTOR_CRITIC = -8,
    //! Lets the internal heuristic choose
    HIPTENSOR_ALGO_DEFAULT = -1,
    //! Uses the more accurate and time-consuming model
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6,

} hiptensorAlgo_t;

//! @brief Workspace size selection
typedef enum
{
    //! At least one algorithm will be available
    HIPTENSOR_WORKSPACE_MIN = 1,
    //! The most suitable algorithm will be available
    HIPTENSOR_WORKSPACE_DEFAULT = 2,
    //! All algorithms will be available
    HIPTENSOR_WORKSPACE_MAX = 3,

} hiptensorWorksizePreference_t;

//! @brief Logging context
//! @details The logger output of certain contexts maybe constrained to these levels
typedef enum
{
    //! No logging
    HIPTENSOR_LOG_LEVEL_OFF = 0,
    //! Log errors
    HIPTENSOR_LOG_LEVEL_ERROR = 1,
    //! Log performance messages
    HIPTENSOR_LOG_LEVEL_PERF_TRACE = 2,
    //! Log performance hints
    HIPTENSOR_LOG_LEVEL_PERF_HINT = 4,
    //! Log selection messages
    HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8,
    //! Log a trace of API calls
    HIPTENSOR_LOG_LEVEL_API_TRACE = 16,

} hiptensorLogLevel_t;

typedef enum
{
    HIPTENSOR_OPERATION_DESCRIPTOR_TAG           = 0,
    HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE   = 1,
    HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS         = 2,
    HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES   = 3,
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT  = 4,
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5,
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6,
} hiptensorOperationDescriptorAttribute_t;

typedef enum
{
    HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE     = 0,
    HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE        = 1,
    HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
    HIPTENSOR_PLAN_PREFERENCE_ALGO              = 3,
    HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK       = 4,
    HIPTENSOR_PLAN_PREFERENCE_JIT               = 5,
} hiptensorPlanPreferenceAttribute_t;

typedef enum
{
    HIPTENSOR_PLAN_REQUIRED_WORKSPACE = 0,
} hiptensorPlanAttribute_t;

typedef enum
{
    HIPTENSOR_AUTOTUNE_MODE_NONE        = 0,
    HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1,
} hiptensorAutotuneMode_t;

typedef enum
{
    HIPTENSOR_CACHE_MODE_NONE     = 0,
    HIPTENSOR_CACHE_MODE_PEDANTIC = 1,
} hiptensorCacheMode_t;

typedef enum
{
    HIPTENSOR_JIT_MODE_NONE    = 0,
    HIPTENSOR_JIT_MODE_DEFAULT = 1,
} hiptensorJitMode_t;

//! @brief Pointer to hiptensorOperationDescriptor
typedef struct hiptensorOperationDescriptor* hiptensorOperationDescriptor_t;

//! @brief Pointer to hiptensorPlan
typedef struct hiptensorPlan*                hiptensorPlan_t;

//! @brief Pointer to hiptensorPlanPreference
typedef struct hiptensorPlanPreference*      hiptensorPlanPreference_t;

//! @brief Pointer to hiptensorHandle
typedef struct hiptensorHandle*              hiptensorHandle_t;

//! @brief Pointer to hiptensorTensorDescriptor
typedef struct hiptensorTensorDescriptor*    hiptensorTensorDescriptor_t;

//! @brief Logging callback
//! The specified callback is invoked whenever logging is enabled and a message is generated.
//! @param logContext The logging context enum
//! @param funcName A string holding the function name where the logging message was generated
//! @param msg A string holding the logging message
typedef void (*hiptensorLoggerCallback_t)(int32_t     logContext,
                                          const char* funcName,
                                          const char* msg);



/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/


hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle);

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle);

hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle,
                                                 const uint32_t    numEntries);

hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle,
                                                      const char              filename[]);

hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle,
                                                       const char        filename[],
                                                       uint32_t*         numCachelinesRead);

hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle,
                                                  const char              filename[]);

hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle, const char filename[]);

hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t      handle,
                                                  hiptensorTensorDescriptor_t* desc,
                                                  const uint32_t               numModes,
                                                  const int64_t                lens[],
                                                  const int64_t                strides[],
                                                  hiptensorDataType_t          dataType,
                                                  uint32_t alignmentRequirement);

hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc);

hiptensorStatus_t hiptensorCreateContraction(const hiptensorHandle_t            handle,
                                             hiptensorOperationDescriptor_t*    desc,
                                             const hiptensorTensorDescriptor_t  descA,
                                             const int32_t                      modeA[],
                                             hiptensorOperator_t                opA,
                                             const hiptensorTensorDescriptor_t  descB,
                                             const int32_t                      modeB[],
                                             hiptensorOperator_t                opB,
                                             const hiptensorTensorDescriptor_t  descC,
                                             const int32_t                      modeC[],
                                             hiptensorOperator_t                opC,
                                             const hiptensorTensorDescriptor_t  descD,
                                             const int32_t                      modeD[],
                                             const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t desc);

hiptensorStatus_t
    hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             const void*                             buf,
                                             size_t                                  sizeInBytes);

hiptensorStatus_t
    hiptensorOperationDescriptorGetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             void*                                   buf,
                                             size_t                                  sizeInBytes);

hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t    handle,
                                                hiptensorPlanPreference_t* pref,
                                                hiptensorAlgo_t            algo,
                                                hiptensorJitMode_t         jitMode);

hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref);

hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t            handle,
                                                      hiptensorPlanPreference_t          pref,
                                                      hiptensorPlanPreferenceAttribute_t attr,
                                                      const void*                        buf,
                                                      size_t sizeInBytes);

hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t  handle,
                                            const hiptensorPlan_t    plan,
                                            hiptensorPlanAttribute_t attr,
                                            void*                    buf,
                                            size_t                   sizeInBytes);

hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t              handle,
                                                 const hiptensorOperationDescriptor_t desc,
                                                 const hiptensorPlanPreference_t      planPref,
                                                 const hiptensorWorksizePreference_t  workspacePref,
                                                 uint64_t* workspaceSizeEstimate);

hiptensorStatus_t hiptensorCreatePermutation(const hiptensorHandle_t            handle,
                                             hiptensorOperationDescriptor_t*    desc,
                                             const hiptensorTensorDescriptor_t  descA,
                                             const int32_t                      modeA[],
                                             hiptensorOperator_t                opA,
                                             const hiptensorTensorDescriptor_t  descB,
                                             const int32_t                      modeB[],
                                             const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t*                     plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit);

hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan);

hiptensorStatus_t hiptensorContract(const hiptensorHandle_t handle,
                                    const hiptensorPlan_t   plan,
                                    const void*             alpha,
                                    const void*             A,
                                    const void*             B,
                                    const void*             beta,
                                    const void*             C,
                                    void*                   D,
                                    void*                   workspace,
                                    uint64_t                workspaceSize,
                                    hipStream_t             stream);

const char* hiptensorGetErrorString(const hiptensorStatus_t error);

hiptensorStatus_t hiptensorPermute(const hiptensorHandle_t handle,
                                   const hiptensorPlan_t   plan,
                                   const void*             alpha,
                                   const void*             A,
                                   void*                   B,
                                   const hipStream_t       stream);

hiptensorStatus_t hiptensorCreateElementwiseBinary(const hiptensorHandle_t            handle,
                                                   hiptensorOperationDescriptor_t*    desc,
                                                   const hiptensorTensorDescriptor_t  descA,
                                                   const int32_t                      modeA[],
                                                   hiptensorOperator_t                opA,
                                                   const hiptensorTensorDescriptor_t  descC,
                                                   const int32_t                      modeC[],
                                                   hiptensorOperator_t                opC,
                                                   const hiptensorTensorDescriptor_t  descD,
                                                   const int32_t                      modeD[],
                                                   hiptensorOperator_t                opAC,
                                                   const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorElementwiseBinaryExecute(const hiptensorHandle_t handle,
                                                    const hiptensorPlan_t   plan,
                                                    const void*             alpha,
                                                    const void*             A,
                                                    const void*             gamma,
                                                    const void*             C,
                                                    void*                   D,
                                                    hipStream_t             stream);

hiptensorStatus_t hiptensorCreateElementwiseTrinary(const hiptensorHandle_t            handle,
                                                    hiptensorOperationDescriptor_t*    desc,
                                                    const hiptensorTensorDescriptor_t  descA,
                                                    const int32_t                      modeA[],
                                                    hiptensorOperator_t                opA,
                                                    const hiptensorTensorDescriptor_t  descB,
                                                    const int32_t                      modeB[],
                                                    hiptensorOperator_t                opB,
                                                    const hiptensorTensorDescriptor_t  descC,
                                                    const int32_t                      modeC[],
                                                    hiptensorOperator_t                opC,
                                                    const hiptensorTensorDescriptor_t  descD,
                                                    const int32_t                      modeD[],
                                                    hiptensorOperator_t                opAB,
                                                    hiptensorOperator_t                opABC,
                                                    const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorElementwiseTrinaryExecute(const hiptensorHandle_t handle,
                                                     const hiptensorPlan_t   plan,
                                                     const void*             alpha,
                                                     const void*             A,
                                                     const void*             beta,
                                                     const void*             B,
                                                     const void*             gamma,
                                                     const void*             C,
                                                     void*                   D,
                                                     hipStream_t             stream);

hiptensorStatus_t hiptensorCreateReduction(const hiptensorHandle_t            handle,
                                           hiptensorOperationDescriptor_t*    desc,
                                           const hiptensorTensorDescriptor_t  descA,
                                           const int32_t                      modeA[],
                                           hiptensorOperator_t                opA,
                                           const hiptensorTensorDescriptor_t  descC,
                                           const int32_t                      modeC[],
                                           hiptensorOperator_t                opC,
                                           const hiptensorTensorDescriptor_t  descD,
                                           const int32_t                      modeD[],
                                           hiptensorOperator_t                opReduce,
                                           const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorReduce(const hiptensorHandle_t handle,
                                  const hiptensorPlan_t   plan,
                                  const void*             alpha,
                                  const void*             A,
                                  const void*             beta,
                                  const void*             C,
                                  void*                   D,
                                  void*                   workspace,
                                  uint64_t                workspaceSize,
                                  hipStream_t             stream);

hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);

hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);

hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);

hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);

hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);

hiptensorStatus_t hiptensorLoggerForceDisable();

int hiptensorGetHiprtVersion();


