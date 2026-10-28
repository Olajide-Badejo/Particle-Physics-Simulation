#pragma once

// Test helpers for the binaries that need the CUDA runtime. Everything that
// needs no device lives in test_support_host.hpp, which this includes.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "test_support_host.hpp"

namespace cps_test {

// Exit code a GPU test binary returns when it cannot run at all. ctest maps it
// to "Not Run" through SKIP_RETURN_CODE, so a machine with no device reports
// the GPU entries as skipped rather than passed.
inline constexpr int SKIP_RETURN_CODE = 77;

// Execution tests need a real device. On a machine without one they must skip
// with the reason stated rather than fail, so a continuous integration run on
// a GPU free runner stays honest instead of merely green.
inline bool cuda_device_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
}

#define SKIP_WITHOUT_GPU()                                                            \
    do {                                                                              \
        if (!::cps_test::cuda_device_available()) {                                   \
            GTEST_SKIP() << "hardware required: no CUDA device is visible, so this "  \
                            "execution test cannot run here";                         \
        }                                                                             \
    } while (false)

}  // namespace cps_test
