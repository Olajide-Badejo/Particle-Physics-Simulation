#include <gtest/gtest.h>

#include <cstdio>

#include "test_support_gpu.hpp"

// Entry point for the test binaries that need a device, in place of gtest_main.
//
// Without a device every case in these binaries skips, and gtest reports a run
// in which everything skipped as a success, so ctest would print "Passed" for
// tests that never ran. Returning SKIP_RETURN_CODE instead lets ctest report
// the whole binary as "Not Run". Listing the tests still works with no device,
// because it executes nothing. On a machine with a device the binary runs
// normally, and a case that skips for its own reason still skips on its own.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!GTEST_FLAG_GET(list_tests) && !cps_test::cuda_device_available()) {
        std::fprintf(stderr,
                     "hardware required: no CUDA device is visible, so none of these tests "
                     "can run here; exiting with code %d (skipped)\n",
                     cps_test::SKIP_RETURN_CODE);
        return cps_test::SKIP_RETURN_CODE;
    }
    return RUN_ALL_TESTS();
}
