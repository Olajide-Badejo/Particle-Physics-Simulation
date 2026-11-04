#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cps/utils/cuda_helpers.hpp"

namespace {

// A pure streaming kernel. Two arrays, one read and one write, no arithmetic
// worth speaking of, grid stride so occupancy rather than index arithmetic
// decides the rate. What comes out is the bandwidth this card actually
// delivers to a well behaved kernel, which is the honest denominator for a
// utilisation figure. Dividing by the theoretical peak instead would flatter
// every kernel in the report by the same fifteen percent or so.
__global__ void stream_copy_kernel(const float4* __restrict__ source, float4* __restrict__ target,
                                   long long count) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        target[i] = source[i];
    }
}

double measure_stream_bandwidth(const cps::DeviceInfo& info) {
    // Half a gigabyte per array is far past any cache on this part.
    const long long bytes_per_array = 512LL * 1024 * 1024;
    const long long vector_count = bytes_per_array / static_cast<long long>(sizeof(float4));

    cps::DeviceBuffer<float4> source(static_cast<std::size_t>(vector_count));
    cps::DeviceBuffer<float4> target(static_cast<std::size_t>(vector_count));
    source.zero();
    target.zero();

    const int block = 256;
    const int grid = info.multiprocessor_count * 32;

    // Warm up, so the measurement does not include the first touch of the
    // pages or the initial clock ramp.
    for (int i = 0; i < 3; ++i) {
        stream_copy_kernel<<<grid, block>>>(source.get(), target.get(), vector_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int repeats = 20;
    cps::CudaTimer timer;
    timer.start(nullptr);
    for (int i = 0; i < repeats; ++i) {
        stream_copy_kernel<<<grid, block>>>(source.get(), target.get(), vector_count);
    }
    const float elapsed_ms = timer.stop(nullptr);
    CUDA_CHECK(cudaGetLastError());

    const double moved_bytes = 2.0 * static_cast<double>(bytes_per_array) * repeats;
    return moved_bytes / (static_cast<double>(elapsed_ms) * 1.0e-3) / 1.0e9;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output_path = "experiments/results/device.json";
        bool run_bandwidth = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (arg == "--no-bandwidth") {
                run_bandwidth = false;
            } else if (arg == "--help") {
                std::printf("gpu_probe [--output <file.json>] [--no-bandwidth]\n");
                return 0;
            }
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::fprintf(stderr, "no CUDA device is visible\n");
            return 1;
        }

        const cps::DeviceInfo info = cps::query_device(0);
        cps::print_device_info(info, stdout);

        double achieved = 0.0;
        if (run_bandwidth) {
            achieved = measure_stream_bandwidth(info);
            std::printf("  measured copy rate   : %.1f GB/s (%.1f%% of peak)\n", achieved,
                        100.0 * achieved / info.peak_bandwidth_gb_s);
        }

        std::string json = cps::device_info_to_json(info);
        // Splice the measured figure in before the closing brace.
        const std::size_t brace = json.rfind('}');
        if (brace != std::string::npos) {
            char extra[128];
            std::snprintf(extra, sizeof(extra), ",\n  \"measured_copy_gb_s\": %.4f\n", achieved);
            json.insert(brace, extra);
        }

        const std::filesystem::path path(output_path);
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream stream(path);
        if (!stream) {
            std::fprintf(stderr, "cannot write %s\n", output_path.c_str());
            return 1;
        }
        stream << json;
        std::printf("\nwrote %s\n", output_path.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
