#include "cps/utils/cuda_helpers.hpp"

#include <sstream>

namespace cps {

DeviceInfo query_device(int device) {
    DeviceInfo info;
    info.device = device;

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.multiprocessor_count = properties.multiProcessorCount;
    info.memory_bus_width_bits = properties.memoryBusWidth;
    info.total_global_memory = properties.totalGlobalMem;
    info.shared_memory_per_block = properties.sharedMemPerBlock;
    info.max_threads_per_block = properties.maxThreadsPerBlock;
    info.warp_size = properties.warpSize;

    // CUDA 13 removed clockRate and memoryClockRate from cudaDeviceProp. Both
    // are still available through the attribute interface, which is what the
    // deprecation notice pointed at.
    CUDA_CHECK(cudaDeviceGetAttribute(&info.clock_khz, cudaDevAttrClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&info.memory_clock_khz, cudaDevAttrMemoryClockRate, device));

    // Double data rate, bus width in bytes, kilohertz to hertz, bytes to
    // gigabytes. This is the denominator of every bandwidth utilisation figure
    // in the report, so it is read from the device rather than quoted from a
    // data sheet.
    info.peak_bandwidth_gb_s = 2.0 * static_cast<double>(info.memory_clock_khz) * 1.0e3 *
                               (static_cast<double>(info.memory_bus_width_bits) / 8.0) / 1.0e9;

    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

std::string device_info_to_json(const DeviceInfo& info) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"device\": " << info.device << ",\n";
    out << "  \"name\": \"" << info.name << "\",\n";
    out << "  \"compute_capability\": \"" << info.compute_major << "." << info.compute_minor
        << "\",\n";
    out << "  \"multiprocessor_count\": " << info.multiprocessor_count << ",\n";
    out << "  \"clock_khz\": " << info.clock_khz << ",\n";
    out << "  \"memory_clock_khz\": " << info.memory_clock_khz << ",\n";
    out << "  \"memory_bus_width_bits\": " << info.memory_bus_width_bits << ",\n";
    out << "  \"peak_bandwidth_gb_s\": " << info.peak_bandwidth_gb_s << ",\n";
    out << "  \"total_global_memory_bytes\": " << info.total_global_memory << ",\n";
    out << "  \"shared_memory_per_block_bytes\": " << info.shared_memory_per_block << ",\n";
    out << "  \"max_threads_per_block\": " << info.max_threads_per_block << ",\n";
    out << "  \"warp_size\": " << info.warp_size << ",\n";
    out << "  \"cuda_runtime_version\": " << info.runtime_version << ",\n";
    out << "  \"cuda_driver_version\": " << info.driver_version << "\n";
    out << "}\n";
    return out.str();
}

void print_device_info(const DeviceInfo& info, std::FILE* stream) {
    std::fprintf(stream, "device %d: %s (compute capability %d.%d)\n", info.device,
                 info.name.c_str(), info.compute_major, info.compute_minor);
    std::fprintf(stream, "  multiprocessors      : %d\n", info.multiprocessor_count);
    std::fprintf(stream, "  core clock           : %.0f MHz\n", info.clock_khz / 1.0e3);
    std::fprintf(stream, "  memory clock         : %.0f MHz effective\n",
                 info.memory_clock_khz / 1.0e3);
    std::fprintf(stream, "  memory bus           : %d bits\n", info.memory_bus_width_bits);
    std::fprintf(stream, "  peak bandwidth       : %.1f GB/s\n", info.peak_bandwidth_gb_s);
    std::fprintf(stream, "  global memory        : %.2f GiB\n",
                 static_cast<double>(info.total_global_memory) / (1024.0 * 1024.0 * 1024.0));
    std::fprintf(stream, "  shared memory/block  : %zu bytes\n", info.shared_memory_per_block);
    std::fprintf(stream, "  max threads/block    : %d\n", info.max_threads_per_block);
    std::fprintf(stream, "  cuda runtime/driver  : %d / %d\n", info.runtime_version,
                 info.driver_version);
}

}  // namespace cps
