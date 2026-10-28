#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace cps {

class CudaError : public std::runtime_error {
   public:
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

// Every CUDA entry point returns a status and almost every real bug in this
// project first showed up as a status that nobody looked at. The macro turns
// a bad status into an exception carrying the call site, which is what makes
// an asynchronous launch failure findable at all.
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t cps_status_ = (call);                                                \
        if (cps_status_ != cudaSuccess) {                                                      \
            throw ::cps::CudaError(std::string(#call) + " failed at " + __FILE__ + ":" +       \
                                   std::to_string(__LINE__) + ": " +                           \
                                   cudaGetErrorString(cps_status_));                           \
        }                                                                                      \
    } while (false)

struct DeviceInfo {
    int device = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    int clock_khz = 0;
    int memory_clock_khz = 0;
    int memory_bus_width_bits = 0;
    std::size_t total_global_memory = 0;
    std::size_t shared_memory_per_block = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    double peak_bandwidth_gb_s = 0.0;
    int runtime_version = 0;
    int driver_version = 0;
};

// CUDA 13 removed clockRate and memoryClockRate from cudaDeviceProp; both are
// read through cudaDeviceGetAttribute instead. The peak bandwidth derived here
// is the figure every bandwidth utilisation percentage in the report is
// measured against, so it is queried rather than hard coded.
DeviceInfo query_device(int device = 0);

std::string device_info_to_json(const DeviceInfo& info);

void print_device_info(const DeviceInfo& info, std::FILE* stream);

// Number of blocks needed to cover n items with the given block size.
inline int grid_for(int n, int block_size) { return (n + block_size - 1) / block_size; }

// Thin wrapper over the CUDA event pair used for all kernel timing. Events are
// recorded on the same stream as the work, so they measure device time rather
// than host side dispatch overhead.
class CudaTimer {
   public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Returns elapsed milliseconds.
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Owning device buffer. Small enough to be obvious, which matters more here
// than generality.
template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
            count_ = count;
        }
    }

    void release() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    void zero() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T)));
        }
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace cps
