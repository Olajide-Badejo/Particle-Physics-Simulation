#pragma once

#include <cuda_runtime.h>

#include <cstddef>
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

// With CPS_DEBUG_SYNC on, every launcher and copy drains its stream before it
// returns, so an ordering bug can be bisected: if a failure goes away in that
// build, something was reading ahead of the stream. Off, which is the default,
// this expands to no code at all. A macro rather than a function so that a
// launch failure caught here names the launcher's own line.
#if defined(CPS_DEBUG_SYNC) && CPS_DEBUG_SYNC
#define CPS_SYNC_POINT(stream) CUDA_CHECK(cudaStreamSynchronize(stream))
#else
#define CPS_SYNC_POINT(stream) static_cast<void>(stream)
#endif

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

// Owning CUDA stream. It is created non-blocking so that nothing can lean on the
// legacy default stream's implicit synchronisation: the default stream waits for
// every blocking stream and they wait for it, and a non-blocking stream takes
// part in neither. Work here is ordered against other work on this stream and
// against nothing else, so a missing synchronisation is a bug that shows up now
// rather than one that hides until the work is captured into a graph.
//
// get() on a moved-from Stream returns nullptr, which CUDA reads as the default
// stream, so a moved-from Stream is not to be used.
class Stream {
   public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    // Destroying a stream does not wait for its work, so whoever owns buffers
    // that work touches has to drain it before freeing them.
    ~Stream() { release(); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            release();
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }

    // Blocks until everything enqueued so far has finished, and reports the
    // failure of any of it.
    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

   private:
    void release() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

// Thin wrapper over the CUDA event pair used for all kernel timing. Events are
// recorded on the same stream as the work, so they measure device time rather
// than host side dispatch overhead. The stream has no default: a timer on the
// default stream while the work runs on another would measure nothing, without
// any error to say so.
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

    void start(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Returns elapsed milliseconds.
    float stop(cudaStream_t stream) {
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

    // Returns with the bytes written. cudaMemset can return first, and it runs
    // on the legacy default stream, which a non-blocking stream is not ordered
    // against, so without the wait a kernel on another stream could read this
    // buffer before it is zero, or have its result zeroed afterwards.
    void zero() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaMemset(data_, 0, count_ * sizeof(T)));
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
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

// Owning pinned host buffer, the staging end of a stream ordered copy. A copy
// from pageable memory is free to finish before it returns or to stage the data
// itself, depending on its size; a copy from pinned memory is a plain entry in
// the stream, which is what lets it be ordered with the launches around it.
template <typename T>
class HostBuffer {
   public:
    HostBuffer() = default;

    explicit HostBuffer(std::size_t count) { allocate(count); }

    ~HostBuffer() { release(); }

    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    HostBuffer(HostBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
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
            CUDA_CHECK(cudaMallocHost(&data_, count * sizeof(T)));
            count_ = count;
        }
    }

    void release() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

   private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Stream ordered copies of count elements. Both return once the copy is enqueued,
// so the host range has to stay alive and untouched until the stream has passed
// it, and a read of copy_to_host_async's result has to wait for that.
template <typename T>
void copy_to_device_async(T* device, const T* host, std::size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    CPS_SYNC_POINT(stream);
}

template <typename T>
void copy_to_host_async(T* host, const T* device, std::size_t count, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CPS_SYNC_POINT(stream);
}

}  // namespace cps
