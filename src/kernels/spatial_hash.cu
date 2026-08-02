#include <cub/device/device_radix_sort.cuh>

#include "physics/spatial_hash.hpp"
#include "solvers/sph_passes.hpp"
#include "utils/cuda_helpers.hpp"

namespace cps {
namespace {

__global__ void hash_particles_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                      const float* __restrict__ z, uint32_t* __restrict__ bucket,
                                      int64_t* __restrict__ cell_key, uint32_t* __restrict__ index,
                                      int count, float cell_size, uint32_t table_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int ix = cell_coordinate(x[i], cell_size);
    const int iy = cell_coordinate(y[i], cell_size);
    const int iz = cell_coordinate(z[i], cell_size);
    bucket[i] = spatial_hash(ix, iy, iz, table_size);
    cell_key[i] = cps::cell_key(ix, iy, iz);
    index[i] = static_cast<uint32_t>(i);
}

__global__ void clear_cell_bounds_kernel(uint32_t* __restrict__ cell_start,
                                         uint32_t* __restrict__ cell_end, uint32_t table_size) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) {
        return;
    }
    cell_start[i] = CELL_EMPTY;
    cell_end[i] = CELL_EMPTY;
}

// First and last index of every occupied bucket in the sorted array. A thread
// compares its bucket with its predecessor: a change marks the start of one
// run and the end of the previous one.
__global__ void cell_bounds_kernel(const uint32_t* __restrict__ bucket_sorted,
                                   uint32_t* __restrict__ cell_start,
                                   uint32_t* __restrict__ cell_end, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const uint32_t here = bucket_sorted[i];
    if (i == 0) {
        cell_start[here] = 0u;
    } else {
        const uint32_t previous = bucket_sorted[i - 1];
        if (previous != here) {
            cell_start[here] = static_cast<uint32_t>(i);
            cell_end[previous] = static_cast<uint32_t>(i);
        }
    }
    if (i == count - 1) {
        cell_end[here] = static_cast<uint32_t>(count);
    }
}

__global__ void gather_sorted_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const int64_t* __restrict__ cell_key,
    const uint32_t* __restrict__ index_sorted, float* __restrict__ sx, float* __restrict__ sy,
    float* __restrict__ sz, float* __restrict__ svx, float* __restrict__ svy,
    float* __restrict__ svz, float* __restrict__ smass, int64_t* __restrict__ cell_key_sorted,
    int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }
    const uint32_t source = index_sorted[k];
    sx[k] = x[source];
    sy[k] = y[source];
    sz[k] = z[source];
    svx[k] = vx[source];
    svy[k] = vy[source];
    svz[k] = vz[source];
    smass[k] = mass[source];
    cell_key_sorted[k] = cell_key[source];
}

__global__ void count_collisions_kernel(const uint32_t* __restrict__ bucket_sorted,
                                        const int64_t* __restrict__ cell_key_sorted,
                                        int* __restrict__ counter, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= count) {
        return;
    }
    // Neighbouring entries in the sorted array that share a bucket but not a
    // cell are exactly the particles a naive bucket scan would confuse.
    if (bucket_sorted[i] == bucket_sorted[i - 1] && cell_key_sorted[i] != cell_key_sorted[i - 1]) {
        atomicAdd(counter, 1);
    }
}

// Only the bits that can actually differ need sorting. The table size bounds
// the bucket index, so the radix sort can stop there instead of walking all
// thirty two bits, which is four passes instead of eight at typical sizes.
int significant_bits(uint32_t table_size) {
    int bits = 0;
    uint32_t value = table_size - 1u;
    while (value > 0u) {
        ++bits;
        value >>= 1;
    }
    return bits > 0 ? bits : 1;
}

}  // namespace

void sph_hash_particles(const float* x, const float* y, const float* z, uint32_t* bucket,
                        int64_t* cell_key, uint32_t* index, int count, float cell_size,
                        uint32_t table_size, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    hash_particles_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        x, y, z, bucket, cell_key, index, count, cell_size, table_size);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t sph_sort_temp_bytes(int count, uint32_t table_size) {
    if (count <= 0) {
        return 0;
    }
    std::size_t bytes = 0;
    const int end_bit = significant_bits(table_size);
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                    static_cast<uint32_t*>(nullptr),
                                    static_cast<const uint32_t*>(nullptr),
                                    static_cast<uint32_t*>(nullptr), count, 0, end_bit);
    return bytes;
}

void sph_sort_by_bucket(void* temp, std::size_t temp_bytes, const uint32_t* bucket_in,
                        uint32_t* bucket_out, const uint32_t* index_in, uint32_t* index_out,
                        int count, uint32_t table_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int end_bit = significant_bits(table_size);
    std::size_t bytes = temp_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, bytes, bucket_in, bucket_out, index_in,
                                               index_out, count, 0, end_bit, stream));
}

void sph_clear_cell_bounds(uint32_t* cell_start, uint32_t* cell_end, uint32_t table_size,
                           int block_size, cudaStream_t stream) {
    const int cells = static_cast<int>(table_size);
    clear_cell_bounds_kernel<<<grid_for(cells, block_size), block_size, 0, stream>>>(
        cell_start, cell_end, table_size);
    CUDA_CHECK(cudaGetLastError());
}

void sph_compute_cell_bounds(const uint32_t* bucket_sorted, uint32_t* cell_start,
                             uint32_t* cell_end, int count, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    cell_bounds_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        bucket_sorted, cell_start, cell_end, count);
    CUDA_CHECK(cudaGetLastError());
}

void sph_gather_sorted(const float* x, const float* y, const float* z, const float* vx,
                       const float* vy, const float* vz, const float* mass, const int64_t* cell_key,
                       const uint32_t* index_sorted, float* sx, float* sy, float* sz, float* svx,
                       float* svy, float* svz, float* smass, int64_t* cell_key_sorted, int count,
                       int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    gather_sorted_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        x, y, z, vx, vy, vz, mass, cell_key, index_sorted, sx, sy, sz, svx, svy, svz, smass,
        cell_key_sorted, count);
    CUDA_CHECK(cudaGetLastError());
}

void sph_count_bucket_collisions(const uint32_t* bucket_sorted, const int64_t* cell_key_sorted,
                                 int* counter, int count, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(int), stream));
    count_collisions_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        bucket_sorted, cell_key_sorted, counter, count);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cps
