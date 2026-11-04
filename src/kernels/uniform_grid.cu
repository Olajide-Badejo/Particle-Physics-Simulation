#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>

#include "cps/physics/uniform_grid.hpp"
#include "cps/solvers/sph_passes.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {
namespace {

__global__ void assign_cells_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                    const float* __restrict__ z, uint32_t* __restrict__ cell,
                                    uint32_t* __restrict__ index, int* __restrict__ non_finite_flag,
                                    int count, UniformGrid grid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float px = x[i];
    const float py = y[i];
    const float pz = z[i];
    index[i] = static_cast<uint32_t>(i);
    // The atomic sits in the failure branch, so a healthy step pays for the
    // test and nothing else.
    if (!position_is_finite(px, py, pz)) {
        cell[i] = reserved_slot(grid);
        atomicOr(non_finite_flag, 1);
        return;
    }
    cell[i] = cell_index(grid, px, py, pz);
}

// Where each cell's run begins in the sorted array, as a prefix sum: entry c holds
// the number of particles in cells below c, so cells c0 to c1 are the range from
// entry c0 to entry c1 + 1, and an empty cell is an entry equal to the next one.
// Nothing downstream has to test for an empty cell, and the table needs no clearing.
//
// Each entry is found by its own binary search of the sorted keys, so each entry has
// one writer and the threads do even work. Filling the gaps from the particles' side
// instead is less work in total, but one thread writes each run of empty cells, and
// in a sparsely filled domain one run can be most of the table.
__global__ void cell_bounds_kernel(const uint32_t* __restrict__ cell_sorted,
                                   uint32_t* __restrict__ cell_start, int count,
                                   uint32_t num_cells) {
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell > num_cells) {
        return;
    }
    uint32_t low = 0u;
    uint32_t high = static_cast<uint32_t>(count);
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2u;
        if (cell_sorted[middle] < cell) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    cell_start[cell] = low;
}

__global__ void gather_sorted_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    const float* __restrict__ mass, const uint32_t* __restrict__ id,
    const uint32_t* __restrict__ index_sorted, float* __restrict__ sx, float* __restrict__ sy,
    float* __restrict__ sz, float* __restrict__ svx, float* __restrict__ svy,
    float* __restrict__ svz, float* __restrict__ smass, uint32_t* __restrict__ sid, int count) {
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
    sid[k] = id[source];
}

}  // namespace

void sph_assign_cells(const float* x, const float* y, const float* z, uint32_t* cell,
                      uint32_t* index, int* non_finite_flag, int count, const UniformGrid& grid,
                      int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    assign_cells_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        x, y, z, cell, index, non_finite_flag, count, grid);
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

std::size_t sph_sort_temp_bytes(int count, uint32_t num_cells) {
    if (count <= 0) {
        return 0;
    }
    std::size_t bytes = 0;
    const int end_bit = sort_key_bits(num_cells);
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const uint32_t*>(nullptr),
                                    static_cast<uint32_t*>(nullptr),
                                    static_cast<const uint32_t*>(nullptr),
                                    static_cast<uint32_t*>(nullptr), count, 0, end_bit);
    return bytes;
}

void sph_sort_by_cell(void* temp, std::size_t temp_bytes, const uint32_t* cell_in,
                      uint32_t* cell_out, const uint32_t* index_in, uint32_t* index_out, int count,
                      uint32_t num_cells, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int end_bit = sort_key_bits(num_cells);
    std::size_t bytes = temp_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, bytes, cell_in, cell_out, index_in, index_out,
                                               count, 0, end_bit, stream));
    CPS_SYNC_POINT(stream);
}

void sph_compute_cell_bounds(const uint32_t* cell_sorted, uint32_t* cell_start, int count,
                             uint32_t num_cells, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    // In 64 bits and not through grid_for, because the table can have more entries
    // than an int can count.
    const uint64_t block = static_cast<uint64_t>(block_size);
    const uint64_t blocks = (cell_table_entries(num_cells) + block - 1u) / block;
    cell_bounds_kernel<<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(
        cell_sorted, cell_start, count, num_cells);
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

void sph_gather_sorted(const float* x, const float* y, const float* z, const float* vx,
                       const float* vy, const float* vz, const float* mass, const uint32_t* id,
                       const uint32_t* index_sorted, float* sx, float* sy, float* sz, float* svx,
                       float* svy, float* svz, float* smass, uint32_t* sid, int count,
                       int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    gather_sorted_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        x, y, z, vx, vy, vz, mass, id, index_sorted, sx, sy, sz, svx, svy, svz, smass, sid, count);
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

void allocate_cell_table(DeviceBuffer<uint32_t>& table, std::size_t entries,
                         const std::string& grid_description) {
    try {
        table.allocate(entries);
    } catch (const CudaError& error) {
        // A failed allocation stays in the last error, where the next launch on this
        // thread would take it for its own.
        static_cast<void>(cudaGetLastError());
        throw std::runtime_error("could not allocate the cell table of the fluid grid, " +
                                 grid_description +
                                 ", on the device; raise the smoothing radius or shrink the "
                                 "domain. The device reported: " +
                                 error.what());
    }
}

}  // namespace cps
