#include <stdexcept>
#include <string>

#include "cps/physics/gravity.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {
namespace {

// Rung two: the block stages a tile of source particles in shared memory and
// every thread of the block consumes it.
//
// The structure follows Nyland, Harris and Prins, GPU Gems 3 chapter 31. Each
// tile is read from global memory once per block instead of once per thread,
// so the global traffic for the whole sum falls by roughly the block size.
// The price is two barriers per tile, which is why the tile width matters:
// narrow tiles pay the barrier more often per unit of arithmetic.
//
// The barrier cost against reuse tradeoff is an empirical question, so the
// default tile width comes from the sweep in the debug report, and the other
// instantiations exist so that sweep can keep answering it.
//
// Threads with an index past the end of the array must not return early. They
// still have to reach both barriers on every iteration, so they run the whole
// loop with clamped reads and simply discard the result at the end.
template <int TILE>
__global__ void compute_forces_tiled_kernel(const float* __restrict__ x,
                                            const float* __restrict__ y,
                                            const float* __restrict__ z,
                                            const float* __restrict__ mass,
                                            float* __restrict__ ax_out, float* __restrict__ ay_out,
                                            float* __restrict__ az_out, int count,
                                            float softening_sq, float gravitational_constant) {
    __shared__ float tile_x[TILE];
    __shared__ float tile_y[TILE];
    __shared__ float tile_z[TILE];
    __shared__ float tile_mass[TILE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;

    const float xi = active ? x[i] : 0.0f;
    const float yi = active ? y[i] : 0.0f;
    const float zi = active ? z[i] : 0.0f;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile_start = 0; tile_start < count; tile_start += TILE) {
        // Guards the shared buffer against a thread that is still reading the
        // previous tile.
        __syncthreads();
        for (int t = static_cast<int>(threadIdx.x); t < TILE; t += static_cast<int>(blockDim.x)) {
            const int source = tile_start + t;
            if (source < count) {
                tile_x[t] = x[source];
                tile_y[t] = y[source];
                tile_z[t] = z[source];
                tile_mass[t] = mass[source];
            } else {
                // A padded slot carries zero mass, so it contributes nothing
                // and the inner loop needs no bounds test.
                tile_x[t] = 0.0f;
                tile_y[t] = 0.0f;
                tile_z[t] = 0.0f;
                tile_mass[t] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            accumulate_pair_acceleration(xi, yi, zi, tile_x[t], tile_y[t], tile_z[t], tile_mass[t],
                                         softening_sq, ax, ay, az);
        }
    }

    if (active) {
        ax_out[i] = gravitational_constant * ax;
        ay_out[i] = gravitational_constant * ay;
        az_out[i] = gravitational_constant * az;
    }
}

}  // namespace

void compute_forces_tiled(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size, int tile_size,
                          cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = grid_for(count, block_size);
    const float softening_sq = softening * softening;

    switch (tile_size) {
        case 32:
            compute_forces_tiled_kernel<32><<<grid, block_size, 0, stream>>>(
                x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
            break;
        case 64:
            compute_forces_tiled_kernel<64><<<grid, block_size, 0, stream>>>(
                x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
            break;
        case 128:
            compute_forces_tiled_kernel<128><<<grid, block_size, 0, stream>>>(
                x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
            break;
        case 256:
            compute_forces_tiled_kernel<256><<<grid, block_size, 0, stream>>>(
                x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
            break;
        default:
            throw std::runtime_error("unsupported tile size " + std::to_string(tile_size) +
                                     ", expected 32, 64, 128 or 256");
    }
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

}  // namespace cps
