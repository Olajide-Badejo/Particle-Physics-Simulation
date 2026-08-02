#include "physics/gravity.hpp"
#include "solvers/nbody_solver.hpp"
#include "utils/cuda_helpers.hpp"

namespace cps {
namespace {

// Rung one of the ladder: every thread walks the whole particle array in
// global memory.
//
// This is not a straw man. All threads in a warp read the same j on the same
// cycle, so the load is a broadcast and the line stays hot in L1 for the
// entire warp. What it does not do is share that line between the warps of a
// block, or between blocks, so the same data crosses the L2 boundary once per
// resident block rather than once per tile. The tiled rung measures exactly
// that difference.
__global__ void compute_forces_naive_kernel(const float* __restrict__ x,
                                            const float* __restrict__ y,
                                            const float* __restrict__ z,
                                            const float* __restrict__ mass,
                                            float* __restrict__ ax_out, float* __restrict__ ay_out,
                                            float* __restrict__ az_out, int count,
                                            float softening_sq, float gravitational_constant) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const float xi = x[i];
    const float yi = y[i];
    const float zi = z[i];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int j = 0; j < count; ++j) {
        accumulate_pair_acceleration(xi, yi, zi, x[j], y[j], z[j], mass[j], softening_sq, ax, ay,
                                     az);
    }

    ax_out[i] = gravitational_constant * ax;
    ay_out[i] = gravitational_constant * ay;
    az_out[i] = gravitational_constant * az;
}

}  // namespace

void compute_forces_naive(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = grid_for(count, block_size);
    compute_forces_naive_kernel<<<grid, block_size, 0, stream>>>(
        x, y, z, mass, ax, ay, az, count, softening * softening, gravitational_constant);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cps
