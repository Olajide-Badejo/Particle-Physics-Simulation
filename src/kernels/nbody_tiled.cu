#include "cps/physics/gravity.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "cps/utils/cuda_helpers.hpp"
#include "nbody_blocking.hpp"

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
// loop with clamped reads and simply discard the result at the end. With BF
// targets on a thread some can be live and some not, so each write has its own
// mask.
//
// The outer loop is a grid stride. A block starts at its own multiple of the
// BF * blockDim.x particles it covers and steps by that times the grid, so any
// grid covers every particle exactly once. It depends only on block and grid
// values, so all the threads of a block make the same passes and reach the same
// barriers. On the grid the launcher sizes for count, each block makes one pass.
template <int TILE, int BF>
__device__ __forceinline__ void tiled_force_sum(const float* __restrict__ x,
                                                const float* __restrict__ y,
                                                const float* __restrict__ z,
                                                const float* __restrict__ mass,
                                                float* __restrict__ ax_out,
                                                float* __restrict__ ay_out,
                                                float* __restrict__ az_out, int count,
                                                float softening_sq, float gravitational_constant) {
    const int block_span = BF * static_cast<int>(blockDim.x);
    const int stride = block_span * static_cast<int>(gridDim.x);
    for (int block_first = static_cast<int>(blockIdx.x) * block_span; block_first < count;
         block_first += stride) {
        const int first = first_target<BF>(block_first);
        TargetBlock<BF> targets = load_targets<BF>(x, y, z, first, count);
        accumulate_over_tiles<TILE, BF>(targets, x, y, z, mass, count, softening_sq);

#pragma unroll
        for (int k = 0; k < BF; ++k) {
            const int i = first + k;
            if (i < count) {
                ax_out[i] = gravitational_constant * targets.ax[k];
                ay_out[i] = gravitational_constant * targets.ay[k];
                az_out[i] = gravitational_constant * targets.az[k];
            }
        }
    }
}

// Launch bounds let the compiler fit its register allocation to the block launched.
template <int TILE, int BF, int BLOCK>
__global__ __launch_bounds__(BLOCK) void compute_forces_tiled_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ mass, float* __restrict__ ax_out, float* __restrict__ ay_out,
    float* __restrict__ az_out, int count, float softening_sq, float gravitational_constant) {
    tiled_force_sum<TILE, BF>(x, y, z, mass, ax_out, ay_out, az_out, count, softening_sq,
                              gravitational_constant);
}

// Every block size with no bound of its own.
template <int TILE, int BF>
__global__ void compute_forces_tiled_unbounded_kernel(
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ mass, float* __restrict__ ax_out, float* __restrict__ ay_out,
    float* __restrict__ az_out, int count, float softening_sq, float gravitational_constant) {
    tiled_force_sum<TILE, BF>(x, y, z, mass, ax_out, ay_out, az_out, count, softening_sq,
                              gravitational_constant);
}

struct TiledLaunch {
    const float* x;
    const float* y;
    const float* z;
    const float* mass;
    float* ax;
    float* ay;
    float* az;
    int count;
    float softening_sq;
    float gravitational_constant;
    int block_size;
    int max_grid;
    cudaStream_t stream;

    template <int TILE, int BF, int BLOCK>
    void bounded() const {
        const int grid = blocked_grid(count, BF, block_size, max_grid);
        compute_forces_tiled_kernel<TILE, BF, BLOCK><<<grid, block_size, 0, stream>>>(
            x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
        CUDA_CHECK(cudaGetLastError());
    }

    template <int TILE, int BF>
    void unbounded() const {
        require_block_fits(compute_forces_tiled_unbounded_kernel<TILE, BF>, BF, block_size);
        const int grid = blocked_grid(count, BF, block_size, max_grid);
        compute_forces_tiled_unbounded_kernel<TILE, BF><<<grid, block_size, 0, stream>>>(
            x, y, z, mass, ax, ay, az, count, softening_sq, gravitational_constant);
        CUDA_CHECK(cudaGetLastError());
    }
};

}  // namespace

void compute_forces_tiled(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size, int tile_size,
                          int blocking_factor, cudaStream_t stream, int max_grid) {
    if (count <= 0) {
        return;
    }
    const float softening_sq = softening * softening;
    const TiledLaunch launch{x,  y,  z,     mass,         ax,
                             ay, az, count, softening_sq, gravitational_constant,
                             block_size, max_grid, stream};
    launch_nbody_kernel(launch, tile_size, blocking_factor, block_size);
    CPS_SYNC_POINT(stream);
}

}  // namespace cps
