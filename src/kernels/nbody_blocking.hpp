#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "cps/physics/gravity.hpp"
#include "cps/utils/cuda_helpers.hpp"

// What the tiled and the fused kernel share: register blocking, the force sum over
// staged tiles, and the host dispatch over tile width, blocking factor and launch
// bounds. Private to these two files, so it is not installed.
//
// Register blocking gives each thread BF target particles instead of one, held in
// registers, against the one staged source tile. A source load then feeds BF pair
// evaluations, and no target's arithmetic depends on another's, so a warp has
// something to issue while one sum waits on a long latency result. Whether that wins
// anything on this kernel is a question for the sweep. The block covers BF * blockDim.x
// particles, so the grid shrinks by BF and each block is BF times the work, which also
// lowers how many warps an SM keeps busy.

namespace cps {

// The particles one thread owns during one pass of its block, and their running sums.
//
// Plain arrays, with every loop over them unrolled, so that they become registers. A
// loop left in place here would put them on the stack, which is a far slower kernel
// and looks otherwise normal.
template <int BF>
struct TargetBlock {
    float x[BF];
    float y[BF];
    float z[BF];
    float ax[BF];
    float ay[BF];
    float az[BF];
};

// The first particle of this thread in the pass that starts at block_first.
template <int BF>
__device__ __forceinline__ int first_target(int block_first) {
    return block_first + BF * static_cast<int>(threadIdx.x);
}

// Tail handling. A target past the end of the array sits at the origin and its sum
// is thrown away at the write out, because the thread still has to reach every
// barrier, and it may own live targets beside it. The caller masks each write on
// its own index and a thread never returns early.
template <int BF>
__device__ __forceinline__ TargetBlock<BF> load_targets(const float* __restrict__ x,
                                                        const float* __restrict__ y,
                                                        const float* __restrict__ z, int first,
                                                        int count) {
    TargetBlock<BF> targets;
#pragma unroll
    for (int k = 0; k < BF; ++k) {
        const int i = first + k;
        const bool active = i < count;
        targets.x[k] = active ? x[i] : 0.0f;
        targets.y[k] = active ? y[i] : 0.0f;
        targets.z[k] = active ? z[i] : 0.0f;
        targets.ax[k] = 0.0f;
        targets.ay[k] = 0.0f;
        targets.az[k] = 0.0f;
    }
    return targets;
}

// The whole force sum for one thread's targets.
//
// The block stages one tile of sources in shared memory at a time and every thread
// consumes it, which reads each tile from global memory once per block rather than
// once per thread. This is the part that has to stay the same at every BF:
//
// The j order for each i does not change. Tiles arrive in order and slots are read
// in order, and each target adds its own terms to its own sum in that order, so a
// target's sum is the same expression at every BF and the results are bit identical.
// The k loop only interleaves sums that never touch each other. Do not reassociate
// them and do not reorder the slots.
//
// The tile loop runs the same number of times on every thread, because it depends
// only on count, so all threads of the block reach the same barriers, whatever their
// targets are doing.
//
// Self interaction now falls on whichever of the BF targets is that particle, not on
// the thread's own index. Nothing depends on which: that one pair has a zero numerator
// and a softened denominator, so it adds zero. A target past the end sits at the origin
// and can meet a padded slot there too, which is the same pair of zeros over the same
// softening.
//
// Each pass of the slot loop unrolls TILE / BF slots, which is TILE pair evaluations
// and about the same code at every BF. Unrolling all TILE slots would multiply the
// code by BF, past what the instruction cache holds, and the compiler then takes far
// more registers for it.
template <int TILE, int BF>
__device__ __forceinline__ void accumulate_over_tiles(TargetBlock<BF>& targets,
                                                      const float* __restrict__ x,
                                                      const float* __restrict__ y,
                                                      const float* __restrict__ z,
                                                      const float* __restrict__ mass, int count,
                                                      float softening_sq) {
    __shared__ float tile_x[TILE];
    __shared__ float tile_y[TILE];
    __shared__ float tile_z[TILE];
    __shared__ float tile_mass[TILE];

    constexpr int SLOTS_PER_PASS = TILE / BF;

    for (int tile_start = 0; tile_start < count; tile_start += TILE) {
        // Guards the shared buffer against a thread that is still reading the
        // previous tile, or the last tile of this block's previous pass.
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
        // Guards every read below against a thread that has not staged its slots yet.
        __syncthreads();

#pragma unroll SLOTS_PER_PASS
        for (int t = 0; t < TILE; ++t) {
            const float xj = tile_x[t];
            const float yj = tile_y[t];
            const float zj = tile_z[t];
            const float mj = tile_mass[t];
#pragma unroll
            for (int k = 0; k < BF; ++k) {
                accumulate_pair_acceleration(targets.x[k], targets.y[k], targets.z[k], xj, yj, zj,
                                             mj, softening_sq, targets.ax[k], targets.ay[k],
                                             targets.az[k]);
            }
        }
    }
}

// The block size that gets its own launch bounds, in every tile width and blocking
// factor. Only the default is shipped: each bound multiplies the instantiations
// across tile widths, blocking factors, two kernels, two libraries and six
// architectures. Every other size runs the unbounded instantiation, and nothing
// here refuses a size for lacking a bound. A smaller size does not borrow this bound
// either, because the compiler can read a bound as room for more registers, and the
// unbounded instantiation is what those sizes have always run.
inline constexpr int BOUNDED_BLOCK_SIZE = 256;

// The blocks a launch uses. A block covers BF * block_size particles, so that is what
// the grid is sized for. max_grid caps it, and a block then makes more than one pass
// through the stride loop in the kernels.
inline int blocked_grid(int count, int blocking_factor, int block_size, int max_grid) {
    const int grid = grid_for(count, blocking_factor * block_size);
    return max_grid > 0 ? std::min(grid, max_grid) : grid;
}

// A blocking factor brings a new way for a launch to fail: BF targets need more
// registers per thread, and a whole block of threads has to fit on one SM. The runtime
// would only say too many resources were requested, which names neither knob, so this
// asks first. Which pairs fit depends on what the compiler made of the kernel on this
// architecture, so it is found out here and not when the configuration is read. A block
// no larger than the bounded size is far from the limit at every factor, so the launches
// that matter do not make the call.
template <typename Kernel>
void require_block_fits(Kernel* kernel, int blocking_factor, int block_size) {
    if (block_size <= BOUNDED_BLOCK_SIZE) {
        return;
    }
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    if (block_size > attributes.maxThreadsPerBlock) {
        throw CudaError("block_size " + std::to_string(block_size) + " with blocking factor " +
                        std::to_string(blocking_factor) +
                        " does not fit on this device: that kernel needs more registers than an "
                        "SM has for a block over " +
                        std::to_string(attributes.maxThreadsPerBlock) +
                        " threads, so use a smaller block_size or blocking factor");
    }
}

// Launch is a host side object with two member templates, one per launch shape, which
// perform the launch: bounded<TILE, BF, BLOCK>() and unbounded<TILE, BF>().
template <int TILE, int BF, typename Launch>
void launch_at_block_size(const Launch& launch, int block_size) {
    if (block_size == BOUNDED_BLOCK_SIZE) {
        launch.template bounded<TILE, BF, BOUNDED_BLOCK_SIZE>();
    } else {
        launch.template unbounded<TILE, BF>();
    }
}

template <int TILE, typename Launch>
void launch_at_blocking_factor(const Launch& launch, int blocking_factor, int block_size) {
    switch (blocking_factor) {
        case 1:
            launch_at_block_size<TILE, 1>(launch, block_size);
            break;
        case 2:
            launch_at_block_size<TILE, 2>(launch, block_size);
            break;
        case 4:
            launch_at_block_size<TILE, 4>(launch, block_size);
            break;
        case 8:
            launch_at_block_size<TILE, 8>(launch, block_size);
            break;
        default:
            throw std::runtime_error("unsupported blocking factor " +
                                     std::to_string(blocking_factor) +
                                     ", expected 1, 2, 4 or 8");
    }
}

template <typename Launch>
void launch_nbody_kernel(const Launch& launch, int tile_size, int blocking_factor,
                         int block_size) {
    switch (tile_size) {
        case 32:
            launch_at_blocking_factor<32>(launch, blocking_factor, block_size);
            break;
        case 64:
            launch_at_blocking_factor<64>(launch, blocking_factor, block_size);
            break;
        case 128:
            launch_at_blocking_factor<128>(launch, blocking_factor, block_size);
            break;
        case 256:
            launch_at_blocking_factor<256>(launch, blocking_factor, block_size);
            break;
        default:
            throw std::runtime_error("unsupported tile size " + std::to_string(tile_size) +
                                     ", expected 32, 64, 128 or 256");
    }
}

}  // namespace cps
