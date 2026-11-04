#include "cps/physics/gravity.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "cps/utils/cuda_helpers.hpp"
#include "nbody_blocking.hpp"

namespace cps {
namespace {

// Rung three: the tiled force sum and the integrator update in one launch.
//
// What fusion removes is concrete. The split path takes three launches, and both
// of its integrator launches read the acceleration array, 24N bytes per step.
// The fused path takes one launch and never reads that array. It still writes
// it, 12N bytes, though the engine never reads that write, because the
// diagnostics and the velocity conversions recompute the force first. It also
// updates the velocities once per step instead of twice. What
// fusion cannot remove is the position write, which has to happen either way.
//
// The integrator here is the leapfrog form of velocity Verlet:
//
//     v_{n+1/2} = v_{n-1/2} + a_n dt
//     x_{n+1}   = x_n + v_{n+1/2} dt
//
// which produces the same position sequence as the two launch velocity Verlet
// to within floating point rounding, provided the run starts by offsetting the
// velocity backwards by half a step. That offset is applied once before the
// loop. The synchronised velocity needed by the energy diagnostic is recovered
// as v_n = v_{n-1/2} + a_n dt / 2.
//
// Positions are read from one buffer and written to another. There is no grid
// wide barrier inside a launch, so a block that has finished its force sum and
// updated its own particles in place would corrupt the positions that a block
// still working on its sum is reading. The second buffer costs 12N bytes of
// memory and no additional bandwidth, since the write happens either way.
//
// The force sum is the one in nbody_blocking.hpp, against BF targets per thread,
// and the outer loop is the same grid stride as in the tiled kernel. What follows
// the sum happens per target, in the same arithmetic order at every BF, each one
// masked on its own index: a_n is written out, the velocity updated in place, and
// the position written to the other buffer. A target's velocity belongs to the
// one thread that owns it, so nothing here reads what another target writes. A
// thread with nothing to write does not return, because its block may have another
// pass with barriers that thread has to reach.
template <int TILE, int BF>
__device__ __forceinline__ void fused_step(
    const float* __restrict__ x_in, const float* __restrict__ y_in, const float* __restrict__ z_in,
    float* __restrict__ x_out, float* __restrict__ y_out, float* __restrict__ z_out,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    const float* __restrict__ mass, int count, float softening_sq, float gravitational_constant,
    float dt) {
    const int block_span = BF * static_cast<int>(blockDim.x);
    const int stride = block_span * static_cast<int>(gridDim.x);
    for (int block_first = static_cast<int>(blockIdx.x) * block_span; block_first < count;
         block_first += stride) {
        const int first = first_target<BF>(block_first);
        TargetBlock<BF> targets = load_targets<BF>(x_in, y_in, z_in, first, count);
        accumulate_over_tiles<TILE, BF>(targets, x_in, y_in, z_in, mass, count, softening_sq);

#pragma unroll
        for (int k = 0; k < BF; ++k) {
            const int i = first + k;
            if (i < count) {
                const float ax = targets.ax[k] * gravitational_constant;
                const float ay = targets.ay[k] * gravitational_constant;
                const float az = targets.az[k] * gravitational_constant;

                ax_out[i] = ax;
                ay_out[i] = ay;
                az_out[i] = az;

                const float nvx = vx[i] + ax * dt;
                const float nvy = vy[i] + ay * dt;
                const float nvz = vz[i] + az * dt;

                vx[i] = nvx;
                vy[i] = nvy;
                vz[i] = nvz;

                x_out[i] = targets.x[k] + nvx * dt;
                y_out[i] = targets.y[k] + nvy * dt;
                z_out[i] = targets.z[k] + nvz * dt;
            }
        }
    }
}

// Launch bounds let the compiler fit its register allocation to the block launched.
template <int TILE, int BF, int BLOCK>
__global__ __launch_bounds__(BLOCK) void nbody_fused_step_kernel(
    const float* __restrict__ x_in, const float* __restrict__ y_in, const float* __restrict__ z_in,
    float* __restrict__ x_out, float* __restrict__ y_out, float* __restrict__ z_out,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    const float* __restrict__ mass, int count, float softening_sq, float gravitational_constant,
    float dt) {
    fused_step<TILE, BF>(x_in, y_in, z_in, x_out, y_out, z_out, vx, vy, vz, ax_out, ay_out, az_out,
                         mass, count, softening_sq, gravitational_constant, dt);
}

// Every block size with no bound of its own.
template <int TILE, int BF>
__global__ void nbody_fused_step_unbounded_kernel(
    const float* __restrict__ x_in, const float* __restrict__ y_in, const float* __restrict__ z_in,
    float* __restrict__ x_out, float* __restrict__ y_out, float* __restrict__ z_out,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    const float* __restrict__ mass, int count, float softening_sq, float gravitational_constant,
    float dt) {
    fused_step<TILE, BF>(x_in, y_in, z_in, x_out, y_out, z_out, vx, vy, vz, ax_out, ay_out, az_out,
                         mass, count, softening_sq, gravitational_constant, dt);
}

struct FusedLaunch {
    const float* x_in;
    const float* y_in;
    const float* z_in;
    float* x_out;
    float* y_out;
    float* z_out;
    float* vx;
    float* vy;
    float* vz;
    float* ax;
    float* ay;
    float* az;
    const float* mass;
    int count;
    float softening_sq;
    float gravitational_constant;
    float dt;
    int block_size;
    int max_grid;
    cudaStream_t stream;

    template <int TILE, int BF, int BLOCK>
    void bounded() const {
        const int grid = blocked_grid(count, BF, block_size, max_grid);
        nbody_fused_step_kernel<TILE, BF, BLOCK><<<grid, block_size, 0, stream>>>(
            x_in, y_in, z_in, x_out, y_out, z_out, vx, vy, vz, ax, ay, az, mass, count,
            softening_sq, gravitational_constant, dt);
        CUDA_CHECK(cudaGetLastError());
    }

    template <int TILE, int BF>
    void unbounded() const {
        require_block_fits(nbody_fused_step_unbounded_kernel<TILE, BF>, BF, block_size);
        const int grid = blocked_grid(count, BF, block_size, max_grid);
        nbody_fused_step_unbounded_kernel<TILE, BF><<<grid, block_size, 0, stream>>>(
            x_in, y_in, z_in, x_out, y_out, z_out, vx, vy, vz, ax, ay, az, mass, count,
            softening_sq, gravitational_constant, dt);
        CUDA_CHECK(cudaGetLastError());
    }
};

}  // namespace

void nbody_fused_step(const float* x_in, const float* y_in, const float* z_in, float* x_out,
                      float* y_out, float* z_out, float* vx, float* vy, float* vz, float* ax,
                      float* ay, float* az, const float* mass, int count, float softening,
                      float gravitational_constant, float dt, int block_size, int tile_size,
                      int blocking_factor, cudaStream_t stream, int max_grid) {
    if (count <= 0) {
        return;
    }
    const float softening_sq = softening * softening;
    const FusedLaunch launch{x_in, y_in, z_in, x_out, y_out, z_out,
                             vx,   vy,   vz,   ax,    ay,    az,
                             mass, count, softening_sq, gravitational_constant, dt,
                             block_size, max_grid, stream};
    launch_nbody_kernel(launch, tile_size, blocking_factor, block_size);
    CPS_SYNC_POINT(stream);
}

}  // namespace cps
