#include <stdexcept>
#include <string>

#include "physics/gravity.hpp"
#include "solvers/nbody_solver.hpp"
#include "utils/cuda_helpers.hpp"

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
template <int TILE>
__global__ void nbody_fused_step_kernel(
    const float* __restrict__ x_in, const float* __restrict__ y_in, const float* __restrict__ z_in,
    float* __restrict__ x_out, float* __restrict__ y_out, float* __restrict__ z_out,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ ax_out, float* __restrict__ ay_out, float* __restrict__ az_out,
    const float* __restrict__ mass, int count, float softening_sq, float gravitational_constant,
    float dt) {
    __shared__ float tile_x[TILE];
    __shared__ float tile_y[TILE];
    __shared__ float tile_z[TILE];
    __shared__ float tile_mass[TILE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < count;

    const float xi = active ? x_in[i] : 0.0f;
    const float yi = active ? y_in[i] : 0.0f;
    const float zi = active ? z_in[i] : 0.0f;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile_start = 0; tile_start < count; tile_start += TILE) {
        __syncthreads();
        for (int t = static_cast<int>(threadIdx.x); t < TILE; t += static_cast<int>(blockDim.x)) {
            const int source = tile_start + t;
            if (source < count) {
                tile_x[t] = x_in[source];
                tile_y[t] = y_in[source];
                tile_z[t] = z_in[source];
                tile_mass[t] = mass[source];
            } else {
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

    if (!active) {
        return;
    }

    ax *= gravitational_constant;
    ay *= gravitational_constant;
    az *= gravitational_constant;

    ax_out[i] = ax;
    ay_out[i] = ay;
    az_out[i] = az;

    const float nvx = vx[i] + ax * dt;
    const float nvy = vy[i] + ay * dt;
    const float nvz = vz[i] + az * dt;

    vx[i] = nvx;
    vy[i] = nvy;
    vz[i] = nvz;

    x_out[i] = xi + nvx * dt;
    y_out[i] = yi + nvy * dt;
    z_out[i] = zi + nvz * dt;
}

}  // namespace

void nbody_fused_step(const float* x_in, const float* y_in, const float* z_in, float* x_out,
                      float* y_out, float* z_out, float* vx, float* vy, float* vz, float* ax,
                      float* ay, float* az, const float* mass, int count, float softening,
                      float gravitational_constant, float dt, int block_size, int tile_size,
                      cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    const int grid = grid_for(count, block_size);
    const float softening_sq = softening * softening;

#define CPS_LAUNCH_FUSED(TILE)                                                                   \
    nbody_fused_step_kernel<TILE><<<grid, block_size, 0, stream>>>(                               \
        x_in, y_in, z_in, x_out, y_out, z_out, vx, vy, vz, ax, ay, az, mass, count, softening_sq, \
        gravitational_constant, dt)

    switch (tile_size) {
        case 32:
            CPS_LAUNCH_FUSED(32);
            break;
        case 64:
            CPS_LAUNCH_FUSED(64);
            break;
        case 128:
            CPS_LAUNCH_FUSED(128);
            break;
        case 256:
            CPS_LAUNCH_FUSED(256);
            break;
        default:
            throw std::runtime_error("unsupported tile size " + std::to_string(tile_size) +
                                     ", expected 32, 64, 128 or 256");
    }
#undef CPS_LAUNCH_FUSED

    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cps
