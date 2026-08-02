#include "solvers/nbody_solver.hpp"
#include "utils/cuda_helpers.hpp"

namespace cps {
namespace {

// First half of velocity Verlet: the opening half kick and the drift.
//
//     v <- v + a dt / 2
//     x <- x + v dt
//
// After this kernel the velocity array holds the half step velocity. Writing
// the drift as x + v dt using the already updated v is algebraically identical
// to x + v dt + a dt^2 / 2 and saves one multiply, but more importantly it
// keeps the half step velocity explicit, which is the quantity the closing
// kick expects.
__global__ void verlet_position_kernel(float* __restrict__ x, float* __restrict__ y,
                                       float* __restrict__ z, float* __restrict__ vx,
                                       float* __restrict__ vy, float* __restrict__ vz,
                                       const float* __restrict__ ax, const float* __restrict__ ay,
                                       const float* __restrict__ az, int count, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float half_dt = 0.5f * dt;
    const float nvx = vx[i] + ax[i] * half_dt;
    const float nvy = vy[i] + ay[i] * half_dt;
    const float nvz = vz[i] + az[i] * half_dt;
    vx[i] = nvx;
    vy[i] = nvy;
    vz[i] = nvz;
    x[i] += nvx * dt;
    y[i] += nvy * dt;
    z[i] += nvz * dt;
}

// Second half of velocity Verlet: the closing half kick, using the freshly
// evaluated acceleration.
__global__ void verlet_velocity_kernel(float* __restrict__ vx, float* __restrict__ vy,
                                       float* __restrict__ vz, const float* __restrict__ ax,
                                       const float* __restrict__ ay, const float* __restrict__ az,
                                       int count, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float half_dt = 0.5f * dt;
    vx[i] += ax[i] * half_dt;
    vy[i] += ay[i] * half_dt;
    vz[i] += az[i] * half_dt;
}

// Shifts the velocity by a half step. Called with a negative dt to enter the
// leapfrog convention the fused kernel carries, and with a positive dt to come
// back out of it for a diagnostic or a snapshot.
__global__ void velocity_half_step_kernel(float* __restrict__ vx, float* __restrict__ vy,
                                          float* __restrict__ vz, const float* __restrict__ ax,
                                          const float* __restrict__ ay,
                                          const float* __restrict__ az, int count, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float half_dt = 0.5f * dt;
    vx[i] += ax[i] * half_dt;
    vy[i] += ay[i] * half_dt;
    vz[i] += az[i] * half_dt;
}

}  // namespace

void integrate_verlet_position(float* x, float* y, float* z, float* vx, float* vy, float* vz,
                               const float* ax, const float* ay, const float* az, int count,
                               float dt, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    verlet_position_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        x, y, z, vx, vy, vz, ax, ay, az, count, dt);
    CUDA_CHECK(cudaGetLastError());
}

void integrate_verlet_velocity(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    verlet_velocity_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        vx, vy, vz, ax, ay, az, count, dt);
    CUDA_CHECK(cudaGetLastError());
}

void offset_velocity_half_step(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    velocity_half_step_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        vx, vy, vz, ax, ay, az, count, dt);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cps
