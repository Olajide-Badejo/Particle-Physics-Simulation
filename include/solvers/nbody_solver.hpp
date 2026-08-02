#pragma once

#include <cuda_runtime.h>

namespace cps {

// The three N-body force variants. Each isolates exactly one technique so the
// benchmark ladder attributes its measured delta to one change and not to a
// bundle of them.
//
//   naive  every thread reads every other particle straight from global
//          memory. The reads are uniform across a warp, so the hardware
//          broadcasts and the caches absorb much of the traffic. This is the
//          reference rung, not a straw man.
//
//   tiled  the block cooperatively stages a tile of particles in shared
//          memory and every thread in the block consumes it. Global traffic
//          drops by the block size; the cost is two barriers per tile.
//
//   fused  the tiled force evaluation and the Verlet update happen in one
//          launch, removing a 12N byte round trip of the acceleration array
//          and one launch per step.
//
// The tile width is a compile time constant so the inner loop can be unrolled.
// Thirty two is the specified default, matching the warp width; the other
// instantiations exist so the tile width sweep in the debug report can be run
// against real numbers rather than argued about.
inline constexpr int DEFAULT_TILE_SIZE = 32;

void compute_forces_naive(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size,
                          cudaStream_t stream = nullptr);

void compute_forces_tiled(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size,
                          int tile_size = DEFAULT_TILE_SIZE, cudaStream_t stream = nullptr);

// One complete leapfrog step in a single launch.
//
// Reads positions from the x_in family and writes the advanced positions to
// the x_out family, because there is no grid wide barrier between the force
// sum and the position update. Velocities are private to each thread and are
// updated in place. The accelerations are also written out, purely so the
// diagnostics and the snapshot path can see them; nothing in the step depends
// on reading them back.
void nbody_fused_step(const float* x_in, const float* y_in, const float* z_in, float* x_out,
                      float* y_out, float* z_out, float* vx, float* vy, float* vz, float* ax,
                      float* ay, float* az, const float* mass, int count, float softening,
                      float gravitational_constant, float dt, int block_size,
                      int tile_size = DEFAULT_TILE_SIZE, cudaStream_t stream = nullptr);

// Velocity Verlet, split across two launches for the unfused variants.
//
//   x_{n+1} = x_n + v_n dt + a_n dt^2 / 2
//   v_{n+1} = v_n + (a_n + a_{n+1}) dt / 2
//
// The first launch performs the half kick and the drift, leaving the half step
// velocity in the velocity array. The force kernel then runs. The second
// launch applies the closing half kick. Splitting it this way needs the force
// evaluated once per step, which matters because the force is the entire cost.
void integrate_verlet_position(float* x, float* y, float* z, float* vx, float* vy, float* vz,
                               const float* ax, const float* ay, const float* az, int count,
                               float dt, int block_size, cudaStream_t stream = nullptr);

void integrate_verlet_velocity(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream = nullptr);

// Converts a synchronised velocity into the half step velocity the fused
// leapfrog carries, and back again. Applying the first before the fused loop
// starts is what makes the fused trajectory agree with velocity Verlet rather
// than being offset by half a step.
void offset_velocity_half_step(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream = nullptr);

}  // namespace cps
