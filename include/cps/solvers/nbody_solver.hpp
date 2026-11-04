#pragma once

#include <cuda_runtime.h>

#include "cps/core/config.hpp"
#include "cps/solvers/solver.hpp"

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
//          launch, removing the 12N byte read back of the acceleration array
//          and one launch per step. The acceleration is still written.
//
// Each tile width is its own instantiation, chosen at launch, so the inner loop can
// be unrolled. The default, DEFAULT_TILE_SIZE, is the width the tile sweep measured
// fastest; the others exist so the sweep can keep checking that against real numbers
// rather than arguing about it.
//
// Every launcher takes its stream with no default, so a caller cannot end up on
// the legacy default stream by leaving it out. nullptr names that stream on
// purpose, which is for tests that drive a launcher on its own.

void compute_forces_naive(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size,
                          cudaStream_t stream);

void compute_forces_tiled(const float* x, const float* y, const float* z, const float* mass,
                          float* ax, float* ay, float* az, int count, float softening,
                          float gravitational_constant, int block_size,
                          int tile_size, cudaStream_t stream);

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
                      int tile_size, cudaStream_t stream);

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
                               float dt, int block_size, cudaStream_t stream);

void integrate_verlet_velocity(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream);

// The N-body solver: the three variants above, over the launchers above.
//
// All three seed and recompute with the tiled kernel, the naive variant included,
// so that is the only place nbody.tile_size reaches the naive variant. The fused
// variant carries the half step velocity and the other two the synchronised one.
//
// Positions are double buffered for the fused variant, so x() names a different
// array after each of its steps. The solver holds no pointers across a step.
//
// The split variants can be captured into a graph, which removes the time between
// their kernels. The fused variant launches a single kernel, so it has no such time
// to remove, and it stays on the stream.
class NBodySolver final : public Solver {
   public:
    explicit NBodySolver(const Config& config);

    void initialize(ParticleSystem& particles, cudaStream_t stream) override;
    void on_resume(ParticleSystem& particles, cudaStream_t stream) override;
    void step(ParticleSystem& particles, cudaStream_t stream) override;
    VelocityConvention velocity_convention() const override;
    void prepare_diagnostics(ParticleSystem& particles, cudaStream_t stream) override;
    bool has_potential_energy() const override { return true; }

    bool graph_capturable() const override { return config_.variant != NBodyVariant::Fused; }

    NBodyVariant variant() const { return config_.variant; }
    int tile_size() const { return config_.nbody.tile_size; }

   private:
    // The tiled force evaluation, at the configured width.
    void compute_forces(ParticleSystem& particles, cudaStream_t stream);
    void step_split(ParticleSystem& particles, cudaStream_t stream);
    void step_fused(ParticleSystem& particles, cudaStream_t stream);

    Config config_;
};

}  // namespace cps
