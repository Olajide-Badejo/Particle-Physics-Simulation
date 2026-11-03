#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/initial_conditions.hpp"
#include "core/particle_system.hpp"
#include "utils/cuda_helpers.hpp"

namespace cps {

// Weakly compressible smoothed particle hydrodynamics with spatial hash
// neighbour search.
//
// One step is six device passes:
//
//   1. hash      integer cell coordinate, hash bucket and exact cell key per
//                particle
//   2. sort      CUB radix sort of (bucket, index) pairs
//   3. bounds    first and last entry of every occupied bucket
//   4. gather    particle data permuted into sorted order, so the neighbour
//                loop reads contiguous memory
//   5. density   poly6 sum over the twenty seven surrounding cells, then the
//                equation of state
//   6. force     spiky pressure gradient, viscosity Laplacian, external
//                gravity, kick-drift leapfrog update and boundary handling
//
// The update is the same kick-drift leapfrog as the fused N-body kernel. The
// initial half step offset is not applied, and the viscous term is first order
// because it reads the half step velocity. The first diagnostic row after a
// resume uses a zero acceleration, since nothing recomputes it on load.
//
// The permutation is kept. After a step the particle arrays are in cell sorted
// order, which is the order the next step wants anyway, so the sort has an
// effect that outlives the step that paid for it.
class SphSolver {
   public:
    SphSolver(const Config& config, int count);

    void step(ParticleSystem& particles, float dt);

    // Runs passes one to five only. The tests use it to inspect the neighbour
    // structure and the density field without advancing time.
    void build_neighbours_and_density(ParticleSystem& particles);

    const float* density() const { return density_.get(); }
    const float* pressure() const { return pressure_.get(); }
    uint32_t table_size() const { return table_size_; }
    const SphDerivedParameters& derived() const { return derived_; }

    // Number of particles whose bucket collided with another occupied cell,
    // reported once at start up as a health check on the hash.
    int count_bucket_collisions(ParticleSystem& particles);

    std::size_t device_bytes() const;

   private:
    void hash_particles(ParticleSystem& particles);
    void sort_by_bucket();
    void compute_cell_bounds();
    void gather_sorted(ParticleSystem& particles);

    Config config_;
    SphDerivedParameters derived_{};
    int count_ = 0;
    int block_size_ = 256;
    uint32_t table_size_ = 0;
    float cell_size_ = 0.0f;

    DeviceBuffer<uint32_t> bucket_;
    DeviceBuffer<uint32_t> bucket_sorted_;
    DeviceBuffer<uint32_t> index_;
    DeviceBuffer<uint32_t> index_sorted_;
    DeviceBuffer<int64_t> cell_key_;
    DeviceBuffer<int64_t> cell_key_sorted_;
    DeviceBuffer<uint32_t> cell_start_;
    DeviceBuffer<uint32_t> cell_end_;

    // Sorted copies consumed by the density and force passes.
    DeviceBuffer<float> sx_, sy_, sz_;
    DeviceBuffer<float> svx_, svy_, svz_;
    DeviceBuffer<float> smass_;

    DeviceBuffer<float> density_;
    DeviceBuffer<float> pressure_;

    DeviceBuffer<char> sort_temp_;
    DeviceBuffer<int> collision_counter_;
};

}  // namespace cps
