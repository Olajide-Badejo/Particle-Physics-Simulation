#pragma once

#include <cstdint>
#include <string>

#include "cps/core/config.hpp"
#include "cps/core/initial_conditions.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/physics/uniform_grid.hpp"
#include "cps/solvers/solver.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {

// Weakly compressible smoothed particle hydrodynamics with dense grid neighbour
// search.
//
// One step is six device passes:
//
//   1. cells     cell index per particle, which is its sort key; also where a non
//                finite position is noticed
//   2. sort      CUB radix sort of (cell, index) pairs
//   3. bounds    where each cell's run begins in the sorted array
//   4. gather    particle data permuted into sorted order, so the neighbour
//                loop reads contiguous memory
//   5. density   poly6 sum over the nine surrounding x rows, then the equation of
//                state
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
//
// The constructor refuses a domain whose grid would be too large (see make_sph_grid)
// or that this device cannot hold, and a time step above the stability limits, unless
// the configuration allows that.
class SphSolver final : public Solver {
   public:
    SphSolver(const Config& config, int count);

    // The fluid keeps no force pass outside its step, so these two do nothing. The
    // acceleration array therefore holds zero until the first step, which is what
    // leaves the initial velocity in place and what the first diagnostic row after
    // a resume reads. The state is unaffected and the trajectory stays bit identical.
    void initialize(ParticleSystem&, cudaStream_t) override {}
    void on_resume(ParticleSystem&, cudaStream_t) override {}

    // Enqueues the six passes on the stream and returns without waiting for them.
    void step(ParticleSystem& particles, cudaStream_t stream) override;

    // Every pass reads and writes the same arrays each step, the sort's temporary
    // storage among them, so one capture covers all of them.
    bool graph_capturable() const override { return true; }

    // The kick-drift leapfrog carries a half step velocity. Its initial offset runs
    // while the acceleration is still zero, so it records the convention without
    // changing a velocity, and the initial velocity is taken as v_{-1/2} as it
    // stands. The viscous term is evaluated with that half step velocity, which
    // makes it first order. This stays HalfStep because it decides how the
    // diagnostics read the velocity, and v + a dt/2 is a better estimate of the
    // synchronised velocity than v alone.
    VelocityConvention velocity_convention() const override {
        return VelocityConvention::HalfStep;
    }

    // Nothing to do: the diagnostics read the acceleration the last step wrote.
    void prepare_diagnostics(ParticleSystem&, cudaStream_t) override {}

    bool has_potential_energy() const override { return false; }

    // Throws if non_finite_position_seen() is set. Drains the stream.
    void check_device_errors(long long step, cudaStream_t stream) const override;

    // The derived resolution and the time step limits.
    std::string startup_description() const override;

    // Runs passes one to five only. The tests use it to inspect the neighbour
    // structure and the density field without advancing time.
    void build_neighbours_and_density(ParticleSystem& particles, cudaStream_t stream);

    // Device arrays in cell sorted order, not particle order: density()[k] and
    // pressure()[k] belong to the particle at index sorted_index()[k] of the
    // particle arrays the last pass read. Reading them against those arrays by
    // the same k pairs each value with a different particle. Reading them from
    // the host means draining the stream the passes ran on first.
    const float* density() const { return density_.get(); }
    const float* pressure() const { return pressure_.get(); }
    const uint32_t* sorted_index() const { return index_sorted_.get(); }
    const uint32_t* cell_start() const { return cell_start_.get(); }
    const UniformGrid& grid() const { return grid_; }
    const SphDerivedParameters& derived() const { return derived_; }

    // True once a cell pass has met a NaN or Inf position. It stays set for
    // the life of the solver, because the wall clamp turns an infinite
    // position back into a finite one and leaves nothing later to find. Reading
    // it drains the stream, which has to be the one the passes ran on, so it is
    // not for use on every step. A position that first goes bad in a step's
    // update is only seen by the cell pass of the following step.
    bool non_finite_position_seen(cudaStream_t stream) const;

    std::size_t device_bytes() const;

   private:
    void assign_cells(ParticleSystem& particles, cudaStream_t stream);
    void sort_by_cell(cudaStream_t stream);
    void compute_cell_bounds(cudaStream_t stream);
    void gather_sorted(ParticleSystem& particles, cudaStream_t stream);

    Config config_;
    SphDerivedParameters derived_{};
    int count_ = 0;
    int block_size_ = 256;
    UniformGrid grid_{};

    DeviceBuffer<uint32_t> cell_;
    DeviceBuffer<uint32_t> cell_sorted_;
    DeviceBuffer<uint32_t> index_;
    DeviceBuffer<uint32_t> index_sorted_;
    DeviceBuffer<uint32_t> cell_start_;

    // Sorted copies consumed by the density and force passes.
    DeviceBuffer<float> sx_, sy_, sz_;
    DeviceBuffer<float> svx_, svy_, svz_;
    DeviceBuffer<float> smass_;
    DeviceBuffer<uint32_t> sid_;

    DeviceBuffer<float> density_;
    DeviceBuffer<float> pressure_;

    DeviceBuffer<char> sort_temp_;
    DeviceBuffer<int> non_finite_flag_;

    // Where the read of the flag above lands. Reading the flag is logically
    // const, and it does not change anything the solver computes.
    mutable HostBuffer<int> scalar_staging_;
};

}  // namespace cps
