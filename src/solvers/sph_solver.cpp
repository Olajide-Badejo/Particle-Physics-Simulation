#include "cps/solvers/sph_solver.hpp"

#include <algorithm>
#include <stdexcept>

#include "cps/physics/spatial_hash.hpp"
#include "cps/solvers/sph_passes.hpp"

namespace cps {

SphSolver::SphSolver(const Config& config, int count)
    : config_(config), count_(count), block_size_(config.block_size) {
    if (count <= 0) {
        throw std::runtime_error("the fluid solver needs at least one particle");
    }

    derived_ = derive_sph_parameters(config);
    cell_size_ = derived_.smoothing_radius;
    if (!(cell_size_ > 0.0f)) {
        throw std::runtime_error("the derived smoothing radius is not positive");
    }
    check_sph_domain_cell_range(config.sph.domain_min, config.sph.domain_max, cell_size_);
    check_sph_time_step(config, derived_.smoothing_radius);

    // Twice the particle count keeps the average occupancy near one half, which
    // is where the collision rate stops mattering. A prime modulus is what
    // makes the exclusive or of the three products spread; a power of two mask
    // would throw away the high bits the multiplications just mixed into.
    table_size_ = next_prime(static_cast<uint32_t>(2 * count));

    const std::size_t n = static_cast<std::size_t>(count);
    bucket_.allocate(n);
    bucket_sorted_.allocate(n);
    index_.allocate(n);
    index_sorted_.allocate(n);
    cell_key_.allocate(n);
    cell_key_sorted_.allocate(n);
    cell_start_.allocate(table_size_);
    cell_end_.allocate(table_size_);

    sx_.allocate(n);
    sy_.allocate(n);
    sz_.allocate(n);
    svx_.allocate(n);
    svy_.allocate(n);
    svz_.allocate(n);
    smass_.allocate(n);

    density_.allocate(n);
    pressure_.allocate(n);

    const std::size_t temp_bytes = sph_sort_temp_bytes(count, table_size_);
    sort_temp_.allocate(temp_bytes);

    collision_counter_.allocate(1);
    collision_counter_.zero();
    non_finite_flag_.allocate(1);
    non_finite_flag_.zero();
    scalar_staging_.allocate(1);
}

void SphSolver::hash_particles(ParticleSystem& particles, cudaStream_t stream) {
    sph_hash_particles(particles.x(), particles.y(), particles.z(), bucket_.get(), cell_key_.get(),
                       index_.get(), non_finite_flag_.get(), count_, cell_size_, table_size_,
                       block_size_, stream);
}

void SphSolver::sort_by_bucket(cudaStream_t stream) {
    sph_sort_by_bucket(sort_temp_.get(), sort_temp_.bytes(), bucket_.get(), bucket_sorted_.get(),
                       index_.get(), index_sorted_.get(), count_, table_size_, stream);
}

void SphSolver::compute_cell_bounds(cudaStream_t stream) {
    sph_clear_cell_bounds(cell_start_.get(), cell_end_.get(), table_size_, block_size_, stream);
    sph_compute_cell_bounds(bucket_sorted_.get(), cell_start_.get(), cell_end_.get(), count_,
                            block_size_, stream);
}

void SphSolver::gather_sorted(ParticleSystem& particles, cudaStream_t stream) {
    sph_gather_sorted(particles.x(), particles.y(), particles.z(), particles.vx(), particles.vy(),
                      particles.vz(), particles.mass(), cell_key_.get(), index_sorted_.get(),
                      sx_.get(), sy_.get(), sz_.get(), svx_.get(), svy_.get(), svz_.get(),
                      smass_.get(), cell_key_sorted_.get(), count_, block_size_, stream);
}

void SphSolver::build_neighbours_and_density(ParticleSystem& particles, cudaStream_t stream) {
    hash_particles(particles, stream);
    sort_by_bucket(stream);
    compute_cell_bounds(stream);
    gather_sorted(particles, stream);
    sph_compute_density(sx_.get(), sy_.get(), sz_.get(), smass_.get(), cell_key_sorted_.get(),
                        cell_start_.get(), cell_end_.get(), density_.get(), pressure_.get(), count_,
                        derived_.smoothing_radius, cell_size_, table_size_,
                        config_.sph.rest_density, config_.sph.gas_constant, block_size_, stream);
}

void SphSolver::step(ParticleSystem& particles, float dt, cudaStream_t stream) {
    build_neighbours_and_density(particles, stream);

    const SphConfig& sph = config_.sph;
    sph_forces_and_integrate(
        sx_.get(), sy_.get(), sz_.get(), svx_.get(), svy_.get(), svz_.get(), smass_.get(),
        density_.get(), pressure_.get(), cell_key_sorted_.get(), cell_start_.get(), cell_end_.get(),
        particles.x(), particles.y(), particles.z(), particles.vx(), particles.vy(), particles.vz(),
        particles.ax(), particles.ay(), particles.az(), particles.mass(), count_,
        derived_.smoothing_radius, cell_size_, table_size_, sph.viscosity,
        config_.external_gravity.x, config_.external_gravity.y, config_.external_gravity.z, dt,
        sph.domain_min.x, sph.domain_min.y, sph.domain_min.z, sph.domain_max.x, sph.domain_max.y,
        sph.domain_max.z, sph.boundary_damping, block_size_, stream);
}

int SphSolver::count_bucket_collisions(ParticleSystem& particles, cudaStream_t stream) {
    hash_particles(particles, stream);
    sort_by_bucket(stream);
    compute_cell_bounds(stream);
    gather_sorted(particles, stream);
    sph_count_bucket_collisions(bucket_sorted_.get(), cell_key_sorted_.get(),
                                collision_counter_.get(), count_, block_size_, stream);
    copy_to_host_async(scalar_staging_.get(), collision_counter_.get(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return *scalar_staging_.get();
}

bool SphSolver::non_finite_position_seen(cudaStream_t stream) const {
    copy_to_host_async(scalar_staging_.get(), non_finite_flag_.get(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return *scalar_staging_.get() != 0;
}

std::size_t SphSolver::device_bytes() const {
    return bucket_.bytes() + bucket_sorted_.bytes() + index_.bytes() + index_sorted_.bytes() +
           cell_key_.bytes() + cell_key_sorted_.bytes() + cell_start_.bytes() + cell_end_.bytes() +
           sx_.bytes() + sy_.bytes() + sz_.bytes() + svx_.bytes() + svy_.bytes() + svz_.bytes() +
           smass_.bytes() + density_.bytes() + pressure_.bytes() + sort_temp_.bytes();
}

}  // namespace cps
