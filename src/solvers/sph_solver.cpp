#include "cps/solvers/sph_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "../utils/nvtx_range.hpp"
#include "cps/physics/uniform_grid.hpp"
#include "cps/solvers/sph_passes.hpp"

namespace cps {

SphSolver::SphSolver(const Config& config, int count)
    : config_(config), count_(count), block_size_(config.block_size) {
    if (count <= 0) {
        throw std::runtime_error("the fluid solver needs at least one particle");
    }

    derived_ = derive_sph_parameters(config);
    const float support = derived_.smoothing_radius;
    if (!(support > 0.0f)) {
        throw std::runtime_error("the derived smoothing radius is not positive");
    }
    grid_ = make_sph_grid(config.sph.domain_min, config.sph.domain_max, support);
    check_sph_time_step(config, support);

    // First, because it is the one allocation a domain can make arbitrarily large.
    allocate_cell_table(cell_start_, static_cast<std::size_t>(cell_table_entries(grid_.num_cells)),
                        sph_grid_description(grid_));

    const std::size_t n = static_cast<std::size_t>(count);
    cell_.allocate(n);
    cell_sorted_.allocate(n);
    index_.allocate(n);
    index_sorted_.allocate(n);

    sx_.allocate(n);
    sy_.allocate(n);
    sz_.allocate(n);
    svx_.allocate(n);
    svy_.allocate(n);
    svz_.allocate(n);
    smass_.allocate(n);
    sid_.allocate(n);

    density_.allocate(n);
    pressure_.allocate(n);

    const std::size_t temp_bytes = sph_sort_temp_bytes(count, grid_.num_cells);
    sort_temp_.allocate(temp_bytes);

    non_finite_flag_.allocate(1);
    non_finite_flag_.zero();
    scalar_staging_.allocate(1);
}

void SphSolver::assign_cells(ParticleSystem& particles, cudaStream_t stream) {
    const NvtxRange range("sph cells");
    sph_assign_cells(particles.x(), particles.y(), particles.z(), cell_.get(), index_.get(),
                     non_finite_flag_.get(), count_, grid_, block_size_, stream);
}

void SphSolver::sort_by_cell(cudaStream_t stream) {
    const NvtxRange range("sph sort");
    sph_sort_by_cell(sort_temp_.get(), sort_temp_.bytes(), cell_.get(), cell_sorted_.get(),
                     index_.get(), index_sorted_.get(), count_, grid_.num_cells, stream);
}

void SphSolver::compute_cell_bounds(cudaStream_t stream) {
    const NvtxRange range("sph cell bounds");
    sph_compute_cell_bounds(cell_sorted_.get(), cell_start_.get(), count_, grid_.num_cells,
                            block_size_, stream);
}

void SphSolver::gather_sorted(ParticleSystem& particles, cudaStream_t stream) {
    const NvtxRange range("sph gather");
    sph_gather_sorted(particles.x(), particles.y(), particles.z(), particles.vx(), particles.vy(),
                      particles.vz(), particles.mass(), particles.id(), index_sorted_.get(),
                      sx_.get(), sy_.get(), sz_.get(), svx_.get(), svy_.get(), svz_.get(),
                      smass_.get(), sid_.get(), count_, block_size_, stream);
}

void SphSolver::build_neighbours_and_density(ParticleSystem& particles, cudaStream_t stream) {
    assign_cells(particles, stream);
    sort_by_cell(stream);
    compute_cell_bounds(stream);
    gather_sorted(particles, stream);
    const NvtxRange range("sph density");
    sph_compute_density(sx_.get(), sy_.get(), sz_.get(), smass_.get(), cell_start_.get(),
                        density_.get(), pressure_.get(), count_, derived_.smoothing_radius, grid_,
                        config_.sph.rest_density, config_.sph.gas_constant, block_size_, stream);
}

void SphSolver::step(ParticleSystem& particles, cudaStream_t stream) {
    build_neighbours_and_density(particles, stream);

    const SphConfig& sph = config_.sph;
    const NvtxRange range("sph force and integrate");
    sph_forces_and_integrate(
        sx_.get(), sy_.get(), sz_.get(), svx_.get(), svy_.get(), svz_.get(), smass_.get(),
        density_.get(), pressure_.get(), cell_start_.get(), sid_.get(), particles.x(),
        particles.y(), particles.z(), particles.vx(), particles.vy(), particles.vz(),
        particles.ax(), particles.ay(), particles.az(), particles.mass(), particles.id(), count_,
        derived_.smoothing_radius, grid_, sph.viscosity, config_.external_gravity.x,
        config_.external_gravity.y, config_.external_gravity.z, config_.dt, sph.domain_min.x,
        sph.domain_min.y, sph.domain_min.z, sph.domain_max.x, sph.domain_max.y, sph.domain_max.z,
        sph.boundary_damping, block_size_, stream);
}

void SphSolver::check_device_errors(long long step, cudaStream_t stream) const {
    if (non_finite_position_seen(stream)) {
        throw std::runtime_error("non-finite positions detected by step " + std::to_string(step) +
                                 ": the fluid has blown up, typically from too large a time step");
    }
}

std::string SphSolver::startup_description() const {
    std::string text;
    char line[128];
    std::snprintf(line, sizeof(line), "  smoothing h   : %g\n",
                  static_cast<double>(derived_.smoothing_radius));
    text += line;
    std::snprintf(line, sizeof(line), "  particle mass : %g\n",
                  static_cast<double>(derived_.particle_mass));
    text += line;
    std::snprintf(line, sizeof(line), "  neighbours    : %.1f expected inside the support\n",
                  derived_.expected_neighbour_count);
    text += line;
    text += "  grid          : " + sph_grid_description(grid_) + "\n";

    const auto add_limit = [&](const char* name, double value) {
        if (std::isfinite(value)) {
            std::snprintf(line, sizeof(line), "  dt limit      : %s %g\n", name, value);
        } else {
            std::snprintf(line, sizeof(line), "  dt limit      : %s skipped\n", name);
        }
        text += line;
    };
    const SphTimeStepLimits limits = sph_time_step_limits(config_, derived_.smoothing_radius);
    add_limit("acoustic", limits.acoustic);
    add_limit("viscous", limits.viscous);
    add_limit("body force", limits.body_force);
    return text;
}

bool SphSolver::non_finite_position_seen(cudaStream_t stream) const {
    copy_to_host_async(scalar_staging_.get(), non_finite_flag_.get(), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return *scalar_staging_.get() != 0;
}

std::size_t SphSolver::device_bytes() const {
    return cell_.bytes() + cell_sorted_.bytes() + index_.bytes() + index_sorted_.bytes() +
           cell_start_.bytes() + sx_.bytes() + sy_.bytes() + sz_.bytes() + svx_.bytes() +
           svy_.bytes() + svz_.bytes() + smass_.bytes() + sid_.bytes() + density_.bytes() +
           pressure_.bytes() + sort_temp_.bytes();
}

}  // namespace cps
