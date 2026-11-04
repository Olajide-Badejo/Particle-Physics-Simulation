#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cps/physics/uniform_grid.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {

// Launchers for the individual passes of the fluid step. They take raw
// pointers rather than the solver object so the tests can drive one pass at a
// time and inspect what came out of it. The stream has no default: see
// nbody_solver.hpp.

// Writes each particle's cell index, which is its sort key. A NaN or Inf position
// gets the reserved slot and sets the device int non_finite_flag to one. The pass
// only ever sets the flag; clearing it is up to the caller. Finite positions are
// not affected.
void sph_assign_cells(const float* x, const float* y, const float* z, uint32_t* cell,
                      uint32_t* index, int* non_finite_flag, int count, const UniformGrid& grid,
                      int block_size, cudaStream_t stream);

std::size_t sph_sort_temp_bytes(int count, uint32_t num_cells);

// Sorts only the bits that can differ, which is sort_key_bits(num_cells).
void sph_sort_by_cell(void* temp, std::size_t temp_bytes, const uint32_t* cell_in,
                      uint32_t* cell_out, const uint32_t* index_in, uint32_t* index_out, int count,
                      uint32_t num_cells, cudaStream_t stream);

// Fills the cell table, which has cell_table_entries(num_cells) entries: entry c is
// the number of particles in cells below c, for c from 0 to num_cells inclusive, so
// cells c0 to c1 of a row are the range from entry c0 to entry c1 + 1.
void sph_compute_cell_bounds(const uint32_t* cell_sorted, uint32_t* cell_start, int count,
                             uint32_t num_cells, int block_size, cudaStream_t stream);

void sph_gather_sorted(const float* x, const float* y, const float* z, const float* vx,
                       const float* vy, const float* vz, const float* mass, const uint32_t* id,
                       const uint32_t* index_sorted, float* sx, float* sy, float* sz, float* svx,
                       float* svy, float* svz, float* smass, uint32_t* sid, int count,
                       int block_size, cudaStream_t stream);

void sph_compute_density(const float* sx, const float* sy, const float* sz, const float* smass,
                         const uint32_t* cell_start, float* density, float* pressure, int count,
                         float smoothing_radius, const UniformGrid& grid, float rest_density,
                         float gas_constant, int block_size, cudaStream_t stream);

// Pressure gradient, viscosity, external gravity, kick-drift leapfrog update
// and boundary handling in one pass, writing the result back into the primary
// particle arrays in cell sorted order.
void sph_forces_and_integrate(const float* sx, const float* sy, const float* sz, const float* svx,
                              const float* svy, const float* svz, const float* smass,
                              const float* density, const float* pressure,
                              const uint32_t* cell_start, const uint32_t* sid, float* x, float* y,
                              float* z, float* vx, float* vy, float* vz, float* ax, float* ay,
                              float* az, float* mass, uint32_t* id, int count,
                              float smoothing_radius, const UniformGrid& grid, float viscosity,
                              float gx, float gy, float gz, float dt, float domain_min_x,
                              float domain_min_y, float domain_min_z, float domain_max_x,
                              float domain_max_y, float domain_max_z, float boundary_damping,
                              int block_size, cudaStream_t stream);

// Allocates a cell table, or throws std::runtime_error naming the grid and what the
// device reported. make_sph_grid refuses a grid over the cell limit, and one under
// that limit can still be more than this device holds.
void allocate_cell_table(DeviceBuffer<uint32_t>& table, std::size_t entries,
                         const std::string& grid_description);

}  // namespace cps
