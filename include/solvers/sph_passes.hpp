#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cps {

// Launchers for the individual passes of the fluid step. They take raw
// pointers rather than the solver object so the tests can drive one pass at a
// time and inspect what came out of it.

inline constexpr uint32_t CELL_EMPTY = 0xFFFFFFFFu;

// A NaN or Inf position is filed under NON_FINITE_CELL and sets the device int
// non_finite_flag to one. The pass only ever sets the flag; clearing it is up
// to the caller. Finite positions are not affected.
void sph_hash_particles(const float* x, const float* y, const float* z, uint32_t* bucket,
                        int64_t* cell_key, uint32_t* index, int* non_finite_flag, int count,
                        float cell_size, uint32_t table_size, int block_size,
                        cudaStream_t stream = nullptr);

std::size_t sph_sort_temp_bytes(int count, uint32_t table_size);

void sph_sort_by_bucket(void* temp, std::size_t temp_bytes, const uint32_t* bucket_in,
                        uint32_t* bucket_out, const uint32_t* index_in, uint32_t* index_out,
                        int count, uint32_t table_size, cudaStream_t stream = nullptr);

void sph_clear_cell_bounds(uint32_t* cell_start, uint32_t* cell_end, uint32_t table_size,
                           int block_size, cudaStream_t stream = nullptr);

void sph_compute_cell_bounds(const uint32_t* bucket_sorted, uint32_t* cell_start,
                             uint32_t* cell_end, int count, int block_size,
                             cudaStream_t stream = nullptr);

void sph_gather_sorted(const float* x, const float* y, const float* z, const float* vx,
                       const float* vy, const float* vz, const float* mass, const int64_t* cell_key,
                       const uint32_t* index_sorted, float* sx, float* sy, float* sz, float* svx,
                       float* svy, float* svz, float* smass, int64_t* cell_key_sorted, int count,
                       int block_size, cudaStream_t stream = nullptr);

void sph_compute_density(const float* sx, const float* sy, const float* sz, const float* smass,
                         const int64_t* cell_key_sorted, const uint32_t* cell_start,
                         const uint32_t* cell_end, float* density, float* pressure, int count,
                         float smoothing_radius, float cell_size, uint32_t table_size,
                         float rest_density, float gas_constant, int block_size,
                         cudaStream_t stream = nullptr);

// Pressure gradient, viscosity, external gravity, kick-drift leapfrog update
// and boundary handling in one pass, writing the result back into the primary
// particle arrays in cell sorted order.
void sph_forces_and_integrate(const float* sx, const float* sy, const float* sz, const float* svx,
                              const float* svy, const float* svz, const float* smass,
                              const float* density, const float* pressure,
                              const int64_t* cell_key_sorted, const uint32_t* cell_start,
                              const uint32_t* cell_end, float* x, float* y, float* z, float* vx,
                              float* vy, float* vz, float* ax, float* ay, float* az, float* mass,
                              int count, float smoothing_radius, float cell_size,
                              uint32_t table_size, float viscosity, float gx, float gy, float gz,
                              float dt, float domain_min_x, float domain_min_y, float domain_min_z,
                              float domain_max_x, float domain_max_y, float domain_max_z,
                              float boundary_damping, int block_size, cudaStream_t stream = nullptr);

// Counts particles that share a hash bucket with a particle from a different
// cell. A healthy table keeps this small; a pathological one shows up here
// before it shows up as a wrong density.
void sph_count_bucket_collisions(const uint32_t* bucket_sorted, const int64_t* cell_key_sorted,
                                 int* counter, int count, int block_size,
                                 cudaStream_t stream = nullptr);

}  // namespace cps
