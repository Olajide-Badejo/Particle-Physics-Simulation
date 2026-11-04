#include "cps/physics/spatial_hash.hpp"
#include "cps/physics/sph_kernels.hpp"
#include "cps/solvers/sph_passes.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {
namespace {

// Density by summing the poly6 kernel over the twenty seven cells that can
// hold a neighbour when the cell size equals the smoothing radius.
//
// The exact cell key test inside the bucket scan is not an optimisation, it is
// a correctness requirement. Hashing is many to one, so a bucket can hold
// particles from cells that are nowhere near this one, and two of the twenty
// seven cells visited here can land in the same bucket. Without the test the
// first class of particle is filtered out later by the radius check, but the
// second is not: the same neighbour would be summed twice and the density
// would come out high in a way that depends on the hash rather than on the
// fluid.
__global__ void density_kernel(const float* __restrict__ sx, const float* __restrict__ sy,
                               const float* __restrict__ sz, const float* __restrict__ smass,
                               const int64_t* __restrict__ cell_key_sorted,
                               const uint32_t* __restrict__ cell_start,
                               const uint32_t* __restrict__ cell_end, float* __restrict__ density,
                               float* __restrict__ pressure, int count, float smoothing_radius,
                               float cell_size, uint32_t table_size, float rest_density,
                               float gas_constant) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }

    const float xi = sx[k];
    const float yi = sy[k];
    const float zi = sz[k];
    const float h_sq = smoothing_radius * smoothing_radius;

    const int cx = cell_coordinate(xi, cell_size);
    const int cy = cell_coordinate(yi, cell_size);
    const int cz = cell_coordinate(zi, cell_size);

    float rho = 0.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = cx + dx;
                const int ny = cy + dy;
                const int nz = cz + dz;
                const uint32_t bucket = spatial_hash(nx, ny, nz, table_size);
                const uint32_t start = cell_start[bucket];
                if (start == CELL_EMPTY) {
                    continue;
                }
                const uint32_t end = cell_end[bucket];
                const int64_t wanted = cell_key(nx, ny, nz);
                for (uint32_t m = start; m < end; ++m) {
                    if (cell_key_sorted[m] != wanted) {
                        continue;
                    }
                    const float rx = xi - sx[m];
                    const float ry = yi - sy[m];
                    const float rz = zi - sz[m];
                    const float r_sq = rx * rx + ry * ry + rz * rz;
                    if (r_sq < h_sq) {
                        rho += smass[m] * poly6_kernel(r_sq, smoothing_radius);
                    }
                }
            }
        }
    }

    density[k] = rho;
    pressure[k] = equation_of_state(rho, rest_density, gas_constant);
}

// Pressure gradient, viscosity, gravity, kick-drift leapfrog update and walls.
//
// Fusing the update into the force pass is the same trade the fused N-body
// kernel makes and it is safe for the same reason: the sorted arrays this pass
// reads are a separate copy from the primary arrays it writes, so no block can
// observe a partially updated neighbour.
__global__ void forces_and_integrate_kernel(
    const float* __restrict__ sx, const float* __restrict__ sy, const float* __restrict__ sz,
    const float* __restrict__ svx, const float* __restrict__ svy, const float* __restrict__ svz,
    const float* __restrict__ smass, const float* __restrict__ density,
    const float* __restrict__ pressure, const int64_t* __restrict__ cell_key_sorted,
    const uint32_t* __restrict__ cell_start, const uint32_t* __restrict__ cell_end,
    const uint32_t* __restrict__ sid, float* __restrict__ x, float* __restrict__ y,
    float* __restrict__ z, float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
    float* __restrict__ mass, uint32_t* __restrict__ id, int count, float smoothing_radius,
    float cell_size, uint32_t table_size, float viscosity, float gx, float gy, float gz, float dt,
    float domain_min_x, float domain_min_y, float domain_min_z, float domain_max_x,
    float domain_max_y, float domain_max_z, float boundary_damping) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }

    const float xi = sx[k];
    const float yi = sy[k];
    const float zi = sz[k];
    const float ui = svx[k];
    const float vi = svy[k];
    const float wi = svz[k];
    const float rho_i = density[k];
    const float p_i = pressure[k];

    const int cx = cell_coordinate(xi, cell_size);
    const int cy = cell_coordinate(yi, cell_size);
    const int cz = cell_coordinate(zi, cell_size);

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = cx + dx;
                const int ny = cy + dy;
                const int nz = cz + dz;
                const uint32_t bucket = spatial_hash(nx, ny, nz, table_size);
                const uint32_t start = cell_start[bucket];
                if (start == CELL_EMPTY) {
                    continue;
                }
                const uint32_t end = cell_end[bucket];
                const int64_t wanted = cell_key(nx, ny, nz);
                for (uint32_t m = start; m < end; ++m) {
                    if (static_cast<int>(m) == k || cell_key_sorted[m] != wanted) {
                        continue;
                    }
                    const float rx = xi - sx[m];
                    const float ry = yi - sy[m];
                    const float rz = zi - sz[m];
                    const float r_sq = rx * rx + ry * ry + rz * rz;
                    if (r_sq >= smoothing_radius * smoothing_radius || r_sq <= 0.0f) {
                        continue;
                    }
                    const float r = sqrtf(r_sq);
                    const float rho_j = density[m];
                    if (rho_j <= 0.0f) {
                        continue;
                    }
                    const float m_j = smass[m];

                    // Symmetrised pressure term. Averaging the two pressures
                    // is what keeps the pair interaction equal and opposite;
                    // using p_i alone conserves nothing and drives a slow
                    // drift of the whole body of fluid.
                    const float grad = spiky_gradient_magnitude(r, smoothing_radius);
                    const float pressure_scale =
                        m_j * (p_i + pressure[m]) / (2.0f * rho_j) * grad / r;
                    fx += pressure_scale * rx;
                    fy += pressure_scale * ry;
                    fz += pressure_scale * rz;

                    // Viscosity acts on the velocity difference, so it can
                    // only ever remove relative motion.
                    const float lap = viscosity_laplacian(r, smoothing_radius);
                    const float viscous_scale = viscosity * m_j / rho_j * lap;
                    fx += viscous_scale * (svx[m] - ui);
                    fy += viscous_scale * (svy[m] - vi);
                    fz += viscous_scale * (svz[m] - wi);
                }
            }
        }
    }

    const float inv_rho = rho_i > 0.0f ? 1.0f / rho_i : 0.0f;
    float acc_x = fx * inv_rho + gx;
    float acc_y = fy * inv_rho + gy;
    float acc_z = fz * inv_rho + gz;

    float nvx = ui + acc_x * dt;
    float nvy = vi + acc_y * dt;
    float nvz = wi + acc_z * dt;

    float px = xi + nvx * dt;
    float py = yi + nvy * dt;
    float pz = zi + nvz * dt;

    // Walls. A particle that has crossed is placed back on the boundary and
    // its normal velocity is reversed and damped. Damping matters: a perfectly
    // elastic wall pumps energy into the column on every bounce and the dam
    // break never settles.
    if (px < domain_min_x) {
        px = domain_min_x;
        nvx = -nvx * boundary_damping;
    } else if (px > domain_max_x) {
        px = domain_max_x;
        nvx = -nvx * boundary_damping;
    }
    if (py < domain_min_y) {
        py = domain_min_y;
        nvy = -nvy * boundary_damping;
    } else if (py > domain_max_y) {
        py = domain_max_y;
        nvy = -nvy * boundary_damping;
    }
    if (pz < domain_min_z) {
        pz = domain_min_z;
        nvz = -nvz * boundary_damping;
    } else if (pz > domain_max_z) {
        pz = domain_max_z;
        nvz = -nvz * boundary_damping;
    }

    x[k] = px;
    y[k] = py;
    z[k] = pz;
    vx[k] = nvx;
    vy[k] = nvy;
    vz[k] = nvz;
    ax[k] = acc_x;
    ay[k] = acc_y;
    az[k] = acc_z;
    mass[k] = smass[k];
    id[k] = sid[k];
}

}  // namespace

void sph_compute_density(const float* sx, const float* sy, const float* sz, const float* smass,
                         const int64_t* cell_key_sorted, const uint32_t* cell_start,
                         const uint32_t* cell_end, float* density, float* pressure, int count,
                         float smoothing_radius, float cell_size, uint32_t table_size,
                         float rest_density, float gas_constant, int block_size,
                         cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    density_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        sx, sy, sz, smass, cell_key_sorted, cell_start, cell_end, density, pressure, count,
        smoothing_radius, cell_size, table_size, rest_density, gas_constant);
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

void sph_forces_and_integrate(const float* sx, const float* sy, const float* sz, const float* svx,
                              const float* svy, const float* svz, const float* smass,
                              const float* density, const float* pressure,
                              const int64_t* cell_key_sorted, const uint32_t* cell_start,
                              const uint32_t* cell_end, const uint32_t* sid, float* x, float* y,
                              float* z, float* vx, float* vy, float* vz, float* ax, float* ay,
                              float* az, float* mass, uint32_t* id, int count,
                              float smoothing_radius, float cell_size,
                              uint32_t table_size, float viscosity, float gx, float gy, float gz,
                              float dt, float domain_min_x, float domain_min_y, float domain_min_z,
                              float domain_max_x, float domain_max_y, float domain_max_z,
                              float boundary_damping, int block_size, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    forces_and_integrate_kernel<<<grid_for(count, block_size), block_size, 0, stream>>>(
        sx, sy, sz, svx, svy, svz, smass, density, pressure, cell_key_sorted, cell_start, cell_end,
        sid, x, y, z, vx, vy, vz, ax, ay, az, mass, id, count, smoothing_radius, cell_size,
        table_size,
        viscosity, gx, gy, gz, dt, domain_min_x, domain_min_y, domain_min_z, domain_max_x,
        domain_max_y, domain_max_z, boundary_damping);
    CUDA_CHECK(cudaGetLastError());
    CPS_SYNC_POINT(stream);
}

}  // namespace cps
