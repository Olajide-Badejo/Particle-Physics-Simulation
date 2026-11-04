#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/physics/sph_kernels.hpp"
#include "cps/physics/uniform_grid.hpp"
#include "cps/solvers/sph_passes.hpp"
#include "cps/solvers/sph_solver.hpp"
#include "test_support_gpu.hpp"

// The neighbour search at the places a dense grid can go wrong that a block of fluid
// never reaches: walls, corners, cell boundaries, one cell wide domains and an almost
// empty grid. Each case lays particles out by hand and compares the density and the
// force the device computes, particle by particle, with a host sum over every pair.
//
// The force is what makes this sensitive. The poly6 density of a neighbour near the
// edge of the support is the cube of a small number and could be lost without showing;
// the viscous force is linear in the same small number, so the velocities are set to
// differ, and a dropped neighbour moves an acceleration by far more than the limit.

namespace {

// The cases differ by rounding from the host sum, which is in double precision and
// sums in another order, and by nothing else.
constexpr double DENSITY_LIMIT = 2.0e-5;
constexpr double FORCE_LIMIT = 2.0e-4;

constexpr float SUPPORT = 0.05f;
constexpr float MASS = 0.002f;

struct Particle {
    float x, y, z;
};

struct Scene {
    cps::Vec3 domain_min{0.0f, 0.0f, 0.0f};
    cps::Vec3 domain_max{1.0f, 1.0f, 1.0f};
    float support = SUPPORT;
    // Far above any density these sparse layouts reach, the fluid has no pressure and
    // the force is all viscous. A low one turns the pressure term on.
    float rest_density = 1000.0f;
    std::vector<Particle> particles;
};

cps::Config config_for(const Scene& scene) {
    cps::Config config;
    config.name = "neighbour_search";
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = static_cast<int>(scene.particles.size());
    config.dt = 1.0e-6f;
    config.block_size = 64;
    config.external_gravity = {0.0f, 0.0f, 0.0f};
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.sph.domain_min = scene.domain_min;
    config.sph.domain_max = scene.domain_max;
    config.sph.fluid_min = scene.domain_min;
    config.sph.fluid_max = scene.domain_max;
    config.sph.smoothing_radius = scene.support;
    config.sph.particle_mass = MASS;
    config.sph.viscosity = 0.2f;
    config.sph.rest_density = scene.rest_density;
    return config;
}

// A velocity that differs from particle to particle and on every axis, and is the same
// for a given id on the device and on the host. It is small enough that a step moves
// nothing past a wall it was not already past.
float velocity_of(std::size_t id, int axis) {
    const std::size_t mix = id * 2654435761u + static_cast<std::size_t>(axis) * 40503u;
    return 0.01f * (static_cast<float>(mix % 2048u) / 1024.0f - 1.0f);
}

cps::HostParticleState state_of(const Scene& scene) {
    cps::HostParticleState state;
    state.resize(scene.particles.size());
    for (std::size_t i = 0; i < scene.particles.size(); ++i) {
        state.x[i] = scene.particles[i].x;
        state.y[i] = scene.particles[i].y;
        state.z[i] = scene.particles[i].z;
        state.vx[i] = velocity_of(i, 0);
        state.vy[i] = velocity_of(i, 1);
        state.vz[i] = velocity_of(i, 2);
        state.mass[i] = MASS;
    }
    return state;
}

struct Expected {
    std::vector<double> density;
    std::vector<double> ax, ay, az;
    long long pairs = 0;
    long long pairs_across_a_cell_boundary = 0;
};

// Every pair, by the same formulas and the same float distance test as the kernels,
// summed in double precision. A particle with a non finite position takes no part.
Expected brute_force(const Scene& scene, const cps::Config& config, const cps::UniformGrid& grid) {
    const cps::HostParticleState state = state_of(scene);
    const std::size_t n = state.size();
    const float h = scene.support;
    const float h_sq = h * h;

    Expected expected;
    expected.density.assign(n, 0.0);
    expected.ax.assign(n, 0.0);
    expected.ay.assign(n, 0.0);
    expected.az.assign(n, 0.0);

    const auto finite = [&](std::size_t i) {
        return cps::position_is_finite(state.x[i], state.y[i], state.z[i]);
    };

    for (std::size_t i = 0; i < n; ++i) {
        if (!finite(i)) {
            continue;
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (!finite(j)) {
                continue;
            }
            const float rx = state.x[i] - state.x[j];
            const float ry = state.y[i] - state.y[j];
            const float rz = state.z[i] - state.z[j];
            const float r_sq = rx * rx + ry * ry + rz * rz;
            if (r_sq < h_sq) {
                expected.density[i] +=
                    static_cast<double>(state.mass[j]) *
                    static_cast<double>(cps::poly6_kernel(r_sq, h));
                if (i != j) {
                    ++expected.pairs;
                    if (cps::cell_index(grid, state.x[i], state.y[i], state.z[i]) !=
                        cps::cell_index(grid, state.x[j], state.y[j], state.z[j])) {
                        ++expected.pairs_across_a_cell_boundary;
                    }
                }
            }
        }
    }

    const auto pressure_of = [&](double rho) {
        return static_cast<double>(cps::equation_of_state(static_cast<float>(rho),
                                                          config.sph.rest_density,
                                                          config.sph.gas_constant));
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (!finite(i)) {
            continue;
        }
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j || !finite(j)) {
                continue;
            }
            const float rx = state.x[i] - state.x[j];
            const float ry = state.y[i] - state.y[j];
            const float rz = state.z[i] - state.z[j];
            const float r_sq = rx * rx + ry * ry + rz * rz;
            if (r_sq >= h_sq || r_sq <= 0.0f) {
                continue;
            }
            const float r = std::sqrt(r_sq);
            const double rho_j = expected.density[j];
            const double m_j = state.mass[j];
            const double pressure_scale =
                m_j * (pressure_of(expected.density[i]) + pressure_of(rho_j)) / (2.0 * rho_j) *
                static_cast<double>(cps::spiky_gradient_magnitude(r, h)) / static_cast<double>(r);
            fx += pressure_scale * rx;
            fy += pressure_scale * ry;
            fz += pressure_scale * rz;
            const double viscous_scale = static_cast<double>(config.sph.viscosity) * m_j / rho_j *
                                         static_cast<double>(cps::viscosity_laplacian(r, h));
            fx += viscous_scale * (static_cast<double>(state.vx[j]) - state.vx[i]);
            fy += viscous_scale * (static_cast<double>(state.vy[j]) - state.vy[i]);
            fz += viscous_scale * (static_cast<double>(state.vz[j]) - state.vz[i]);
        }
        expected.ax[i] = fx / expected.density[i];
        expected.ay[i] = fy / expected.density[i];
        expected.az[i] = fz / expected.density[i];
    }
    return expected;
}

template <typename T>
std::vector<T> read_device(const T* device, std::size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// Runs one step of the solver on the scene and compares each particle, found by id,
// with the sum over every pair.
Expected expect_step_matches_brute_force(const Scene& scene) {
    const cps::Config config = config_for(scene);
    const std::size_t n = scene.particles.size();
    const cps::Stream stream;
    cps::ParticleSystem particles(static_cast<int>(n));
    particles.upload(state_of(scene), stream.get());

    cps::SphSolver solver(config, static_cast<int>(n));
    const Expected expected = brute_force(scene, config, solver.grid());

    // The step leaves the density behind in sorted order, with the ids it was gathered
    // under now written back to the particle arrays in that same order.
    solver.step(particles, stream.get());
    stream.synchronize();
    const std::vector<float> density = read_device(solver.density(), n);
    const std::vector<uint32_t> id = read_device(particles.id(), n);
    const std::vector<float> ax = read_device(particles.ax(), n);
    const std::vector<float> ay = read_device(particles.ay(), n);
    const std::vector<float> az = read_device(particles.az(), n);

    double largest_density = 0.0;
    double largest_acceleration = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        largest_density = std::max(largest_density, expected.density[i]);
        largest_acceleration =
            std::max({largest_acceleration, std::abs(expected.ax[i]), std::abs(expected.ay[i]),
                      std::abs(expected.az[i])});
    }
    EXPECT_GT(largest_density, 0.0);

    std::vector<int> seen(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = id[k];
        EXPECT_LT(i, n);
        ++seen[i];
        EXPECT_NEAR(density[k], expected.density[i], DENSITY_LIMIT * largest_density)
            << "id " << i;
        const double limit = FORCE_LIMIT * largest_acceleration + 1.0e-6;
        EXPECT_NEAR(ax[k], expected.ax[i], limit) << "id " << i;
        EXPECT_NEAR(ay[k], expected.ay[i], limit) << "id " << i;
        EXPECT_NEAR(az[k], expected.az[i], limit) << "id " << i;
    }
    EXPECT_EQ(*std::min_element(seen.begin(), seen.end()), 1);
    EXPECT_EQ(*std::max_element(seen.begin(), seen.end()), 1);
    return expected;
}

// The same comparison with the force all viscous, which is linear in how far a
// neighbour is inside the support, and again with pressure. Returns the sum, so a case
// can check that it actually held the pairs it was written to hold.
Expected expect_matches_brute_force(Scene scene) {
    const Expected expected = expect_step_matches_brute_force(scene);
    scene.rest_density = 1.0f;
    expect_step_matches_brute_force(scene);
    return expected;
}

// A neighbour spaced well inside the support on each side of a point, staying between
// the walls of the scene.
void add_cluster_around(Scene& scene, Particle centre) {
    const float step = 0.45f * scene.support;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const float x = centre.x + static_cast<float>(dx) * step;
                const float y = centre.y + static_cast<float>(dy) * step;
                const float z = centre.z + static_cast<float>(dz) * step;
                if (x >= scene.domain_min.x && x <= scene.domain_max.x &&
                    y >= scene.domain_min.y && y <= scene.domain_max.y &&
                    z >= scene.domain_min.z && z <= scene.domain_max.z) {
                    scene.particles.push_back({x, y, z});
                }
            }
        }
    }
}

TEST(NeighbourSearch, ParticlesOnEveryWallAndCornerMatchABruteForceSum) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    scene.domain_min = {-0.2f, 0.1f, -0.3f};
    scene.domain_max = {0.3f, 0.4f, 0.1f};
    const float xs[] = {scene.domain_min.x, 0.5f * (scene.domain_min.x + scene.domain_max.x),
                        scene.domain_max.x};
    const float ys[] = {scene.domain_min.y, 0.5f * (scene.domain_min.y + scene.domain_max.y),
                        scene.domain_max.y};
    const float zs[] = {scene.domain_min.z, 0.5f * (scene.domain_min.z + scene.domain_max.z),
                        scene.domain_max.z};
    // Eight corners, twelve edges, six faces and the middle.
    for (const float x : xs) {
        for (const float y : ys) {
            for (const float z : zs) {
                add_cluster_around(scene, {x, y, z});
            }
        }
    }
    const Expected expected = expect_matches_brute_force(scene);
    EXPECT_GT(expected.pairs, 2000);
    EXPECT_GT(expected.pairs_across_a_cell_boundary, 500);
}

// Pairs placed on purpose across every boundary of a row of cells, with the first of
// each within a float or two of the boundary and the second as far out as the distance
// test allows, along each axis and along the diagonal.
TEST(NeighbourSearch, PairsAcrossCellBoundariesMatchABruteForceSum) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    scene.domain_max = {1.0f, 1.0f, 1.0f};
    const cps::UniformGrid grid =
        cps::make_sph_grid(scene.domain_min, scene.domain_max, scene.support);
    const float cell = grid.cell_size;
    const float reach = 0.92f * scene.support;
    const float diagonal_reach = reach / std::sqrt(3.0f);
    for (int k = 2; k < 18; ++k) {
        const float boundary = static_cast<float>(k) * cell;
        const float other = 0.5f * cell + static_cast<float>(k % 5) * 3.0f * cell;
        for (const float at : {std::nextafter(boundary, 0.0f), boundary,
                               std::nextafter(boundary, 2.0f * boundary)}) {
            scene.particles.push_back({at, other, other});
            scene.particles.push_back({at + reach, other, other});
            scene.particles.push_back({at - reach, other, other});
            scene.particles.push_back({other + 0.2f, at, other});
            scene.particles.push_back({other + 0.2f, at + reach, other});
            scene.particles.push_back({other + 0.2f, at - reach, other});
            scene.particles.push_back({other + 0.4f, other, at});
            scene.particles.push_back({other + 0.4f, other, at + reach});
            scene.particles.push_back({other + 0.4f, other, at - reach});
            scene.particles.push_back({at + 0.3f, at + 0.1f, at});
            scene.particles.push_back({at + 0.3f + diagonal_reach, at + 0.1f + diagonal_reach,
                                       at + diagonal_reach});
        }
    }
    const Expected expected = expect_matches_brute_force(scene);
    EXPECT_GT(expected.pairs, 500);
    EXPECT_GT(expected.pairs_across_a_cell_boundary, 200);
}

// A domain narrower than one support on an axis has one cell on it. That is where a row
// is one cell, and where a search that does not stop at a wall would read outside the
// row or the table.
TEST(NeighbourSearch, OneCellWideDomainsMatchABruteForceSum) {
    SKIP_WITHOUT_GPU();
    const float narrow = 0.6f * SUPPORT;
    const float wide = 0.5f;
    const cps::Vec3 shapes[] = {{narrow, wide, wide}, {wide, narrow, wide}, {wide, wide, narrow},
                                {narrow, narrow, wide}, {narrow, wide, narrow},
                                {wide, narrow, narrow}, {narrow, narrow, narrow}};
    std::mt19937 rng(20261104u);
    for (const cps::Vec3& shape : shapes) {
        SCOPED_TRACE(std::to_string(shape.x) + " x " + std::to_string(shape.y) + " x " +
                     std::to_string(shape.z));
        Scene scene;
        scene.domain_max = shape;
        std::uniform_real_distribution<float> along_x(0.0f, shape.x);
        std::uniform_real_distribution<float> along_y(0.0f, shape.y);
        std::uniform_real_distribution<float> along_z(0.0f, shape.z);
        for (int i = 0; i < 600; ++i) {
            scene.particles.push_back({along_x(rng), along_y(rng), along_z(rng)});
        }
        // Both walls of the narrow axes, in one cell.
        scene.particles.push_back({0.0f, 0.0f, 0.0f});
        scene.particles.push_back({shape.x, shape.y, shape.z});
        const cps::UniformGrid grid =
            cps::make_sph_grid(scene.domain_min, scene.domain_max, scene.support);
        EXPECT_EQ(grid.nx == 1, shape.x == narrow);
        EXPECT_EQ(grid.ny == 1, shape.y == narrow);
        EXPECT_EQ(grid.nz == 1, shape.z == narrow);
        const Expected expected = expect_matches_brute_force(scene);
        EXPECT_GT(expected.pairs, 0);
    }
}

// Nearly every entry of the table is an empty cell here, including all those around each
// particle. The second case is one particle, whose sum is only itself.
TEST(NeighbourSearch, ANearlyEmptyGridMatchesABruteForceSum) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    scene.domain_max = {8.0f, 8.0f, 8.0f};
    const float h = scene.support;
    scene.particles = {
        // Alone at each end of the index, in the first and last cells.
        {0.0f, 0.0f, 0.0f},
        {8.0f, 8.0f, 8.0f},
        // Alone in the middle.
        {4.0f, 4.0f, 4.0f},
        // A pair in the same cell, a pair in adjacent x cells, in adjacent rows and in
        // adjacent layers, each far from everything else.
        {1.0f, 1.0f, 1.0f},
        {1.0f + 0.3f * h, 1.0f, 1.0f},
        {2.0f, 1.0f, 1.0f},
        {2.0f + 0.9f * h, 1.0f, 1.0f},
        {1.0f, 2.0f, 1.0f},
        {1.0f, 2.0f + 0.9f * h, 1.0f},
        {1.0f, 1.0f, 2.0f},
        {1.0f, 1.0f, 2.0f + 0.9f * h},
    };
    const cps::UniformGrid grid =
        cps::make_sph_grid(scene.domain_min, scene.domain_max, scene.support);
    EXPECT_GT(grid.num_cells, 1000000u);
    const Expected expected = expect_matches_brute_force(scene);
    EXPECT_EQ(expected.pairs, 8);

    Scene alone;
    alone.domain_max = scene.domain_max;
    alone.particles = {{3.0f, 3.0f, 3.0f}};
    EXPECT_EQ(expect_matches_brute_force(alone).pairs, 0);
}

// The positions are finite and outside the domain, which the clamp has to bring back
// into the edge cells without losing a neighbour or running out of the table, and which
// a float to int conversion on its own would not survive.
TEST(NeighbourSearch, PositionsPastTheWallsMatchABruteForceSum) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    scene.domain_max = {0.5f, 0.5f, 0.5f};
    const float h = scene.support;
    const float big = 1.0e30f;
    // Pairs that straddle each wall, and pairs past it.
    for (int axis = 0; axis < 3; ++axis) {
        const auto at = [&](float along) {
            Particle p{0.25f, 0.25f, 0.25f};
            (axis == 0 ? p.x : axis == 1 ? p.y : p.z) = along;
            return p;
        };
        for (const float along : {-0.4f * h, 0.4f * h, -3.0f * h, -3.4f * h, 0.5f - 0.4f * h,
                                  0.5f + 0.4f * h, 0.5f + 3.0f * h, 0.5f + 3.4f * h, big, -big}) {
            scene.particles.push_back(at(along));
        }
    }
    const Expected expected = expect_matches_brute_force(scene);
    EXPECT_GE(expected.pairs, 2 * 3 * 4);
}

// A grid of exactly 512 cells, where the reserved slot needs one more bit than any real
// cell does. If the sort's bit range left that bit out, the bad particle would be filed
// under cell zero, and the particles there would read its NaN. The neighbours of the
// bad particle match a sum that leaves it out, in the first and the last cell both.
TEST(NeighbourSearch, TheReservedSlotIsSortedApartAtAPowerOfTwoCellCount) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    const float side = 7.5f * SUPPORT;
    scene.domain_max = {side, side, side};
    const cps::UniformGrid grid =
        cps::make_sph_grid(scene.domain_min, scene.domain_max, scene.support);
    ASSERT_EQ(grid.num_cells, 512u);

    add_cluster_around(scene, {0.0f, 0.0f, 0.0f});
    add_cluster_around(scene, {side, side, side});
    add_cluster_around(scene, {0.5f * side, 0.5f * side, 0.5f * side});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    scene.particles.push_back({nan, 0.0f, 0.0f});
    scene.particles.push_back({0.0f, inf, 0.0f});
    scene.particles.push_back({0.0f, 0.0f, -inf});

    const cps::Config config = config_for(scene);
    const std::size_t n = scene.particles.size();
    const Expected expected = brute_force(scene, config, grid);
    EXPECT_GT(expected.pairs, 100);

    const cps::Stream stream;
    cps::ParticleSystem particles(static_cast<int>(n));
    particles.upload(state_of(scene), stream.get());
    cps::SphSolver solver(config, static_cast<int>(n));
    solver.build_neighbours_and_density(particles, stream.get());
    stream.synchronize();
    EXPECT_TRUE(solver.non_finite_position_seen(stream.get()));

    const std::vector<uint32_t> table = read_device(solver.cell_start(), 513u);
    EXPECT_EQ(table[512], static_cast<uint32_t>(n - 3)) << "the finite particles should come first";

    const std::vector<float> density = read_device(solver.density(), n);
    const std::vector<uint32_t> sorted_index = read_device(solver.sorted_index(), n);
    double largest = 0.0;
    for (const double rho : expected.density) {
        largest = std::max(largest, rho);
    }
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = sorted_index[k];
        // That covers the bad particles too, whose sum over nothing is zero.
        EXPECT_NEAR(density[k], expected.density[i], DENSITY_LIMIT * largest) << "index " << i;
    }
    for (std::size_t k = n - 3; k < n; ++k) {
        EXPECT_GE(sorted_index[k], n - 3) << "a finite particle sorted past the real cells";
    }
}

// Below the limit the grid is refused only by the device. A table that no device holds
// has to fail in the error that names the grid, and leave nothing behind.
TEST(NeighbourSearch, AFailedTableAllocationNamesTheGrid) {
    SKIP_WITHOUT_GPU();
    cps::DeviceBuffer<uint32_t> table;
    const std::string description = "1 x 2 x 3 cells";
    const std::size_t no_device_has_this = std::size_t{1} << 50;
    try {
        cps::allocate_cell_table(table, no_device_has_this, description);
        ADD_FAILURE() << "expected the allocation to fail";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("could not allocate the cell table"), std::string::npos) << message;
        EXPECT_NE(message.find(description), std::string::npos) << message;
        EXPECT_NE(message.find("raise the smoothing radius or shrink the domain"),
                  std::string::npos)
            << message;
    }
    EXPECT_EQ(table.get(), nullptr);
    EXPECT_EQ(table.count(), 0u);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the next launch would report this failure";

    cps::allocate_cell_table(table, 16u, description);
    EXPECT_NE(table.get(), nullptr);
    EXPECT_EQ(table.count(), 16u);
}

// The whole domain check, seen from the solver.
TEST(NeighbourSearch, TheSolverRefusesAGridOverTheCellLimitAndSaysHowBig) {
    SKIP_WITHOUT_GPU();
    Scene scene;
    scene.domain_max = {3000.0f, 3000.0f, 3000.0f};
    scene.support = 1.0f;
    scene.particles = {{1.0f, 1.0f, 1.0f}};
    cps::Config config = config_for(scene);
    config.dt = 1.0e-4f;
    std::string message;
    try {
        cps::SphSolver solver(config, 1);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    EXPECT_NE(message.find("the fluid grid is too large"), std::string::npos) << message;
    EXPECT_NE(message.find(" x "), std::string::npos) << message;
    EXPECT_NE(message.find("MiB"), std::string::npos) << message;
}

}  // namespace
