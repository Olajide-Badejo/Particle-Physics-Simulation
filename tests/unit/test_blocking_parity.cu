#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/config.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

// Register blocking must not reach the numbers. A blocked thread holds BF targets
// against the same staged tile, and each target adds the same terms in the same
// order, so everything here is compared by bit pattern with BF 1 and nothing has a
// tolerance.
//
// Two binaries compile this file: unit_tests_gpu links the strict library and
// unit_tests_gpu_fast links physics_core, so the claim is held on both math settings.
// The fast math library could contract the arithmetic differently per instantiation,
// so a failure in that binary alone means exactly that, and is not to be loosened.

namespace {

using cps_test::random_cloud;
using cps_test::reference_accelerations;

constexpr float SOFTENING = 0.05f;
constexpr float GRAVITY = 1.0f;
constexpr float DT = 1.0e-3f;

constexpr int BLOCKING_FACTORS[] = {1, 2, 4, 8};
constexpr int TILE_WIDTHS[] = {32, 64, 128, 256};

std::string describe(int tile, int blocking_factor, int block, int count) {
    return "tile " + std::to_string(tile) + " BF " + std::to_string(blocking_factor) + " block " +
           std::to_string(block) + " N " + std::to_string(count);
}

// How far apart two floats are, in representable values, for the failure message.
std::int64_t ulp_distance(float a, float b) {
    const auto ordinal = [](float value) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        const std::int64_t magnitude = bits & 0x7fffffffu;
        return (bits >> 31) != 0u ? -magnitude : magnitude;
    };
    return std::abs(ordinal(a) - ordinal(b));
}

void expect_same_bits(const std::vector<float>& reference, const std::vector<float>& measured,
                      const std::string& label) {
    ASSERT_EQ(reference.size(), measured.size()) << label;
    std::size_t differing = 0;
    std::size_t worst = 0;
    std::int64_t worst_ulps = -1;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(reference[i]) !=
            std::bit_cast<std::uint32_t>(measured[i])) {
            ++differing;
            const std::int64_t ulps = ulp_distance(reference[i], measured[i]);
            if (ulps > worst_ulps) {
                worst_ulps = ulps;
                worst = i;
            }
        }
    }
    if (differing != 0) {
        ADD_FAILURE() << label << ": " << differing << " of " << reference.size()
                      << " differ, the worst by " << worst_ulps << " ulp at index " << worst
                      << ", " << reference[worst] << " against " << measured[worst];
    }
}

std::vector<float> read_device(const float* device, int count) {
    std::vector<float> out(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(out.data(), device, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

// All ones is a NaN. A target the kernel forgets to write then shows up as a
// difference, where the same array left over from the last launch would not.
void poison(float* device, int count) {
    CUDA_CHECK(cudaMemset(device, 0xff, static_cast<std::size_t>(count) * sizeof(float)));
}

struct Forces {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

Forces read_forces(const cps::ParticleSystem& particles) {
    return {read_device(particles.ax(), particles.size()),
            read_device(particles.ay(), particles.size()),
            read_device(particles.az(), particles.size())};
}

void expect_same_forces(const Forces& reference, const Forces& measured, const std::string& label) {
    expect_same_bits(reference.x, measured.x, label + " ax");
    expect_same_bits(reference.y, measured.y, label + " ay");
    expect_same_bits(reference.z, measured.z, label + " az");
}

Forces tiled_forces(const cps::HostParticleState& state, int tile, int blocking_factor, int block,
                    int max_grid = 0) {
    const int count = static_cast<int>(state.x.size());
    cps::ParticleSystem particles(count);
    particles.upload(state, nullptr);
    poison(particles.ax(), count);
    poison(particles.ay(), count);
    poison(particles.az(), count);
    cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, block, tile, blocking_factor, nullptr, max_grid);
    CUDA_CHECK(cudaDeviceSynchronize());
    return read_forces(particles);
}

// The whole state of a run, so a wrong velocity or a position written to the wrong
// buffer cannot hide behind a correct force.
struct RunState {
    std::vector<float> x, y, z, vx, vy, vz;
    Forces forces;
};

RunState read_run_state(const cps::ParticleSystem& particles) {
    const int count = particles.size();
    return {read_device(particles.x(), count),  read_device(particles.y(), count),
            read_device(particles.z(), count),  read_device(particles.vx(), count),
            read_device(particles.vy(), count), read_device(particles.vz(), count),
            read_forces(particles)};
}

void expect_same_run_state(const RunState& reference, const RunState& measured,
                           const std::string& label) {
    expect_same_bits(reference.x, measured.x, label + " x");
    expect_same_bits(reference.y, measured.y, label + " y");
    expect_same_bits(reference.z, measured.z, label + " z");
    expect_same_bits(reference.vx, measured.vx, label + " vx");
    expect_same_bits(reference.vy, measured.vy, label + " vy");
    expect_same_bits(reference.vz, measured.vz, label + " vz");
    expect_same_forces(reference.forces, measured.forces, label);
}

RunState run_fused(const cps::HostParticleState& state, int steps, int tile, int blocking_factor,
                   int block, int max_grid = 0) {
    const int count = static_cast<int>(state.x.size());
    cps::ParticleSystem particles(count);
    particles.upload(state, nullptr);
    for (int s = 0; s < steps; ++s) {
        poison(particles.x_next(), count);
        poison(particles.y_next(), count);
        poison(particles.z_next(), count);
        poison(particles.ax(), count);
        poison(particles.ay(), count);
        poison(particles.az(), count);
        cps::nbody_fused_step(particles.x(), particles.y(), particles.z(), particles.x_next(),
                              particles.y_next(), particles.z_next(), particles.vx(),
                              particles.vy(), particles.vz(), particles.ax(), particles.ay(),
                              particles.az(), particles.mass(), count, SOFTENING, GRAVITY, DT,
                              block, tile, blocking_factor, nullptr, max_grid);
        particles.swap_position_buffers();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return read_run_state(particles);
}

RunState run_split(const cps::HostParticleState& state, int steps, int tile, int blocking_factor,
                   int block) {
    const int count = static_cast<int>(state.x.size());
    cps::ParticleSystem particles(count);
    particles.upload(state, nullptr);
    const auto forces = [&] {
        poison(particles.ax(), count);
        poison(particles.ay(), count);
        poison(particles.az(), count);
        cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                                  particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                                  GRAVITY, block, tile, blocking_factor, nullptr);
    };
    forces();
    for (int s = 0; s < steps; ++s) {
        cps::integrate_verlet_position(particles.x(), particles.y(), particles.z(), particles.vx(),
                                       particles.vy(), particles.vz(), particles.ax(),
                                       particles.ay(), particles.az(), count, DT, block, nullptr);
        forces();
        cps::integrate_verlet_velocity(particles.vx(), particles.vy(), particles.vz(),
                                       particles.ax(), particles.ay(), particles.az(), count, DT,
                                       block, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return read_run_state(particles);
}

// Nothing here is a statement about a library the binary does not link. The
// definition comes from the library target and the expectation from this binary's.
TEST(KernelParity, ThisBinaryLinksTheLibraryItIsNamedFor) {
    EXPECT_EQ(CPS_FAST_MATH, CPS_EXPECTED_FAST_MATH);
}

// Each count leaves a partly filled last block at every factor, and a partly filled
// last tile at every width, so the masks and the padded slots are both in play.
TEST(KernelParity, EveryBlockingFactorIsBitIdentical) {
    SKIP_WITHOUT_GPU();
    struct Shape {
        int block;
        int count;
    };
    const Shape shapes[] = {{32, 733}, {64, 1533}, {128, 1533}, {256, 2741}};
    for (const Shape& shape : shapes) {
        const cps::HostParticleState state = random_cloud(shape.count, 8128u);
        for (const int tile : TILE_WIDTHS) {
            const Forces reference = tiled_forces(state, tile, 1, shape.block);
            for (const int blocking_factor : BLOCKING_FACTORS) {
                ASSERT_NE(shape.count % (blocking_factor * shape.block), 0);
                ASSERT_NE(shape.count % tile, 0);
                expect_same_forces(reference,
                                   tiled_forces(state, tile, blocking_factor, shape.block),
                                   describe(tile, blocking_factor, shape.block, shape.count));
            }
        }
    }
}

// Several steps, so a difference in one step feeds the next one, and on the whole
// state, so the integrator half of the fused kernel is held as well.
TEST(KernelParity, TheFusedStepIsBitIdenticalAtEveryBlockingFactor) {
    SKIP_WITHOUT_GPU();
    const int count = 733;
    const int steps = 5;
    const cps::HostParticleState state = random_cloud(count, 6174u);
    for (const int block : {128, 256}) {
        for (const int tile : TILE_WIDTHS) {
            const RunState reference = run_fused(state, steps, tile, 1, block);
            for (const int blocking_factor : BLOCKING_FACTORS) {
                expect_same_run_state(reference,
                                      run_fused(state, steps, tile, blocking_factor, block),
                                      "fused " + describe(tile, blocking_factor, block, count));
            }
        }
    }
}

TEST(KernelParity, TheSplitStepIsBitIdenticalAtEveryBlockingFactor) {
    SKIP_WITHOUT_GPU();
    const int count = 733;
    const int steps = 5;
    const cps::HostParticleState state = random_cloud(count, 1729u);
    for (const int block : {128, 256}) {
        for (const int tile : TILE_WIDTHS) {
            const RunState reference = run_split(state, steps, tile, 1, block);
            for (const int blocking_factor : BLOCKING_FACTORS) {
                expect_same_run_state(reference,
                                      run_split(state, steps, tile, blocking_factor, block),
                                      "split " + describe(tile, blocking_factor, block, count));
            }
        }
    }
}

// Where the tail handling can go wrong: fewer particles than one thread has targets,
// one short of a block, and one past a whole number of blocked blocks. The first count
// has no other particle to act on it, so the self interaction is all there is.
TEST(KernelParity, VerySmallCountsAreBitIdenticalAndMatchTheReference) {
    SKIP_WITHOUT_GPU();
    for (const int block : {32, 64, 256}) {
        for (const int blocking_factor : BLOCKING_FACTORS) {
            for (const int count : {1, 2, 3, block - 1, blocking_factor * block + 1}) {
                const cps::HostParticleState state =
                    random_cloud(count, 271u + static_cast<std::uint32_t>(count));
                std::vector<double> ax, ay, az;
                reference_accelerations(state, SOFTENING, GRAVITY, ax, ay, az);
                for (const int tile : {32, 256}) {
                    const std::string label = describe(tile, blocking_factor, block, count);
                    const Forces forces = tiled_forces(state, tile, blocking_factor, block);
                    expect_same_forces(tiled_forces(state, tile, 1, block), forces, label);
                    for (std::size_t i = 0; i < forces.x.size(); ++i) {
                        ASSERT_TRUE(std::isfinite(forces.x[i])) << label << " index " << i;
                        const double scale =
                            std::sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]) + 1.0e-3;
                        EXPECT_LT(std::abs(forces.x[i] - ax[i]) / scale, 1.0e-4) << label;
                        EXPECT_LT(std::abs(forces.y[i] - ay[i]) / scale, 1.0e-4) << label;
                        EXPECT_LT(std::abs(forces.z[i] - az[i]) / scale, 1.0e-4) << label;
                    }

                    const RunState fused = run_fused(state, 3, tile, blocking_factor, block);
                    expect_same_run_state(run_fused(state, 3, tile, 1, block), fused,
                                          "fused " + label);
                }
            }
        }
    }
}

// The self interaction lands on one of a thread's targets now, not on the thread's own
// index. With a single massive particle in the slot of each target in turn, that
// particle has only its own pair acting on it, which has to come to exactly zero,
// while everything else around it feels a finite pull.
TEST(KernelParity, SelfInteractionAddsNothingOnEveryTarget) {
    SKIP_WITHOUT_GPU();
    const int count = 37;
    for (const int blocking_factor : BLOCKING_FACTORS) {
        for (int heavy = 0; heavy < blocking_factor; ++heavy) {
            cps::HostParticleState state = random_cloud(count, 1999u);
            for (int i = 0; i < count; ++i) {
                state.mass[static_cast<std::size_t>(i)] = i == heavy ? 1.0f : 0.0f;
            }
            for (const int tile : {32, 256}) {
                const std::string label = describe(tile, blocking_factor, 64, count) + " heavy " +
                                          std::to_string(heavy);
                const Forces forces = tiled_forces(state, tile, blocking_factor, 64);
                for (std::size_t i = 0; i < forces.x.size(); ++i) {
                    ASSERT_TRUE(std::isfinite(forces.x[i])) << label << " index " << i;
                    ASSERT_TRUE(std::isfinite(forces.y[i])) << label << " index " << i;
                    ASSERT_TRUE(std::isfinite(forces.z[i])) << label << " index " << i;
                }
                const std::size_t own = static_cast<std::size_t>(heavy);
                EXPECT_EQ(forces.x[own], 0.0f) << label;
                EXPECT_EQ(forces.y[own], 0.0f) << label;
                EXPECT_EQ(forces.z[own], 0.0f) << label;
                const std::size_t other = own == 0u ? 1u : 0u;
                EXPECT_NE(forces.x[other], 0.0f) << label;
            }
        }
    }
}

// The launch shape must not reach the answer, so every block size gives what 256 gives.
// Odd multiples of 32 are here because the configuration allows them, and they run the
// unbounded instantiation. A large block with a high factor may not fit on an SM, and
// that refusal has to say which two knobs did it. It comes before any launch, so a
// memory checker run over this test sees no failed launch either.
TEST(KernelParity, EveryBlockSizeIsBitIdenticalOrRefusedByName) {
    SKIP_WITHOUT_GPU();
    const int count = 777;
    const cps::HostParticleState state = random_cloud(count, 496u);
    for (const int tile : {32, 256}) {
        const Forces reference = tiled_forces(state, tile, 1, 256);
        for (const int block : {32, 64, 96, 128, 160, 256, 288, 512, 768, 992, 1024}) {
            expect_same_forces(reference, tiled_forces(state, tile, 1, block),
                               describe(tile, 1, block, count));
            for (const int blocking_factor : {2, 4, 8}) {
                const std::string label = describe(tile, blocking_factor, block, count);
                try {
                    expect_same_forces(reference, tiled_forces(state, tile, blocking_factor, block),
                                       label);
                } catch (const cps::CudaError& error) {
                    // The default block must never come to this.
                    EXPECT_GT(block, 256) << label << ": " << error.what();
                    const std::string message = error.what();
                    EXPECT_NE(message.find("block_size " + std::to_string(block)),
                              std::string::npos)
                        << message;
                    EXPECT_NE(message.find("blocking factor " + std::to_string(blocking_factor)),
                              std::string::npos)
                        << message;
                }
            }
        }
    }
}

// A launch on fewer blocks than it needs is the stride loop's whole purpose: a block
// makes several passes, at each of which it must stage every tile again.
TEST(KernelParity, ASmallerGridStridesOverEveryParticle) {
    SKIP_WITHOUT_GPU();
    const int count = 733;
    const cps::HostParticleState state = random_cloud(count, 8191u);
    for (const int block : {64, 256}) {
        for (const int tile : {32, 256}) {
            for (const int blocking_factor : BLOCKING_FACTORS) {
                const Forces forces = tiled_forces(state, tile, blocking_factor, block);
                const RunState fused = run_fused(state, 2, tile, blocking_factor, block);
                for (const int max_grid : {1, 2, 3}) {
                    const std::string label = describe(tile, blocking_factor, block, count) +
                                              " grid " + std::to_string(max_grid);
                    expect_same_forces(forces,
                                       tiled_forces(state, tile, blocking_factor, block, max_grid),
                                       label);
                    expect_same_run_state(
                        fused, run_fused(state, 2, tile, blocking_factor, block, max_grid),
                        "fused " + label);
                }
            }
        }
    }
}

TEST(KernelParity, RefusesABlockingFactorWithNoKernel) {
    SKIP_WITHOUT_GPU();
    const int count = 64;
    cps::ParticleSystem particles(count);
    particles.upload(random_cloud(count, 1u), nullptr);
    for (const int blocking_factor : {-1, 0, 3, 16}) {
        SCOPED_TRACE(blocking_factor);
        const auto expect_named = [&](const auto& launch) {
            try {
                launch();
                ADD_FAILURE() << "nothing refused it";
            } catch (const std::runtime_error& error) {
                const std::string message = error.what();
                EXPECT_NE(message.find("unsupported blocking factor " +
                                       std::to_string(blocking_factor) +
                                       ", expected 1, 2, 4 or 8"),
                          std::string::npos)
                    << message;
            }
        };
        expect_named([&] {
            cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                                      particles.ax(), particles.ay(), particles.az(), count,
                                      SOFTENING, GRAVITY, 64, 64, blocking_factor, nullptr);
        });
        expect_named([&] {
            cps::nbody_fused_step(particles.x(), particles.y(), particles.z(), particles.x_next(),
                                  particles.y_next(), particles.z_next(), particles.vx(),
                                  particles.vy(), particles.vz(), particles.ax(), particles.ay(),
                                  particles.az(), particles.mass(), count, SOFTENING, GRAVITY, DT,
                                  64, 64, blocking_factor, nullptr);
        });
    }
}

}  // namespace
