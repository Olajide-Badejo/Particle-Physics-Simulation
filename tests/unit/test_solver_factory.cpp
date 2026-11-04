#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "cps/solvers/nbody_solver.hpp"
#include "cps/solvers/solver_factory.hpp"
#include "cps/solvers/sph_solver.hpp"
#include "test_support_gpu.hpp"

namespace {

cps::Config nbody_config(cps::NBodyVariant variant) {
    cps::Config config;
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = 64;
    config.softening = 0.05f;
    return config;
}

cps::Config sph_config() {
    cps::Config config;
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = 512;
    config.dt = 1.0e-4f;
    config.block_size = 128;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    return config;
}

std::string factory_error(const cps::Config& config) {
    try {
        static_cast<void>(cps::make_solver(config));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

// Splitting the velocity Verlet keeps the velocity synchronised, and the fused
// leapfrog carries it half a step behind. Getting this wrong for one variant moves
// its trajectory by half a step and its diagnostics by more.
TEST(SolverFactory, EachNBodyVariantGetsTheRightSolverAndConvention) {
    SKIP_WITHOUT_GPU();
    const struct {
        cps::NBodyVariant variant;
        cps::VelocityConvention convention;
    } cases[] = {{cps::NBodyVariant::Naive, cps::VelocityConvention::Synchronised},
                 {cps::NBodyVariant::Tiled, cps::VelocityConvention::Synchronised},
                 {cps::NBodyVariant::Fused, cps::VelocityConvention::HalfStep}};
    for (const auto& expected : cases) {
        SCOPED_TRACE(cps::to_string(expected.variant));
        const std::unique_ptr<cps::Solver> solver =
            cps::make_solver(nbody_config(expected.variant));
        const auto* nbody = dynamic_cast<const cps::NBodySolver*>(solver.get());
        ASSERT_NE(nbody, nullptr);
        EXPECT_EQ(nbody->variant(), expected.variant);
        EXPECT_EQ(solver->velocity_convention(), expected.convention);
        EXPECT_TRUE(solver->has_potential_energy());
        EXPECT_EQ(solver->startup_description(), "");
    }
}

// The fluid carries a half step velocity and has no gravitational potential.
TEST(SolverFactory, TheFluidGetsTheRightSolverAndConvention) {
    SKIP_WITHOUT_GPU();
    const std::unique_ptr<cps::Solver> solver = cps::make_solver(sph_config());
    ASSERT_NE(dynamic_cast<const cps::SphSolver*>(solver.get()), nullptr);
    EXPECT_EQ(solver->velocity_convention(), cps::VelocityConvention::HalfStep);
    EXPECT_FALSE(solver->has_potential_energy());
}

// Every width runs to the same bits, so nothing in a trajectory can show that the
// configured width reached the solver. This is what can.
TEST(SolverFactory, TheConfiguredTileWidthReachesTheSolver) {
    SKIP_WITHOUT_GPU();
    for (const int width : {32, 64, 128, 256}) {
        SCOPED_TRACE(width);
        cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
        config.nbody.tile_size = width;
        const std::unique_ptr<cps::Solver> solver = cps::make_solver(config);
        EXPECT_EQ(dynamic_cast<const cps::NBodySolver&>(*solver).tile_size(), width);
    }
}

// What main.cpp prints for the fluid, which has to name each limit.
TEST(SolverFactory, TheFluidDescribesItsResolutionAndItsLimits) {
    SKIP_WITHOUT_GPU();
    const std::string text = cps::make_solver(sph_config())->startup_description();
    for (const char* expected :
         {"  smoothing h   : ", "  particle mass : ", "  neighbours    : ",
          "  dt limit      : acoustic ", "  dt limit      : viscous ",
          "  dt limit      : body force "}) {
        EXPECT_NE(text.find(expected), std::string::npos) << expected << " in\n" << text;
    }
    EXPECT_EQ(text.back(), '\n');
}

TEST(SolverFactory, AnInvalidConfigurationGivesANamedError) {
    SKIP_WITHOUT_GPU();
    cps::Config tile = nbody_config(cps::NBodyVariant::Tiled);
    tile.nbody.tile_size = 100;
    EXPECT_NE(factory_error(tile).find("nbody.tile_size must be 32, 64, 128 or 256, got 100"),
              std::string::npos)
        << factory_error(tile);

    cps::Config softening = nbody_config(cps::NBodyVariant::Tiled);
    softening.softening = 0.0f;
    EXPECT_NE(factory_error(softening).find("softening"), std::string::npos)
        << factory_error(softening);
}

// The enums can only hold these values through a cast, but a value added to one
// without a case in the solver must not fall through to some other physics.
TEST(SolverFactory, AnUnknownSolverOrVariantGivesANamedError) {
    SKIP_WITHOUT_GPU();
    cps::Config solver = sph_config();
    solver.solver = static_cast<cps::SolverType>(99);
    EXPECT_NE(factory_error(solver).find("unknown solver type 99"), std::string::npos)
        << factory_error(solver);

    cps::Config variant = nbody_config(cps::NBodyVariant::Tiled);
    variant.variant = static_cast<cps::NBodyVariant>(99);
    EXPECT_NE(factory_error(variant).find("unknown N-body variant 99"), std::string::npos)
        << factory_error(variant);
}

// The solver's own refusals come through the factory unchanged.
TEST(SolverFactory, TheFluidRefusesATooLargeTimeStepThroughTheFactory) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config();
    config.dt = 1.0f;
    const std::string message = factory_error(config);
    EXPECT_NE(message.find("exceeds the"), std::string::npos) << message;
    EXPECT_NE(message.find("--allow-unstable-dt"), std::string::npos) << message;
}

}  // namespace
