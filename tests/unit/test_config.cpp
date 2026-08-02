#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>

#include "core/config.hpp"

namespace {

TEST(Config, DefaultsAreUsableWithoutAnyFields) {
    const cps::Config config = cps::Config::from_json_string("{}");
    EXPECT_EQ(config.solver, cps::SolverType::NBody);
    EXPECT_EQ(config.variant, cps::NBodyVariant::Tiled);
    EXPECT_EQ(config.initial_condition, cps::InitialCondition::Plummer);
    EXPECT_GT(config.num_particles, 0);
    EXPECT_GT(config.dt, 0.0f);
}

TEST(Config, ParsesEveryTopLevelField) {
    const std::string text = R"({
        "name": "example",
        "solver": "nbody",
        "variant": "fused",
        "initial_condition": "plummer",
        "num_particles": 4096,
        "steps": 250,
        "dt": 0.002,
        "softening": 0.05,
        "gravitational_constant": 2.5,
        "seed": 99,
        "block_size": 128,
        "total_mass": 3.0,
        "plummer_radius": 1.5,
        "snapshot_interval": 40,
        "snapshot_dir": "snaps",
        "diagnostics_interval": 5,
        "diagnostics_path": "diag.csv"
    })";
    const cps::Config config = cps::Config::from_json_string(text);
    EXPECT_EQ(config.name, "example");
    EXPECT_EQ(config.variant, cps::NBodyVariant::Fused);
    EXPECT_EQ(config.num_particles, 4096);
    EXPECT_EQ(config.steps, 250);
    EXPECT_FLOAT_EQ(config.dt, 0.002f);
    EXPECT_FLOAT_EQ(config.softening, 0.05f);
    EXPECT_FLOAT_EQ(config.gravitational_constant, 2.5f);
    EXPECT_EQ(config.seed, 99u);
    EXPECT_EQ(config.block_size, 128);
    EXPECT_EQ(config.snapshot_interval, 40);
    EXPECT_EQ(config.snapshot_dir, "snaps");
    EXPECT_EQ(config.diagnostics_interval, 5);
    EXPECT_EQ(config.diagnostics_path, "diag.csv");
}

TEST(Config, ParsesNestedFluidBlock) {
    const std::string text = R"({
        "solver": "sph",
        "initial_condition": "dam_break",
        "num_particles": 1000,
        "sph": {
            "rest_density": 998.0,
            "gas_constant": 1500.0,
            "viscosity": 0.35,
            "domain_min": [0.0, 0.0, 0.0],
            "domain_max": [2.0, 1.0, 1.0],
            "fluid_min": [0.1, 0.0, 0.0],
            "fluid_max": [0.5, 0.6, 1.0]
        }
    })";
    const cps::Config config = cps::Config::from_json_string(text);
    EXPECT_EQ(config.solver, cps::SolverType::Sph);
    EXPECT_FLOAT_EQ(config.sph.rest_density, 998.0f);
    EXPECT_FLOAT_EQ(config.sph.gas_constant, 1500.0f);
    EXPECT_FLOAT_EQ(config.sph.viscosity, 0.35f);
    EXPECT_FLOAT_EQ(config.sph.domain_max.x, 2.0f);
    EXPECT_FLOAT_EQ(config.sph.fluid_max.y, 0.6f);
}

TEST(Config, RejectsMalformedJson) {
    EXPECT_THROW(cps::Config::from_json_string("{ not json"), std::runtime_error);
}

TEST(Config, RejectsUnknownEnumerationValues) {
    EXPECT_THROW(cps::Config::from_json_string(R"({"solver": "barnes_hut"})"), std::runtime_error);
    EXPECT_THROW(cps::Config::from_json_string(R"({"variant": "magic"})"), std::runtime_error);
}

TEST(Config, RejectsBlockSizeThatIsNotAWarpMultiple) {
    EXPECT_THROW(cps::Config::from_json_string(R"({"block_size": 100})"), std::runtime_error);
    EXPECT_THROW(cps::Config::from_json_string(R"({"block_size": 2048})"), std::runtime_error);
}

TEST(Config, RejectsNonPositiveTimeStep) {
    EXPECT_THROW(cps::Config::from_json_string(R"({"dt": 0.0})"), std::runtime_error);
    EXPECT_THROW(cps::Config::from_json_string(R"({"dt": -0.001})"), std::runtime_error);
}

TEST(Config, RejectsMismatchedSolverAndInitialCondition) {
    EXPECT_THROW(cps::Config::from_json_string(
                     R"({"solver": "nbody", "initial_condition": "dam_break"})"),
                 std::runtime_error);
    EXPECT_THROW(
        cps::Config::from_json_string(R"({"solver": "sph", "initial_condition": "plummer"})"),
        std::runtime_error);
}

TEST(Config, TwoBodyRequiresExactlyTwoParticles) {
    EXPECT_THROW(cps::Config::from_json_string(
                     R"({"initial_condition": "two_body", "num_particles": 3})"),
                 std::runtime_error);
    EXPECT_NO_THROW(cps::Config::from_json_string(
        R"({"initial_condition": "two_body", "num_particles": 2})"));
}

TEST(Config, RejectsFluidBlockOutsideTheDomain) {
    const std::string text = R"({
        "solver": "sph",
        "initial_condition": "dam_break",
        "sph": {
            "domain_min": [0.0, 0.0, 0.0],
            "domain_max": [1.0, 1.0, 1.0],
            "fluid_min": [0.0, 0.0, 0.0],
            "fluid_max": [1.5, 0.5, 0.5]
        }
    })";
    EXPECT_THROW(cps::Config::from_json_string(text), std::runtime_error);
}

// Round tripping matters because the benchmark driver records the resolved
// configuration next to every result row.
TEST(Config, SurvivesARoundTripThroughJson) {
    const cps::Config original = cps::Config::from_json_string(R"({
        "name": "round_trip", "variant": "naive", "num_particles": 777,
        "dt": 0.0025, "softening": 0.02, "seed": 12345
    })");
    const cps::Config restored = cps::Config::from_json_string(original.to_json_string());
    EXPECT_EQ(restored.name, original.name);
    EXPECT_EQ(restored.variant, original.variant);
    EXPECT_EQ(restored.num_particles, original.num_particles);
    EXPECT_FLOAT_EQ(restored.dt, original.dt);
    EXPECT_FLOAT_EQ(restored.softening, original.softening);
    EXPECT_EQ(restored.seed, original.seed);
}

// The three shipped configurations are part of the reproducibility promise, so
// a broken one has to fail here rather than half an hour into a sweep.
TEST(Config, ShippedConfigurationsParse) {
    const std::filesystem::path root(CPS_PROJECT_ROOT);
    for (const char* name : {"two_body.json", "plummer_10k.json", "dam_break_50k.json"}) {
        const std::filesystem::path path = root / "configs" / name;
        ASSERT_TRUE(std::filesystem::exists(path)) << path.string();
        EXPECT_NO_THROW(cps::Config::from_json_file(path)) << name;
    }
}

}  // namespace
