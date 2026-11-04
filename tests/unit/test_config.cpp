#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/config.hpp"
#include "cps/core/physics_fingerprint.hpp"

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

// A config that omits a field gets these values, so they are part of the
// interface. The diagnostics cadence in particular is sparse by default
// because each row costs an O(N^2) potential pass.
TEST(Config, DefaultsMatchTheDocumentedValues) {
    const cps::Config config;
    EXPECT_EQ(config.name, "unnamed");
    EXPECT_EQ(config.num_particles, 10000);
    EXPECT_EQ(config.steps, 1000);
    EXPECT_FLOAT_EQ(config.dt, 1.0e-3f);
    EXPECT_FLOAT_EQ(config.softening, 0.01f);
    EXPECT_FLOAT_EQ(config.gravitational_constant, 1.0f);
    EXPECT_EQ(config.seed, 20260802u);
    EXPECT_EQ(config.block_size, 256);
    EXPECT_EQ(config.snapshot_interval, 1000);
    EXPECT_EQ(config.diagnostics_interval, 100);
    EXPECT_EQ(config.diagnostics_path, "experiments/results/diagnostics.csv");
    EXPECT_FLOAT_EQ(config.sph.boundary_damping, 0.4f);
    EXPECT_FLOAT_EQ(config.sph.jitter, 0.05f);
    EXPECT_NO_THROW(config.validate());

    const cps::Config parsed = cps::Config::from_json_string("{}");
    EXPECT_EQ(parsed.to_json_string(), config.to_json_string());
}

// Each case is one bad value for one field. The message must name the field,
// so the case also carries the text the error has to contain.
struct InvalidJsonCase {
    const char* label;
    const char* json;
    const char* expected_in_message;
};

void PrintTo(const InvalidJsonCase& value, std::ostream* out) { *out << value.label; }

class ConfigRejectsInvalidJson : public ::testing::TestWithParam<InvalidJsonCase> {};

TEST_P(ConfigRejectsInvalidJson, NamingTheField) {
    const InvalidJsonCase& param = GetParam();
    try {
        cps::Config::from_json_string(param.json);
        FAIL() << "expected an error for " << param.json;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(param.expected_in_message), std::string::npos)
            << "message was: " << error.what();
    }
}

// JSON has no NaN or Inf literal. The nearest it gets is 1e39, which is beyond
// the float range, so the reader rejects it before narrowing.
INSTANTIATE_TEST_SUITE_P(
    Fields, ConfigRejectsInvalidJson,
    ::testing::Values(
        InvalidJsonCase{"num_particles_one", R"({"num_particles": 1})", "num_particles"},
        InvalidJsonCase{"num_particles_zero", R"({"num_particles": 0})", "num_particles"},
        InvalidJsonCase{"num_particles_negative", R"({"num_particles": -5})", "num_particles"},
        InvalidJsonCase{"num_particles_fractional", R"({"num_particles": 10.5})", "num_particles"},
        InvalidJsonCase{"num_particles_above_int", R"({"num_particles": 3000000000})",
                        "num_particles"},
        InvalidJsonCase{"num_particles_string", R"({"num_particles": "100"})", "num_particles"},
        InvalidJsonCase{"steps_negative", R"({"steps": -1})", "steps"},
        InvalidJsonCase{"steps_fractional", R"({"steps": 1.5})", "steps"},
        InvalidJsonCase{"dt_zero", R"({"dt": 0.0})", "dt"},
        InvalidJsonCase{"dt_negative", R"({"dt": -0.001})", "dt"},
        InvalidJsonCase{"dt_infinite", R"({"dt": 1e39})", "dt"},
        InvalidJsonCase{"dt_string", R"({"dt": "0.1"})", "dt"},
        InvalidJsonCase{"softening_negative", R"({"softening": -0.01})", "softening"},
        InvalidJsonCase{"softening_zero_for_nbody", R"({"softening": 0})", "softening"},
        InvalidJsonCase{"softening_infinite", R"({"softening": 1e39})", "softening"},
        InvalidJsonCase{"gravitational_constant_zero", R"({"gravitational_constant": 0})",
                        "gravitational_constant"},
        InvalidJsonCase{"gravitational_constant_negative", R"({"gravitational_constant": -1})",
                        "gravitational_constant"},
        InvalidJsonCase{"seed_negative", R"({"seed": -1})", "seed"},
        InvalidJsonCase{"seed_above_uint32", R"({"seed": 4294967296})", "seed"},
        InvalidJsonCase{"seed_fractional", R"({"seed": 7.5})", "seed"},
        InvalidJsonCase{"block_size_zero", R"({"block_size": 0})", "block_size"},
        InvalidJsonCase{"block_size_not_warp_multiple", R"({"block_size": 100})", "block_size"},
        InvalidJsonCase{"block_size_too_large", R"({"block_size": 2048})", "block_size"},
        InvalidJsonCase{"total_mass_zero", R"({"total_mass": 0})", "total_mass"},
        InvalidJsonCase{"total_mass_negative", R"({"total_mass": -2})", "total_mass"},
        InvalidJsonCase{"plummer_radius_zero", R"({"plummer_radius": 0})", "plummer_radius"},
        InvalidJsonCase{"plummer_radius_negative", R"({"plummer_radius": -1})", "plummer_radius"},
        InvalidJsonCase{"two_body_separation_zero", R"({"two_body_separation": 0})",
                        "two_body_separation"},
        InvalidJsonCase{"two_body_separation_negative", R"({"two_body_separation": -1})",
                        "two_body_separation"},
        InvalidJsonCase{"external_gravity_infinite", R"({"external_gravity": [0, 1e39, 0]})",
                        "external_gravity"},
        InvalidJsonCase{"external_gravity_short", R"({"external_gravity": [0, 1]})",
                        "external_gravity"},
        InvalidJsonCase{"external_gravity_string", R"({"external_gravity": [0, "a", 0]})",
                        "external_gravity[1]"},
        InvalidJsonCase{"snapshot_interval_negative", R"({"snapshot_interval": -1})",
                        "snapshot_interval"},
        InvalidJsonCase{"diagnostics_interval_negative", R"({"diagnostics_interval": -1})",
                        "diagnostics_interval"},
        InvalidJsonCase{"diagnostics_interval_fractional", R"({"diagnostics_interval": 2.5})",
                        "diagnostics_interval"},
        InvalidJsonCase{"name_not_string", R"({"name": 3})", "name"},
        InvalidJsonCase{"sph_not_object", R"({"sph": 3})", "sph"},
        InvalidJsonCase{"smoothing_radius_negative", R"({"sph": {"smoothing_radius": -0.1}})",
                        "sph.smoothing_radius"},
        InvalidJsonCase{"smoothing_radius_infinite", R"({"sph": {"smoothing_radius": 1e39}})",
                        "sph.smoothing_radius"},
        InvalidJsonCase{"rest_density_zero", R"({"sph": {"rest_density": 0}})", "sph.rest_density"},
        InvalidJsonCase{"rest_density_negative", R"({"sph": {"rest_density": -1}})",
                        "sph.rest_density"},
        InvalidJsonCase{"gas_constant_zero", R"({"sph": {"gas_constant": 0}})", "sph.gas_constant"},
        InvalidJsonCase{"viscosity_negative", R"({"sph": {"viscosity": -0.1}})", "sph.viscosity"},
        InvalidJsonCase{"viscosity_infinite", R"({"sph": {"viscosity": 1e39}})", "sph.viscosity"},
        InvalidJsonCase{"particle_mass_negative", R"({"sph": {"particle_mass": -1}})",
                        "sph.particle_mass"},
        InvalidJsonCase{"boundary_damping_negative", R"({"sph": {"boundary_damping": -0.1}})",
                        "sph.boundary_damping"},
        InvalidJsonCase{"boundary_damping_above_one", R"({"sph": {"boundary_damping": 1.5}})",
                        "sph.boundary_damping"},
        InvalidJsonCase{"jitter_negative", R"({"sph": {"jitter": -0.01}})", "sph.jitter"},
        InvalidJsonCase{"jitter_half", R"({"sph": {"jitter": 0.5}})", "sph.jitter"},
        InvalidJsonCase{"domain_inverted",
                        R"({"sph": {"domain_min": [1, 0, 0], "domain_max": [0, 1, 1]}})",
                        "sph.domain_min"},
        InvalidJsonCase{"domain_flat",
                        R"({"sph": {"domain_min": [0, 0, 0], "domain_max": [1, 0, 1]}})",
                        "sph.domain_min"},
        InvalidJsonCase{"domain_infinite", R"({"sph": {"domain_max": [1, 1e39, 1]}})",
                        "sph.domain_max"},
        InvalidJsonCase{"fluid_inverted",
                        R"({"sph": {"fluid_min": [0.4, 0, 0], "fluid_max": [0.1, 0.5, 0.5]}})",
                        "sph.fluid_min"},
        InvalidJsonCase{"fluid_outside_domain", R"({"sph": {"fluid_max": [1.5, 0.5, 0.5]}})",
                        "inside the domain"}),
    [](const ::testing::TestParamInfo<InvalidJsonCase>& info) { return std::string(info.param.label); });

// JSON cannot spell NaN, but a Config built in code or changed by a command
// line override can hold one, so validate() is checked directly.
struct NanCase {
    const char* label;
    std::function<void(cps::Config&)> poison;
    const char* expected_in_message;
};

void PrintTo(const NanCase& value, std::ostream* out) { *out << value.label; }

class ConfigRejectsNan : public ::testing::TestWithParam<NanCase> {};

TEST_P(ConfigRejectsNan, NamingTheField) {
    cps::Config config;
    ASSERT_NO_THROW(config.validate());
    GetParam().poison(config);
    try {
        config.validate();
        FAIL() << "expected an error";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(GetParam().expected_in_message), std::string::npos)
            << "message was: " << message;
        EXPECT_NE(message.find("nan"), std::string::npos) << "message was: " << message;
    }
}

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

INSTANTIATE_TEST_SUITE_P(
    Fields, ConfigRejectsNan,
    ::testing::Values(
        NanCase{"dt", [](cps::Config& c) { c.dt = kNan; }, "dt"},
        NanCase{"softening", [](cps::Config& c) { c.softening = kNan; }, "softening"},
        NanCase{"gravitational_constant", [](cps::Config& c) { c.gravitational_constant = kNan; },
                "gravitational_constant"},
        NanCase{"total_mass", [](cps::Config& c) { c.total_mass = kNan; }, "total_mass"},
        NanCase{"plummer_radius", [](cps::Config& c) { c.plummer_radius = kNan; },
                "plummer_radius"},
        NanCase{"two_body_separation", [](cps::Config& c) { c.two_body_separation = kNan; },
                "two_body_separation"},
        NanCase{"external_gravity", [](cps::Config& c) { c.external_gravity.y = kNan; },
                "external_gravity"},
        NanCase{"smoothing_radius", [](cps::Config& c) { c.sph.smoothing_radius = kNan; },
                "sph.smoothing_radius"},
        NanCase{"rest_density", [](cps::Config& c) { c.sph.rest_density = kNan; },
                "sph.rest_density"},
        NanCase{"gas_constant", [](cps::Config& c) { c.sph.gas_constant = kNan; },
                "sph.gas_constant"},
        NanCase{"viscosity", [](cps::Config& c) { c.sph.viscosity = kNan; }, "sph.viscosity"},
        NanCase{"particle_mass", [](cps::Config& c) { c.sph.particle_mass = kNan; },
                "sph.particle_mass"},
        NanCase{"boundary_damping", [](cps::Config& c) { c.sph.boundary_damping = kNan; },
                "sph.boundary_damping"},
        NanCase{"jitter", [](cps::Config& c) { c.sph.jitter = kNan; }, "sph.jitter"},
        NanCase{"domain_min", [](cps::Config& c) { c.sph.domain_min.x = kNan; }, "sph.domain_min"},
        NanCase{"domain_max", [](cps::Config& c) { c.sph.domain_max.z = kNan; }, "sph.domain_max"},
        NanCase{"fluid_min", [](cps::Config& c) { c.sph.fluid_min.y = kNan; }, "sph.fluid_min"},
        NanCase{"fluid_max", [](cps::Config& c) { c.sph.fluid_max.x = kNan; }, "sph.fluid_max"}),
    [](const ::testing::TestParamInfo<NanCase>& info) { return std::string(info.param.label); });

TEST(Config, AcceptsTheEdgesOfEachRange) {
    EXPECT_NO_THROW(cps::Config::from_json_string(R"({"steps": 0})"));
    // The fluid solver has no self pair for softening to protect.
    EXPECT_NO_THROW(cps::Config::from_json_string(
        R"({"solver": "sph", "initial_condition": "dam_break", "softening": 0})"));
    EXPECT_NO_THROW(cps::Config::from_json_string(
        R"({"snapshot_interval": 0, "diagnostics_interval": 0, "seed": 4294967295})"));
    EXPECT_NO_THROW(cps::Config::from_json_string(
        R"({"sph": {"boundary_damping": 0, "jitter": 0, "viscosity": 0}})"));
    EXPECT_NO_THROW(cps::Config::from_json_string(
        R"({"sph": {"boundary_damping": 1, "jitter": 0.49, "smoothing_radius": 0}})"));
    // An integer written with an exponent or a trailing .0 is still a float in
    // JSON, so it is rejected for an integer field like any other fraction.
    EXPECT_THROW(cps::Config::from_json_string(R"({"num_particles": 1000.0})"), std::runtime_error);
}

// The library itself truncates a fractional number when asked for an int.
// This is why integer fields check is_number_integer() before converting.
TEST(Config, NlohmannTruncatesFractionalNumbersToInt) {
    const nlohmann::json value = nlohmann::json::parse("10.7");
    EXPECT_FALSE(value.is_number_integer());
    EXPECT_EQ(value.get<int>(), 10);
}

std::string error_of(const std::string& text) {
    try {
        cps::Config::from_json_string(text);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

TEST(Config, RejectsMisspelledTopLevelKey) {
    const std::string message = error_of(R"({"num_particle": 100})");
    EXPECT_NE(message.find("'num_particle'"), std::string::npos) << message;
    EXPECT_NE(message.find("did you mean 'initial_condition.num_particles'"), std::string::npos)
        << message;

    // A fluid key at the top level points at the fluid block.
    const std::string misplaced = error_of(R"({"viscocity": 1})");
    EXPECT_NE(misplaced.find("'viscocity'"), std::string::npos) << misplaced;
    EXPECT_NE(misplaced.find("did you mean 'sph.viscosity'"), std::string::npos) << misplaced;
}

TEST(Config, RejectsMisspelledKeyInsideTheFluidBlock) {
    const std::string message = error_of(R"({"sph": {"viscocity": 0.3}})");
    EXPECT_NE(message.find("'sph.viscocity'"), std::string::npos) << message;
    EXPECT_NE(message.find("did you mean 'viscosity'"), std::string::npos) << message;
}

TEST(Config, RejectsUnexpectedNestedObject) {
    const std::string top = error_of(R"({"gravity": {"dt": 0.1}})");
    EXPECT_NE(top.find("unknown configuration key 'gravity'"), std::string::npos) << top;

    const std::string nested = error_of(R"({"sph": {"kernel": {"name": "wendland"}}})");
    EXPECT_NE(nested.find("unknown configuration key 'sph.kernel'"), std::string::npos) << nested;

    // An object where a number belongs is a type error naming the field.
    const std::string misplaced = error_of(R"({"dt": {"value": 0.1}})");
    EXPECT_NE(misplaced.find("'run.dt'"), std::string::npos) << misplaced;
}

// An unknown key in the N-body block is named with its full path, and a tile
// width left at the top level points at its home.
TEST(Config, ReadsTheTileWidthFromTheNBodyBlock) {
    EXPECT_EQ(cps::Config::from_json_string("{}").nbody.tile_size, cps::DEFAULT_TILE_SIZE);
    for (const int width : {32, 64, 128, 256}) {
        SCOPED_TRACE(width);
        const std::string text = R"({"nbody": {"tile_size": )" + std::to_string(width) + "}}";
        const cps::Config config = cps::Config::from_json_string(text);
        EXPECT_EQ(config.nbody.tile_size, width);
        EXPECT_EQ(cps::Config::from_json_string(config.to_json_string()).nbody.tile_size, width);
    }
}

TEST(Config, RejectsATileWidthWithNoKernel) {
    for (const char* width : {"0", "-32", "16", "100", "512"}) {
        SCOPED_TRACE(width);
        const std::string message =
            error_of(R"({"nbody": {"tile_size": )" + std::string(width) + "}}");
        EXPECT_NE(message.find("nbody.tile_size must be 32, 64, 128 or 256, got " +
                               std::string(width)),
                  std::string::npos)
            << message;
    }
    const std::string fractional = error_of(R"({"nbody": {"tile_size": 64.5}})");
    EXPECT_NE(fractional.find("'nbody.tile_size' must be an integer"), std::string::npos)
        << fractional;
}

TEST(Config, RejectsAnNBodyBlockThatIsNotAnObject) {
    const std::string message = error_of(R"({"nbody": 64})");
    EXPECT_NE(message.find("configuration field 'nbody' must be an object"), std::string::npos)
        << message;
}

TEST(Config, RejectsMisspelledKeyInsideTheNBodyBlock) {
    const std::string message = error_of(R"({"nbody": {"tile_sise": 64}})");
    EXPECT_NE(message.find("'nbody.tile_sise'"), std::string::npos) << message;
    EXPECT_NE(message.find("did you mean 'tile_size'"), std::string::npos) << message;

    const std::string misplaced = error_of(R"({"nbody": {"dt": 0.1}})");
    EXPECT_NE(misplaced.find("unknown configuration key 'nbody.dt'"), std::string::npos)
        << misplaced;
}

TEST(Config, ATileWidthAtTheTopLevelPointsAtTheNBodyBlock) {
    const std::string message = error_of(R"({"tile_size": 64})");
    EXPECT_NE(message.find("unknown configuration key 'tile_size'"), std::string::npos) << message;
    EXPECT_NE(message.find("did you mean 'nbody.tile_size'"), std::string::npos) << message;
}

// A block nothing reads is still refused by its own name.
TEST(Config, RejectsAnUnknownBlockAtTheTopLevel) {
    const std::string message = error_of(R"({"gravity": {"dt": 0.1}})");
    EXPECT_NE(message.find("unknown configuration key 'gravity'"), std::string::npos) << message;
}

TEST(Config, UnrelatedUnknownKeyGetsNoSuggestion) {
    const std::string message = error_of(R"({"comment": "hello"})");
    EXPECT_NE(message.find("'comment'"), std::string::npos) << message;
    EXPECT_EQ(message.find("did you mean"), std::string::npos) << message;
}

TEST(Config, AllowUnstableDtMustBeABoolean) {
    EXPECT_FALSE(cps::Config::from_json_string("{}").allow_unstable_dt);
    EXPECT_TRUE(cps::Config::from_json_string(R"({"allow_unstable_dt": true})").allow_unstable_dt);
    EXPECT_THROW(cps::Config::from_json_string(R"({"allow_unstable_dt": 1})"), std::runtime_error);
    EXPECT_THROW(cps::Config::from_json_string(R"({"allow_unstable_dt": "yes"})"),
                 std::runtime_error);
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

void expect_same_vec3(const cps::Vec3& a, const cps::Vec3& b, const char* field) {
    EXPECT_EQ(a.x, b.x) << field;
    EXPECT_EQ(a.y, b.y) << field;
    EXPECT_EQ(a.z, b.z) << field;
}

// Exact equality on every field. The serialised form is the canonical
// fingerprint of a run, so nothing may be lost or rounded on the way through.
void expect_same_config(const cps::Config& a, const cps::Config& b) {
    EXPECT_EQ(a.allow_unstable_dt, b.allow_unstable_dt);
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.solver, b.solver);
    EXPECT_EQ(a.variant, b.variant);
    EXPECT_EQ(a.initial_condition, b.initial_condition);
    EXPECT_EQ(a.num_particles, b.num_particles);
    EXPECT_EQ(a.steps, b.steps);
    EXPECT_EQ(a.dt, b.dt);
    EXPECT_EQ(a.softening, b.softening);
    EXPECT_EQ(a.gravitational_constant, b.gravitational_constant);
    EXPECT_EQ(a.seed, b.seed);
    EXPECT_EQ(a.block_size, b.block_size);
    EXPECT_EQ(a.total_mass, b.total_mass);
    EXPECT_EQ(a.plummer_radius, b.plummer_radius);
    EXPECT_EQ(a.two_body_separation, b.two_body_separation);
    expect_same_vec3(a.external_gravity, b.external_gravity, "external_gravity");
    EXPECT_EQ(a.snapshot_interval, b.snapshot_interval);
    EXPECT_EQ(a.snapshot_dir, b.snapshot_dir);
    EXPECT_EQ(a.diagnostics_interval, b.diagnostics_interval);
    EXPECT_EQ(a.diagnostics_path, b.diagnostics_path);
    EXPECT_EQ(a.nbody.tile_size, b.nbody.tile_size);
    EXPECT_EQ(a.sph.smoothing_radius, b.sph.smoothing_radius);
    EXPECT_EQ(a.sph.rest_density, b.sph.rest_density);
    EXPECT_EQ(a.sph.gas_constant, b.sph.gas_constant);
    EXPECT_EQ(a.sph.viscosity, b.sph.viscosity);
    EXPECT_EQ(a.sph.particle_mass, b.sph.particle_mass);
    EXPECT_EQ(a.sph.boundary_damping, b.sph.boundary_damping);
    EXPECT_EQ(a.sph.jitter, b.sph.jitter);
    expect_same_vec3(a.sph.domain_min, b.sph.domain_min, "sph.domain_min");
    expect_same_vec3(a.sph.domain_max, b.sph.domain_max, "sph.domain_max");
    expect_same_vec3(a.sph.fluid_min, b.sph.fluid_min, "sph.fluid_min");
    expect_same_vec3(a.sph.fluid_max, b.sph.fluid_max, "sph.fluid_max");
}

// Every field is set away from its default, with values such as 0.1 that have
// no exact binary form, so a lossy or partial serialisation shows up.
TEST(Config, RoundTripsEveryFieldExactly) {
    const cps::Config original = cps::Config::from_json_string(R"({
        "name": "every_field", "solver": "sph", "variant": "fused",
        "initial_condition": "dam_break", "num_particles": 12345, "steps": 9876543210,
        "dt": 0.0001234, "allow_unstable_dt": true, "softening": 0.1,
        "gravitational_constant": 6.674e-3,
        "seed": 4294967295, "block_size": 512, "total_mass": 3.3,
        "plummer_radius": 0.7, "two_body_separation": 2.2,
        "external_gravity": [0.1, -9.80665, 0.3], "snapshot_interval": 77,
        "snapshot_dir": "snaps/x", "diagnostics_interval": 13, "diagnostics_path": "d.csv",
        "nbody": {"tile_size": 64},
        "sph": {
            "smoothing_radius": 0.0123, "rest_density": 998.2, "gas_constant": 1234.5,
            "viscosity": 0.31, "particle_mass": 0.0042, "boundary_damping": 0.9,
            "jitter": 0.11, "domain_min": [-0.1, -0.2, -0.3], "domain_max": [1.1, 1.2, 1.3],
            "fluid_min": [0.0, 0.1, 0.2], "fluid_max": [0.3, 0.4, 0.5]
        }
    })");
    const std::string once = original.to_json_string();
    const cps::Config restored = cps::Config::from_json_string(once);
    expect_same_config(original, restored);
    EXPECT_EQ(restored.to_json_string(), once);
    // The compact form carries the same content.
    expect_same_config(original, cps::Config::from_json_string(original.to_json_string(-1)));
}

TEST(Config, ShippedConfigurationsRoundTrip) {
    const std::filesystem::path root(CPS_PROJECT_ROOT);
    for (const char* name : {"two_body.json", "plummer_10k.json", "dam_break_50k.json"}) {
        SCOPED_TRACE(name);
        const cps::Config original = cps::Config::from_json_file(root / "presets" / name);
        const cps::Config restored = cps::Config::from_json_string(original.to_json_string());
        expect_same_config(original, restored);
        EXPECT_EQ(restored.to_json_string(), original.to_json_string());
    }
}

// The three shipped configurations are part of the reproducibility promise, so
// a broken one has to fail here rather than half an hour into a sweep.
TEST(Config, ShippedConfigurationsParse) {
    const std::filesystem::path root(CPS_PROJECT_ROOT);
    for (const char* name : {"two_body.json", "plummer_10k.json", "dam_break_50k.json"}) {
        const std::filesystem::path path = root / "presets" / name;
        ASSERT_TRUE(std::filesystem::exists(path)) << path.string();
        EXPECT_NO_THROW(cps::Config::from_json_file(path)) << name;
    }
}

constexpr const char* kPresetNames[] = {"two_body.json", "plummer_10k.json", "dam_break_50k.json"};

std::filesystem::path fixture_path(const char* name) {
    return std::filesystem::path(CPS_PROJECT_ROOT) / "tests" / "data" / "configs_v1" / name;
}

std::filesystem::path preset_path(const char* name) {
    return std::filesystem::path(CPS_PROJECT_ROOT) / "presets" / name;
}

std::string file_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream) << path.string();
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

// The same run in either form, down to the fingerprint a snapshot records.
TEST(Config, V1AndV2FormsHaveTheSameFingerprint) {
    for (const char* name : kPresetNames) {
        SCOPED_TRACE(name);
        const cps::Config v1 = cps::Config::from_json_file(fixture_path(name));
        const cps::Config v2 = cps::Config::from_json_file(preset_path(name));
        expect_same_config(v1, v2);
        EXPECT_EQ(cps::physics_fingerprint(v1), cps::physics_fingerprint(v2));
        EXPECT_EQ(cps::physics_json(cps::physics_fields(v1)),
                  cps::physics_json(cps::physics_fields(v2)));
    }
}

// A preset is what the migration produces, not a second hand maintained copy.
// The file ends in the newline --print-config adds.
TEST(Config, PresetsAreExactlyTheMigratedFixtures) {
    for (const char* name : kPresetNames) {
        SCOPED_TRACE(name);
        EXPECT_EQ(cps::Config::from_json_file(fixture_path(name)).to_json_string() + "\n",
                  file_text(preset_path(name)));
    }
}

TEST(Config, PrintingAPresetGivesBackItsBytes) {
    for (const char* name : kPresetNames) {
        SCOPED_TRACE(name);
        const std::string text = file_text(preset_path(name));
        EXPECT_EQ(cps::Config::from_json_string(text).to_json_string() + "\n", text);
    }
}

TEST(Config, PrintsEverySectionInOrderWithNothingOmitted) {
    const nlohmann::ordered_json root =
        nlohmann::ordered_json::parse(cps::Config{}.to_json_string());
    std::vector<std::string> keys;
    for (const auto& item : root.items()) {
        keys.push_back(item.key());
    }
    EXPECT_EQ(keys, (std::vector<std::string>{"schema", "name", "solver", "run",
                                              "initial_condition", "nbody", "sph", "output"}));
    EXPECT_EQ(root.at("schema"), 2);
    EXPECT_EQ(root.at("run").size(), 4u);
    EXPECT_EQ(root.at("initial_condition").size(), 6u);
    EXPECT_EQ(root.at("nbody").size(), 4u);
    EXPECT_EQ(root.at("sph").size(), 12u);
    EXPECT_EQ(root.at("output").size(), 4u);
}

TEST(Config, PrintsAFloatAsItsShortestDecimal) {
    cps::Config config;
    config.dt = 1.0e-3f;
    config.softening = 0.1f;
    config.gravitational_constant = 1.0f;
    const std::string text = config.to_json_string();
    EXPECT_NE(text.find("\"dt\": 0.001,"), std::string::npos) << text;
    EXPECT_NE(text.find("\"softening\": 0.1,"), std::string::npos) << text;
    EXPECT_NE(text.find("\"gravitational_constant\": 1.0,"), std::string::npos) << text;
    EXPECT_NE(text.find("\"external_gravity\": [0.0, -9.81, 0.0]"), std::string::npos) << text;
    config.dt = 1.0e-4f;
    EXPECT_NE(config.to_json_string().find("\"dt\": 0.0001,"), std::string::npos);
}

void expect_floats_round_trip(const std::vector<float>& values) {
    for (const float value : values) {
        // External gravity is the one field that takes any finite value.
        cps::Config config;
        config.external_gravity = {value, -value, value};
        for (const int indent : {2, -1}) {
            const cps::Config restored =
                cps::Config::from_json_string(config.to_json_string(indent));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(restored.external_gravity.x),
                      std::bit_cast<std::uint32_t>(value))
                << value;
            EXPECT_EQ(std::bit_cast<std::uint32_t>(restored.external_gravity.y),
                      std::bit_cast<std::uint32_t>(-value))
                << value;
        }
    }
}

TEST(Config, FloatsRoundTripBitForBit) {
    using limits = std::numeric_limits<float>;
    expect_floats_round_trip({
        0.0f,
        0.1f,
        1.0e-3f,
        1.0e-4f,
        9.81f,
        1.0f,
        1000.0f,
        1.0e10f,
        16777216.0f,
        limits::denorm_min(),
        limits::denorm_min() * 3.0f,
        limits::min(),
        std::nextafter(limits::min(), 0.0f),
        limits::max(),
        std::nextafter(limits::max(), 0.0f),
        // The largest float's shortest text is out of range as a double. This one's
        // reads through a double to the neighbouring float.
        std::bit_cast<float>(std::uint32_t{0x15ae43fd}),
    });

    // Uniform bit patterns are uniform in magnitude across the whole range, and
    // a mask on the exponent adds denormals, which they otherwise rarely reach.
    std::mt19937 generator(20260802u);
    std::vector<float> sample;
    while (sample.size() < 4000) {
        std::uint32_t bits = generator();
        if (sample.size() % 8 == 0) {
            bits &= 0x807fffffu;
        }
        const float value = std::bit_cast<float>(bits);
        if (std::isfinite(value)) {
            sample.push_back(value);
        }
    }
    expect_floats_round_trip(sample);
}

TEST(Config, AFlatKeyInSchema2PointsAtItsSection) {
    const struct {
        const char* key;
        const char* value;
        const char* home;
    } cases[] = {
        {"dt", "0.1", "run.dt"},
        {"steps", "1", "run.steps"},
        {"num_particles", "4", "initial_condition.num_particles"},
        {"variant", "\"naive\"", "nbody.variant"},
        {"tile_size", "64", "nbody.tile_size"},
        {"external_gravity", "[0, 0, 0]", "sph.external_gravity"},
        {"viscosity", "0.1", "sph.viscosity"},
        {"snapshot_dir", "\"x\"", "output.snapshot_dir"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.key);
        const std::string message = error_of(std::string(R"({"schema": 2, ")") + c.key + "\": " +
                                             c.value + "}");
        EXPECT_NE(message.find("unknown configuration key '" + std::string(c.key) + "'"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("did you mean '" + std::string(c.home) + "'"), std::string::npos)
            << message;
    }
}

TEST(Config, AKeyInTheWrongSectionPointsAtItsSection) {
    const std::string message = error_of(R"({"schema": 2, "sph": {"dt": 0.1}})");
    EXPECT_NE(message.find("'sph.dt'"), std::string::npos) << message;
    EXPECT_NE(message.find("did you mean 'run.dt'"), std::string::npos) << message;
}

TEST(Config, RejectsAMisspelledKeyInEachSection) {
    const struct {
        const char* json;
        const char* key;
        const char* suggestion;
    } cases[] = {
        {R"({"schema": 2, "run": {"step": 1}})", "'run.step'", "'steps'"},
        {R"({"schema": 2, "initial_condition": {"sead": 1}})", "'initial_condition.sead'", "'seed'"},
        {R"({"schema": 2, "nbody": {"varient": "naive"}})", "'nbody.varient'", "'variant'"},
        {R"({"schema": 2, "sph": {"viscocity": 0.1}})", "'sph.viscocity'", "'viscosity'"},
        {R"({"schema": 2, "output": {"snapshot_dri": "x"}})", "'output.snapshot_dri'",
         "'snapshot_dir'"},
        {R"({"schema": 2, "nam": "x"})", "'nam'", "'name'"},
        {R"({"schema": 2, "runs": {}})", "'runs'", "'run'"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.json);
        const std::string message = error_of(c.json);
        EXPECT_NE(message.find(c.key), std::string::npos) << message;
        EXPECT_NE(message.find("did you mean " + std::string(c.suggestion)), std::string::npos)
            << message;
    }
}

TEST(Config, RejectsASectionThatIsNotAnObject) {
    for (const char* section : {"run", "initial_condition", "nbody", "sph", "output"}) {
        SCOPED_TRACE(section);
        const std::string message =
            error_of(std::string(R"({"schema": 2, ")") + section + R"(": 3})");
        EXPECT_NE(message.find("configuration field '" + std::string(section) +
                               "' must be an object"),
                  std::string::npos)
            << message;
    }
}

TEST(Config, ReadsEverySectionOfSchema2) {
    const cps::Config config = cps::Config::from_json_string(R"({
        "schema": 2, "name": "sections", "solver": "sph",
        "run": {"steps": 7, "dt": 0.002, "block_size": 64, "allow_unstable_dt": true},
        "initial_condition": {"type": "dam_break", "num_particles": 8, "seed": 9,
                              "total_mass": 2.0, "plummer_radius": 3.0, "two_body_separation": 4.0},
        "nbody": {"variant": "naive", "softening": 0.5, "gravitational_constant": 6.0,
                  "tile_size": 32},
        "sph": {"viscosity": 0.3, "external_gravity": [1.0, 2.0, 3.0]},
        "output": {"snapshot_interval": 10, "snapshot_dir": "s",
                   "diagnostics_interval": 11, "diagnostics_path": "d"}
    })");
    EXPECT_EQ(config.name, "sections");
    EXPECT_EQ(config.solver, cps::SolverType::Sph);
    EXPECT_EQ(config.steps, 7);
    EXPECT_EQ(config.dt, 0.002f);
    EXPECT_EQ(config.block_size, 64);
    EXPECT_TRUE(config.allow_unstable_dt);
    EXPECT_EQ(config.initial_condition, cps::InitialCondition::DamBreak);
    EXPECT_EQ(config.num_particles, 8);
    EXPECT_EQ(config.seed, 9u);
    EXPECT_EQ(config.total_mass, 2.0f);
    EXPECT_EQ(config.plummer_radius, 3.0f);
    EXPECT_EQ(config.two_body_separation, 4.0f);
    EXPECT_EQ(config.variant, cps::NBodyVariant::Naive);
    EXPECT_EQ(config.softening, 0.5f);
    EXPECT_EQ(config.gravitational_constant, 6.0f);
    EXPECT_EQ(config.nbody.tile_size, 32);
    EXPECT_EQ(config.sph.viscosity, 0.3f);
    EXPECT_EQ(config.external_gravity.z, 3.0f);
    EXPECT_EQ(config.snapshot_interval, 10);
    EXPECT_EQ(config.snapshot_dir, "s");
    EXPECT_EQ(config.diagnostics_interval, 11);
    EXPECT_EQ(config.diagnostics_path, "d");
}

TEST(Config, ASchema2FileWithNoSectionsIsTheDefaults) {
    expect_same_config(cps::Config::from_json_string(R"({"schema": 2})"),
                       cps::Config::from_json_string("{}"));
}

TEST(Config, Schema1IsAcceptedByNameAsAFlatFile) {
    EXPECT_EQ(cps::Config::from_json_string(R"({"schema": 1, "dt": 0.002})").dt, 0.002f);
}

TEST(Config, RejectsAnUnsupportedSchemaListingWhatIsSupported) {
    for (const char* value : {"0", "3", "-1", "99999999999"}) {
        SCOPED_TRACE(value);
        const std::string message = error_of(std::string(R"({"schema": )") + value + "}");
        EXPECT_NE(message.find("unsupported configuration schema " + std::string(value)),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find("1 or 2 are supported"), std::string::npos) << message;
    }
}

TEST(Config, RejectsASchemaThatIsNotAnInteger) {
    for (const char* value : {"2.0", "\"2\"", "true", "null", "[2]"}) {
        SCOPED_TRACE(value);
        const std::string message = error_of(std::string(R"({"schema": )") + value + "}");
        EXPECT_NE(message.find("'schema' must be an integer"), std::string::npos) << message;
    }
}

// These were errors before schema 2 existed. Migration must not make them work
// in a file that does not ask for schema 2.
TEST(Config, ASchema1FileCannotUseKeysThatOnlySchema2Has) {
    for (const char* text :
         {R"({"run": {"dt": 0.1}})", R"({"output": {}})", R"({"schema": 1, "run": {}})",
          R"({"nbody": {"softening": 0.5}})", R"({"variant": "naive", "nbody": {"variant": "fused"}})",
          R"({"sph": {"external_gravity": [0, 0, 0]}})",
          R"({"initial_condition": {"type": "plummer"}})"}) {
        SCOPED_TRACE(text);
        EXPECT_THROW(cps::Config::from_json_string(text), std::runtime_error);
    }
}

TEST(Config, AV1FileWarnsOnceNamingItself) {
    const std::filesystem::path path = fixture_path("dam_break_50k.json");
    ::testing::internal::CaptureStderr();
    cps::Config::from_json_file(path);
    const std::string captured = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(std::count(captured.begin(), captured.end(), '\n'), 1) << captured;
    EXPECT_NE(captured.find(path.string()), std::string::npos) << captured;
    EXPECT_NE(captured.find("--print-config"), std::string::npos) << captured;
}

TEST(Config, AV2FileDoesNotWarn) {
    ::testing::internal::CaptureStderr();
    cps::Config::from_json_file(preset_path("dam_break_50k.json"));
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

}  // namespace
