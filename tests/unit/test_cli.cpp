#include <gtest/gtest.h>

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cli.hpp"

namespace {

cps::CliOptions parse(std::vector<std::string> args) {
    args.insert(args.begin(), "cuda_physics_sim");
    std::vector<const char*> argv;
    for (const std::string& arg : args) {
        argv.push_back(arg.c_str());
    }
    return cps::parse_cli(static_cast<int>(argv.size()), argv.data());
}

std::string error_of(std::vector<std::string> args) {
    try {
        parse(std::move(args));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

TEST(Cli, NoArgumentsGiveNoOverrides) {
    const cps::CliOptions options = parse({});
    EXPECT_FALSE(options.show_help);
    EXPECT_FALSE(options.resume);
    EXPECT_FALSE(options.device_info);
    EXPECT_FALSE(options.print_config);
    EXPECT_TRUE(options.config_path.empty());
    EXPECT_FALSE(options.solver);
    EXPECT_FALSE(options.variant);
    EXPECT_FALSE(options.steps);
    EXPECT_FALSE(options.particles);
    EXPECT_FALSE(options.dt);
    EXPECT_FALSE(options.seed);
    EXPECT_FALSE(options.snapshot_dir);
    EXPECT_FALSE(options.snapshot_every);
    EXPECT_FALSE(options.diagnostics_path);
    EXPECT_FALSE(options.diag_every);
}

TEST(Cli, ParsesEveryFlag) {
    const cps::CliOptions options =
        parse({"--config", "c.json", "--solver", "sph", "--variant", "fused", "--steps", "0",
               "--particles", "2147483647", "--dt", "0.0005", "--seed", "4294967295",
               "--snapshot-dir", "snaps", "--snapshot-every", "50", "--diagnostics", "",
               "--diag-every", "0", "--resume", "--device-info", "--print-config"});
    EXPECT_EQ(options.config_path, "c.json");
    EXPECT_EQ(options.solver, cps::SolverType::Sph);
    EXPECT_EQ(options.variant, cps::NBodyVariant::Fused);
    EXPECT_EQ(options.steps, 0);
    EXPECT_EQ(options.particles, 2147483647);
    ASSERT_TRUE(options.dt);
    EXPECT_FLOAT_EQ(*options.dt, 0.0005f);
    EXPECT_EQ(options.seed, 4294967295u);
    EXPECT_EQ(options.snapshot_dir, "snaps");
    EXPECT_EQ(options.snapshot_every, 50);
    EXPECT_EQ(options.diagnostics_path, "");
    EXPECT_EQ(options.diag_every, 0);
    EXPECT_TRUE(options.resume);
    EXPECT_TRUE(options.device_info);
    EXPECT_TRUE(options.print_config);
}

TEST(Cli, HelpStopsParsing) {
    EXPECT_TRUE(parse({"--help", "--no-such-flag"}).show_help);
    EXPECT_TRUE(parse({"-h"}).show_help);
}

// Each case is one bad command line and the text its error must contain.
struct BadCliCase {
    const char* label;
    std::vector<std::string> args;
    const char* expected_in_message;
};

void PrintTo(const BadCliCase& value, std::ostream* out) { *out << value.label; }

class CliRejects : public ::testing::TestWithParam<BadCliCase> {};

TEST_P(CliRejects, NamingTheFlag) {
    const std::string message = error_of(GetParam().args);
    ASSERT_FALSE(message.empty()) << "expected an error";
    EXPECT_NE(message.find(GetParam().expected_in_message), std::string::npos)
        << "message was: " << message;
    // The library's own message, such as "stoll", says nothing about the flag.
    EXPECT_EQ(message.find("stoll"), std::string::npos) << message;
    EXPECT_EQ(message.find("stod"), std::string::npos) << message;
}

INSTANTIATE_TEST_SUITE_P(
    Flags, CliRejects,
    ::testing::Values(
        BadCliCase{"dt_zero", {"--config", "c.json", "--dt", "0"}, "--dt"},
        BadCliCase{"dt_negative", {"--dt", "-0.1"}, "--dt"},
        BadCliCase{"dt_nan", {"--dt", "nan"}, "--dt"},
        BadCliCase{"dt_inf", {"--dt", "inf"}, "--dt"},
        BadCliCase{"dt_beyond_float", {"--dt", "1e300"}, "--dt"},
        BadCliCase{"dt_text", {"--dt", "fast"}, "--dt"},
        BadCliCase{"dt_trailing_junk", {"--dt", "0.1s"}, "--dt"},
        BadCliCase{"particles_negative", {"--particles", "-5"}, "--particles"},
        BadCliCase{"particles_zero", {"--particles", "0"}, "--particles"},
        BadCliCase{"particles_above_int", {"--particles", "2147483648"}, "--particles"},
        BadCliCase{"particles_beyond_long_long", {"--particles", "99999999999999999999"},
                   "--particles"},
        BadCliCase{"particles_fractional", {"--particles", "10.5"}, "--particles"},
        BadCliCase{"steps_text", {"--steps", "abc"}, "--steps"},
        BadCliCase{"steps_negative", {"--steps", "-1"}, "--steps"},
        BadCliCase{"steps_empty", {"--steps", ""}, "--steps"},
        BadCliCase{"steps_leading_space", {"--steps", " 5"}, "--steps"},
        BadCliCase{"seed_negative", {"--seed", "-1"}, "--seed"},
        BadCliCase{"seed_above_uint32", {"--seed", "4294967296"}, "--seed"},
        BadCliCase{"snapshot_every_negative", {"--snapshot-every", "-2"}, "--snapshot-every"},
        BadCliCase{"diag_every_text", {"--diag-every", "often"}, "--diag-every"},
        BadCliCase{"solver_unknown", {"--solver", "barnes_hut"}, "--solver"},
        BadCliCase{"variant_unknown", {"--variant", "magic"}, "--variant"},
        BadCliCase{"missing_config", {"--config"}, "--config"},
        BadCliCase{"missing_solver", {"--solver"}, "--solver"},
        BadCliCase{"missing_variant", {"--variant"}, "--variant"},
        BadCliCase{"missing_steps", {"--steps"}, "--steps"},
        BadCliCase{"missing_particles", {"--particles"}, "--particles"},
        BadCliCase{"missing_dt", {"--config", "c.json", "--dt"}, "--dt"},
        BadCliCase{"missing_seed", {"--seed"}, "--seed"},
        BadCliCase{"missing_snapshot_dir", {"--snapshot-dir"}, "--snapshot-dir"},
        BadCliCase{"missing_snapshot_every", {"--snapshot-every"}, "--snapshot-every"},
        BadCliCase{"missing_diagnostics", {"--diagnostics"}, "--diagnostics"},
        BadCliCase{"missing_diag_every", {"--diag-every"}, "--diag-every"},
        BadCliCase{"unknown_option", {"--partciles", "5"}, "--partciles"}),
    [](const ::testing::TestParamInfo<BadCliCase>& info) { return std::string(info.param.label); });

TEST(Cli, OverridesReplaceFileValues) {
    cps::Config config;
    config.snapshot_dir = "from_file";
    config.diagnostics_path = "from_file.csv";
    const cps::CliOptions options =
        parse({"--solver", "nbody", "--variant", "naive", "--steps", "7", "--particles", "64",
               "--dt", "0.25", "--seed", "3", "--snapshot-dir", "cli", "--snapshot-every", "2",
               "--diagnostics", "cli.csv", "--diag-every", "5"});
    cps::apply_cli_overrides(options, config);
    EXPECT_EQ(config.solver, cps::SolverType::NBody);
    EXPECT_EQ(config.variant, cps::NBodyVariant::Naive);
    EXPECT_EQ(config.steps, 7);
    EXPECT_EQ(config.num_particles, 64);
    EXPECT_FLOAT_EQ(config.dt, 0.25f);
    EXPECT_EQ(config.seed, 3u);
    EXPECT_EQ(config.snapshot_dir, "cli");
    EXPECT_EQ(config.snapshot_interval, 2);
    EXPECT_EQ(config.diagnostics_path, "cli.csv");
    EXPECT_EQ(config.diagnostics_interval, 5);
    EXPECT_NO_THROW(config.validate());
}

TEST(Cli, AbsentOverridesLeaveTheConfigAlone) {
    const cps::Config original;
    cps::Config config = original;
    cps::apply_cli_overrides(parse({"--config", "c.json"}), config);
    EXPECT_EQ(config.to_json_string(), original.to_json_string());
}

// These two keep their original meaning: an empty snapshot directory is
// ignored, while an empty diagnostics path turns the file off.
TEST(Cli, EmptyPathOverridesKeepTheirMeaning) {
    cps::Config config;
    config.snapshot_dir = "from_file";
    config.diagnostics_path = "from_file.csv";
    cps::apply_cli_overrides(parse({"--snapshot-dir", "", "--diagnostics", ""}), config);
    EXPECT_EQ(config.snapshot_dir, "from_file");
    EXPECT_EQ(config.diagnostics_path, "");
}

}  // namespace
