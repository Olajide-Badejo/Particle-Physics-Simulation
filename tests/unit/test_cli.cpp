#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cps/core/cli.hpp"
#include "cps/core/convert.hpp"
#include "cps/core/presets.hpp"
#include "cps/core/snapshot.hpp"

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
    EXPECT_FALSE(options.force_resume);
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

TEST(Cli, VersionStopsParsing) {
    const cps::CliOptions options = parse({"--version", "--no-such-flag"});
    EXPECT_TRUE(options.show_version);
    EXPECT_FALSE(options.show_help);
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

TEST(Cli, AllowUnstableDtSetsTheConfigFieldAndAbsenceLeavesItAlone) {
    EXPECT_FALSE(parse({"--config", "c.json"}).allow_unstable_dt);
    cps::Config config;
    cps::apply_cli_overrides(parse({"--allow-unstable-dt"}), config);
    EXPECT_TRUE(config.allow_unstable_dt);

    // The file can ask for it too, and leaving the flag off does not clear it.
    cps::apply_cli_overrides(parse({"--config", "c.json"}), config);
    EXPECT_TRUE(config.allow_unstable_dt);
}

// Forcing a resume past a physics mismatch only makes sense when resuming, so
// the flag implies --resume rather than needing both.
TEST(Cli, ForceResumeParsesAndImpliesResume) {
    const cps::CliOptions forced = parse({"--config", "c.json", "--force-resume"});
    EXPECT_TRUE(forced.force_resume);
    EXPECT_TRUE(forced.resume);

    const cps::CliOptions both = parse({"--resume", "--force-resume"});
    EXPECT_TRUE(both.force_resume);
    EXPECT_TRUE(both.resume);

    const cps::CliOptions plain = parse({"--resume"});
    EXPECT_TRUE(plain.resume);
    EXPECT_FALSE(plain.force_resume);
}

TEST(Cli, VtkFlagsReachTheConfiguration) {
    const cps::CliOptions options = parse({"--vtk-every", "25", "--vtk-dir", "frames"});
    EXPECT_EQ(options.vtk_every, 25);
    EXPECT_EQ(options.vtk_dir, "frames");
    cps::Config config;
    cps::apply_cli_overrides(options, config);
    EXPECT_EQ(config.vtk_interval, 25);
    EXPECT_EQ(config.vtk_dir, "frames");

    // Absent flags leave the file's values alone, and 0 turns frames off.
    cps::Config from_file;
    from_file.vtk_interval = 7;
    from_file.vtk_dir = "kept";
    cps::apply_cli_overrides(parse({"--config", "c.json"}), from_file);
    EXPECT_EQ(from_file.vtk_interval, 7);
    EXPECT_EQ(from_file.vtk_dir, "kept");
    cps::apply_cli_overrides(parse({"--vtk-every", "0"}), from_file);
    EXPECT_EQ(from_file.vtk_interval, 0);
}

TEST(Cli, VtkFlagErrorsNameTheFlag) {
    EXPECT_NE(error_of({"--vtk-every", "-1"}).find("--vtk-every"), std::string::npos);
    EXPECT_NE(error_of({"--vtk-every", "often"}).find("--vtk-every"), std::string::npos);
    EXPECT_NE(error_of({"--vtk-every"}).find("--vtk-every"), std::string::npos);
    EXPECT_NE(error_of({"--vtk-dir"}).find("--vtk-dir"), std::string::npos);
    EXPECT_NE(error_of({"--vtk-dir", ""}).find("--vtk-dir"), std::string::npos);
}

TEST(Cli, UsageListsForceResume) {
    EXPECT_NE(cps::cli_usage().find("--force-resume"), std::string::npos);
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

cps::CliOptions command(std::vector<std::string> args) {
    args.insert(args.begin(), "cps");
    std::vector<const char*> argv;
    for (const std::string& arg : args) {
        argv.push_back(arg.c_str());
    }
    return cps::parse_command_line(static_cast<int>(argv.size()), argv.data());
}

std::string command_error(std::vector<std::string> args) {
    try {
        command(std::move(args));
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return "";
}

TEST(CliCommand, NoArgumentsAskForUsage) {
    EXPECT_EQ(command({}).command, cps::CliCommand::Usage);
}

TEST(CliCommand, RunTakesItsTargetAndEveryFlag) {
    const cps::CliOptions options =
        command({"run", "dam_break_50k", "--steps", "7", "--resume", "--diagnostics", ""});
    EXPECT_EQ(options.command, cps::CliCommand::Run);
    EXPECT_EQ(options.run_target, "dam_break_50k");
    EXPECT_EQ(options.steps, 7);
    EXPECT_TRUE(options.resume);
    EXPECT_EQ(options.diagnostics_path, "");
}

TEST(CliCommand, RunTargetIsAPresetOnlyWithoutSeparatorOrJson) {
    EXPECT_TRUE(cps::run_target_is_preset("two_body"));
    EXPECT_FALSE(cps::run_target_is_preset("two_body.json"));
    EXPECT_FALSE(cps::run_target_is_preset("./two_body"));
    EXPECT_FALSE(cps::run_target_is_preset("presets/two_body"));
    EXPECT_FALSE(cps::run_target_is_preset("presets\\two_body"));
}

TEST(CliCommand, EachSubcommandParses) {
    EXPECT_EQ(command({"presets"}).command, cps::CliCommand::Presets);
    EXPECT_EQ(command({"info"}).command, cps::CliCommand::Info);
    const cps::CliOptions plain = command({"convert", "in", "out"});
    EXPECT_EQ(plain.command, cps::CliCommand::Convert);
    EXPECT_EQ(plain.convert_input, "in");
    EXPECT_EQ(plain.convert_output, "out");
    EXPECT_TRUE(plain.config_path.empty());
    EXPECT_EQ(command({"convert", "--config", "c.json", "in", "out"}).config_path, "c.json");
    EXPECT_EQ(command({"convert", "in", "out", "--config", "c.json"}).config_path, "c.json");
}

TEST(CliCommand, HelpIsPerSubcommand) {
    for (const char* name : {"run", "presets", "info", "convert"}) {
        const cps::CliOptions options = command({name, "--help"});
        EXPECT_TRUE(options.show_help) << name;
        EXPECT_NE(cps::cli_usage(options.command).find(std::string("usage: cps ") + name),
                  std::string::npos)
            << name;
    }
    EXPECT_TRUE(command({"run", "two_body", "--help"}).show_help);
    EXPECT_EQ(cps::cli_usage(command({"--help"}).command), cps::cli_usage());
    EXPECT_TRUE(command({"--version"}).show_version);
}

TEST(CliCommand, TheOldFormIsLegacyWithTheSameMeaning) {
    const cps::CliOptions options = command({"--config", "x.json", "--steps", "3"});
    EXPECT_EQ(options.command, cps::CliCommand::Legacy);
    EXPECT_EQ(options.config_path, "x.json");
    EXPECT_EQ(options.steps, 3);
    EXPECT_TRUE(command({"--device-info"}).device_info);
    EXPECT_TRUE(command({"--config", "x.json", "--print-config"}).print_config);
    EXPECT_NE(cps::legacy_form_notice().find("next major version"), std::string::npos);
    EXPECT_NE(cps::legacy_form_notice().find("cps run"), std::string::npos);
    EXPECT_EQ(cps::legacy_form_notice().find('\n'), std::string::npos);
}

// main exits 2 on anything parsing throws.
TEST(CliCommand, ErrorsNameWhatIsWrong) {
    EXPECT_NE(command_error({"frobnicate"}).find("unknown subcommand 'frobnicate'"),
              std::string::npos);
    EXPECT_NE(command_error({"run"}).find("cps run needs"), std::string::npos);
    EXPECT_NE(command_error({"run", "--steps", "5"}).find("cps run needs"), std::string::npos);
    EXPECT_NE(command_error({"run", "x", "--steps", "abc"}).find("--steps"), std::string::npos);
    EXPECT_NE(command_error({"run", "x", "--config", "y.json"}).find("--config"),
              std::string::npos);
    EXPECT_NE(command_error({"run", "x", "--nope"}).find("--nope"), std::string::npos);
    EXPECT_NE(command_error({"presets", "extra"}).find("cps presets"), std::string::npos);
    EXPECT_NE(command_error({"info", "extra"}).find("cps info"), std::string::npos);
    EXPECT_NE(command_error({"convert", "in"}).find("cps convert"), std::string::npos);
    EXPECT_NE(command_error({"convert", "a", "b", "c"}).find("cps convert"), std::string::npos);
    EXPECT_NE(command_error({"convert", "a", "b", "--nope"}).find("--nope"), std::string::npos);
    EXPECT_NE(command_error({"convert", "a", "b", "--config"}).find("--config"),
              std::string::npos);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

TEST(CliPresets, EveryFileInThePresetDirectoryIsEmbeddedByteForByte) {
    std::size_t files = 0;
    const std::filesystem::path directory = std::filesystem::path(CPS_PROJECT_ROOT) / "presets";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().stem().string();
        const cps::EmbeddedPreset* preset = cps::find_preset(name);
        ASSERT_NE(preset, nullptr) << name;
        EXPECT_EQ(preset->json, read_file(entry.path()));
        EXPECT_NE(cps::preset_listing().find(name), std::string::npos);
        ++files;
    }
    EXPECT_EQ(cps::embedded_presets().size(), files);
    EXPECT_EQ(cps::find_preset("nonexistent"), nullptr);
}

class CliConvert : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        directory_ = std::filesystem::temp_directory_path() /
                     (std::string("cps_cli_convert_") + info->name());
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
    }
    void TearDown() override { std::filesystem::remove_all(directory_); }

    std::filesystem::path write(const char* name, const std::string& text) const {
        std::ofstream(directory_ / name, std::ios::binary) << text;
        return directory_ / name;
    }

    std::filesystem::path directory_;
    std::filesystem::path root_{CPS_PROJECT_ROOT};
};

TEST_F(CliConvert, EachV1FixtureGivesItsPresetsBytes) {
    const std::filesystem::path out = directory_ / "out.json";
    for (const char* name : {"dam_break_50k", "plummer_10k", "two_body"}) {
        const std::string file = std::string(name) + ".json";
        cps::convert_file(root_ / "tests/data/configs_v1" / file, out, "");
        EXPECT_EQ(read_file(out), read_file(root_ / "presets" / file)) << name;
    }
}

TEST_F(CliConvert, Cps1NeedsAConfigAndGetsAFingerprintAndIds) {
    const std::filesystem::path v1 = root_ / "tests/data/snapshot_v1_4p.bin";
    const std::filesystem::path out = directory_ / "out.bin";
    EXPECT_THROW(cps::convert_file(v1, out, ""), cps::ConvertInputError);
    EXPECT_FALSE(std::filesystem::exists(out));

    const auto matching = write("four.json", R"({"schema": 1, "num_particles": 4})");
    cps::convert_file(v1, out, matching.string());
    const cps::Snapshot before = cps::read_snapshot(v1);
    const cps::Snapshot after = cps::read_snapshot(out);
    EXPECT_EQ(after.format_version, cps::SNAPSHOT_VERSION);
    EXPECT_NE(after.fingerprint, 0u);
    EXPECT_FALSE(after.physics.empty());
    EXPECT_EQ(after.step, before.step);
    EXPECT_EQ(after.flags, before.flags);
    EXPECT_EQ(after.state.x, before.state.x);
    EXPECT_EQ(after.state.vz, before.state.vz);
    EXPECT_EQ(after.state.id, (std::vector<uint32_t>{0, 1, 2, 3}));

    const auto wrong_count = write("five.json", R"({"schema": 1, "num_particles": 5})");
    EXPECT_THROW(cps::convert_file(v1, out, wrong_count.string()), cps::ConvertInputError);
}

TEST_F(CliConvert, RefusesWhatItCannotConvert) {
    const std::filesystem::path out = directory_ / "out";
    const auto text = write("notes.txt", "not a configuration");
    EXPECT_THROW(cps::convert_file(text, out, ""), cps::ConvertInputError);
    EXPECT_THROW(cps::convert_file(directory_ / "absent", out, ""), cps::ConvertInputError);
    EXPECT_THROW(cps::convert_file(root_ / "tests/data/snapshot_v2_4p.bin", out, ""),
                 cps::ConvertInputError);
    const auto config = write("c.json", "{}");
    EXPECT_THROW(cps::convert_file(config, out, config.string()), cps::ConvertInputError);
}

}  // namespace
