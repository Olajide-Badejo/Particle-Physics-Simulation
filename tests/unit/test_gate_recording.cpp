#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test_support_host.hpp"

namespace fs = std::filesystem;

namespace {

// The gate merge is what keeps an ordinary test run from rewriting the
// committed validation_gates.csv, and what lets a recording run replace only
// the gates it measured. These cases exercise it on a temporary file, never on
// the committed one.
class GateRecordingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() /
                     ("cps_gate_test_" + std::to_string(stamp) + "_" + std::to_string(counter_++));
        fs::create_directories(directory_);
        path_ = directory_ / "validation_gates.csv";
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    void write(const std::string& text) const {
        std::ofstream out(path_, std::ios::binary);
        out << text;
    }

    std::string read() const {
        std::ifstream in(path_, std::ios::binary);
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }

    static cps_test::GateRow row(const std::string& gate, const std::string& rest) {
        return {gate, gate + "," + rest};
    }

    fs::path directory_;
    fs::path path_;
    static inline int counter_ = 0;
};

const std::vector<std::string> REGISTRY = {"alpha", "beta", "gamma"};

TEST_F(GateRecordingTest, APartialUpdateReplacesOnlyTheMeasuredGate) {
    write("gate,measured,limit,comparison,passed\n"
          "alpha,1,2,less_than,1\n"
          "beta,3,4,less_than,1\n"
          "gamma,5,6,greater_than,0\n");
    std::ostringstream warnings;
    ASSERT_TRUE(cps_test::merge_gate_rows(path_, REGISTRY, {row("beta", "7,8,less_than,1")},
                                          warnings));
    EXPECT_EQ(read(),
              "gate,measured,limit,comparison,passed\n"
              "alpha,1,2,less_than,1\n"
              "beta,7,8,less_than,1\n"
              "gamma,5,6,greater_than,0\n");
    EXPECT_TRUE(warnings.str().empty()) << warnings.str();
    EXPECT_FALSE(fs::exists(path_.string() + ".tmp")) << "the temporary file was left behind";
}

TEST_F(GateRecordingTest, ANewGateIsPlacedInRegistryOrder) {
    write("gate,measured,limit,comparison,passed\n"
          "alpha,1,2,less_than,1\n"
          "gamma,5,6,greater_than,1\n");
    std::ostringstream warnings;
    ASSERT_TRUE(cps_test::merge_gate_rows(path_, REGISTRY, {row("beta", "3,4,less_than,1")},
                                          warnings));
    EXPECT_EQ(read(),
              "gate,measured,limit,comparison,passed\n"
              "alpha,1,2,less_than,1\n"
              "beta,3,4,less_than,1\n"
              "gamma,5,6,greater_than,1\n");
    EXPECT_TRUE(warnings.str().empty()) << warnings.str();
}

TEST_F(GateRecordingTest, AnUnknownExistingRowIsKeptAndWarnedAbout) {
    write("gate,measured,limit,comparison,passed\n"
          "retired_gate,9,9,less_than,1\n"
          "alpha,1,2,less_than,1\n");
    std::ostringstream warnings;
    ASSERT_TRUE(cps_test::merge_gate_rows(path_, REGISTRY, {row("gamma", "5,6,greater_than,1")},
                                          warnings));
    EXPECT_EQ(read(),
              "gate,measured,limit,comparison,passed\n"
              "alpha,1,2,less_than,1\n"
              "gamma,5,6,greater_than,1\n"
              "retired_gate,9,9,less_than,1\n");
    EXPECT_NE(warnings.str().find("retired_gate"), std::string::npos) << warnings.str();
}

TEST_F(GateRecordingTest, AMissingFileIsCreatedWithTheHeader) {
    std::ostringstream warnings;
    ASSERT_TRUE(cps_test::merge_gate_rows(path_, REGISTRY, {row("alpha", "1,2,less_than,1")},
                                          warnings));
    EXPECT_EQ(read(),
              "gate,measured,limit,comparison,passed\n"
              "alpha,1,2,less_than,1\n");
}

// The report generator parses these rows, so the number format is part of the
// contract. This is a row as it appears in the committed file.
TEST(GateRecording, RowsAreFormattedExactlyAsTheCommittedFileHasThem) {
    EXPECT_EQ(cps_test::format_gate_row("force_vs_cpu_reference_100_particles",
                                        5.7660898445530794e-07, 1.0e-5, "less_than", true),
              "force_vs_cpu_reference_100_particles,5.7660898445530794e-07,"
              "1.0000000000000001e-05,less_than,1");
    EXPECT_EQ(cps_test::format_gate_row("g", 0.5, 0.001, "greater_than", false),
              "g,0.5,0.001,greater_than,0");
}

TEST(GateRecording, TheRegistryListsEachGateOnce) {
    const std::vector<std::string> registry =
        cps_test::read_gate_registry(cps_test::gate_registry_path());
    ASSERT_FALSE(registry.empty()) << cps_test::gate_registry_path().string();
    const std::set<std::string> unique(registry.begin(), registry.end());
    EXPECT_EQ(unique.size(), registry.size());
    for (const std::string& name : registry) {
        EXPECT_EQ(name.find(','), std::string::npos) << name;
    }
}

}  // namespace
