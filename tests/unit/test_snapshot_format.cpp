#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "core/snapshot.hpp"

namespace fs = std::filesystem;

namespace {

class SnapshotTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() /
                     ("cps_snapshot_test_" + std::to_string(stamp) + "_" +
                      std::to_string(counter_++));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    static cps::Snapshot make_random_snapshot(std::size_t count, uint32_t step, uint32_t flags) {
        cps::Snapshot snapshot;
        snapshot.step = step;
        snapshot.flags = flags;
        snapshot.state.resize(count);
        std::mt19937 rng(1234u + step);
        auto next = [&rng]() {
            return static_cast<float>((static_cast<double>(rng()) / 4294967296.0) * 200.0 - 100.0);
        };
        for (std::size_t i = 0; i < count; ++i) {
            snapshot.state.x[i] = next();
            snapshot.state.y[i] = next();
            snapshot.state.z[i] = next();
            snapshot.state.vx[i] = next();
            snapshot.state.vy[i] = next();
            snapshot.state.vz[i] = next();
            snapshot.state.mass[i] = std::abs(next()) + 0.5f;
        }
        return snapshot;
    }

    fs::path directory_;
    static int counter_;
};

int SnapshotTest::counter_ = 0;

// Every value has to come back with the same bits it went in with. A snapshot
// that is merely close is useless: the resume path claims bit identity.
TEST_F(SnapshotTest, RoundTripsEveryValueExactly) {
    const cps::Snapshot original = make_random_snapshot(1000, 4200u, 0u);
    const fs::path path = directory_ / "snap.bin";
    cps::write_snapshot(path, original);

    const cps::Snapshot restored = cps::read_snapshot(path);
    ASSERT_EQ(restored.state.size(), original.state.size());
    EXPECT_EQ(restored.step, original.step);
    EXPECT_EQ(restored.flags, original.flags);
    for (std::size_t i = 0; i < original.state.size(); ++i) {
        EXPECT_EQ(restored.state.x[i], original.state.x[i]) << "index " << i;
        EXPECT_EQ(restored.state.y[i], original.state.y[i]) << "index " << i;
        EXPECT_EQ(restored.state.z[i], original.state.z[i]) << "index " << i;
        EXPECT_EQ(restored.state.vx[i], original.state.vx[i]) << "index " << i;
        EXPECT_EQ(restored.state.vy[i], original.state.vy[i]) << "index " << i;
        EXPECT_EQ(restored.state.vz[i], original.state.vz[i]) << "index " << i;
        EXPECT_EQ(restored.state.mass[i], original.state.mass[i]) << "index " << i;
    }
}

TEST_F(SnapshotTest, PreservesTheHalfStepVelocityFlag) {
    const cps::Snapshot original =
        make_random_snapshot(16, 7u, cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY);
    const fs::path path = directory_ / "flagged.bin";
    cps::write_snapshot(path, original);
    const cps::Snapshot restored = cps::read_snapshot(path);
    EXPECT_TRUE((restored.flags & cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u);
}

TEST_F(SnapshotTest, HeaderIsExactlySixteenBytesAndFileLengthFollowsFromIt) {
    const std::size_t count = 64;
    const cps::Snapshot original = make_random_snapshot(count, 1u, 0u);
    const fs::path path = directory_ / "sized.bin";
    cps::write_snapshot(path, original);

    const std::uintmax_t expected =
        sizeof(cps::SnapshotHeader) + count * cps::SNAPSHOT_ARRAY_COUNT * sizeof(float);
    EXPECT_EQ(fs::file_size(path), expected);
}

// The write goes to a sibling temporary and is renamed into place, so a
// successful write must leave nothing else behind.
TEST_F(SnapshotTest, LeavesNoTemporaryFileBehind) {
    const cps::Snapshot original = make_random_snapshot(32, 3u, 0u);
    const fs::path path = directory_ / "atomic.bin";
    cps::write_snapshot(path, original);

    int entries = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory_)) {
        ++entries;
        EXPECT_EQ(entry.path().extension().string(), ".bin");
    }
    EXPECT_EQ(entries, 1);
}

TEST_F(SnapshotTest, OverwritingAnExistingSnapshotSucceeds) {
    const fs::path path = directory_ / "repeat.bin";
    cps::write_snapshot(path, make_random_snapshot(8, 1u, 0u));
    const cps::Snapshot second = make_random_snapshot(8, 2u, 0u);
    ASSERT_NO_THROW(cps::write_snapshot(path, second));
    EXPECT_EQ(cps::read_snapshot(path).step, 2u);
}

TEST_F(SnapshotTest, RejectsAFileThatIsNotASnapshot) {
    const fs::path path = directory_ / "foreign.bin";
    {
        std::ofstream stream(path, std::ios::binary);
        const char rubbish[64] = {};
        stream.write(rubbish, sizeof(rubbish));
    }
    EXPECT_THROW(cps::read_snapshot(path), std::runtime_error);
}

TEST_F(SnapshotTest, RejectsATruncatedSnapshot) {
    const fs::path path = directory_ / "cut.bin";
    cps::write_snapshot(path, make_random_snapshot(100, 5u, 0u));
    const std::uintmax_t full = fs::file_size(path);
    fs::resize_file(path, full / 2);
    EXPECT_THROW(cps::read_snapshot(path), std::runtime_error);
}

TEST_F(SnapshotTest, FindsTheHighestNumberedSnapshot) {
    for (uint32_t step : {0u, 1000u, 250u, 12000u, 999u}) {
        cps::write_snapshot(cps::snapshot_path_for_step(directory_, step),
                            make_random_snapshot(4, step, 0u));
    }
    const fs::path newest = cps::latest_snapshot(directory_);
    ASSERT_FALSE(newest.empty());
    EXPECT_EQ(cps::read_snapshot(newest).step, 12000u);
}

TEST_F(SnapshotTest, ReportsNoSnapshotForAnEmptyDirectory) {
    EXPECT_TRUE(cps::latest_snapshot(directory_).empty());
    EXPECT_TRUE(cps::latest_snapshot(directory_ / "missing").empty());
}

// Zero padded names are what makes the lexical and numeric orders agree, which
// matters when a human looks at the directory rather than the resume code.
TEST_F(SnapshotTest, NamesSnapshotsWithAZeroPaddedStep) {
    EXPECT_EQ(cps::snapshot_path_for_step("dir", 42u).filename().string(), "snap_000000042.bin");
    EXPECT_EQ(cps::snapshot_path_for_step("dir", 123456789u).filename().string(),
              "snap_123456789.bin");
}

TEST_F(SnapshotTest, RejectsInconsistentArrayLengths) {
    cps::Snapshot broken = make_random_snapshot(10, 1u, 0u);
    broken.state.vz.pop_back();
    EXPECT_THROW(cps::write_snapshot(directory_ / "bad.bin", broken), std::runtime_error);
}

}  // namespace
