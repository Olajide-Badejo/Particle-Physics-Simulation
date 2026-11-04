#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/io/vtk.hpp"

namespace fs = std::filesystem;

namespace {

using Bytes = std::vector<unsigned char>;

Bytes read_all(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string text_of(const fs::path& path) {
    const Bytes bytes = read_all(path);
    return std::string(bytes.begin(), bytes.end());
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

// Exactly representable values and a shuffled id, so the fixtures do not depend on
// the platform. Golden fixtures:
//
// tests/data/vtk_4p/ holds the series of this state at steps 0 and 10, written by the
// code under test. The byte test below pins them, and tests/scripts/test_cps_io.py
// reads them from Python. To regenerate them after a deliberate format change, run
//
//   unit_tests_host --gtest_also_run_disabled_tests
//                   --gtest_filter='VtkFixtures.DISABLED_Regenerate'
//
// and say in the commit why the bytes moved.
cps::HostParticleState small_state() {
    cps::HostParticleState state;
    state.x = {-0.5f, 0.5f, -0.5f, 0.5f};
    state.y = {-0.25f, -0.5f, 0.75f, 0.5f};
    state.z = {0.0f, 0.25f, -0.25f, 1.0f};
    state.vx = {0.125f, -0.125f, 0.0f, 2.0f};
    state.vy = {0.5f, 0.0f, 0.375f, -0.375f};
    state.vz = {-0.0f, 0.0625f, 4.0f, -0.0625f};
    state.mass = {0.25f, 0.5f, 1.0f, 2.0f};
    state.id = {2u, 0u, 3u, 1u};
    return state;
}

template <typename T>
T load(const Bytes& bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
std::vector<T> load_array(const Bytes& bytes, std::size_t offset, std::size_t count) {
    std::vector<T> values(count);
    std::memcpy(values.data(), bytes.data() + offset, count * sizeof(T));
    return values;
}

class VtkTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() /
                     ("cps_vtk_test_" + std::to_string(stamp) + "_" + std::to_string(counter_++));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    // A frame at step as the series expects to find it.
    void frame_at(cps::VtkSeries& series, uint64_t step) {
        series.add_frame(small_state(), cps::VtkFrameInfo{step, 0.25 * static_cast<double>(step), true});
    }

    std::vector<uint64_t> indexed_steps() const {
        std::vector<uint64_t> steps;
        for (const cps::VtkSeriesEntry& entry :
             cps::read_vtk_series_index(directory_ / cps::VTK_SERIES_FILE_NAME)) {
            steps.push_back(entry.step);
        }
        return steps;
    }

    fs::path directory_;
    static int counter_;
};

int VtkTest::counter_ = 0;

// ---------------------------------------------------------------------------
// One frame
// ---------------------------------------------------------------------------

TEST_F(VtkTest, TheHeaderDeclaresTheFormat) {
    const fs::path path = directory_ / "f.vtp";
    cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{7u, 0.5, false});
    const std::string text = text_of(path);
    const auto has = [&text](const char* wanted) {
        EXPECT_NE(text.find(wanted), std::string::npos) << wanted;
    };
    has("type=\"PolyData\"");
    has("byte_order=\"LittleEndian\"");
    has("header_type=\"UInt64\"");
    has("NumberOfPoints=\"4\"");
    has("NumberOfVerts=\"4\"");
    has("<AppendedData encoding=\"raw\">");
    has("Name=\"velocity\" NumberOfComponents=\"3\"");
    has("type=\"UInt32\" Name=\"id\"");
    EXPECT_EQ(text.find("density"), std::string::npos);
    EXPECT_EQ(text.find("pressure"), std::string::npos);
}

TEST_F(VtkTest, FieldDataRecordsTimeStepAndVelocityConvention) {
    const fs::path path = directory_ / "f.vtp";
    cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{7u, 0.1, true});
    std::string text = text_of(path);
    EXPECT_NE(text.find("Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">"
                        "0.10000000000000001</DataArray>"),
              std::string::npos);
    EXPECT_NE(text.find("Name=\"Step\" NumberOfTuples=\"1\" format=\"ascii\">7</DataArray>"),
              std::string::npos);
    // "half_step" as bytes, then the terminator.
    EXPECT_NE(text.find("Name=\"VelocityConvention\" NumberOfTuples=\"1\" format=\"ascii\">"
                        "104 97 108 102 95 115 116 101 112 0</Array>"),
              std::string::npos);

    cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{7u, 0.1, false});
    text = text_of(path);
    // "synchronised".
    EXPECT_NE(text.find(">115 121 110 99 104 114 111 110 105 115 101 100 0</Array>"),
              std::string::npos);
}

// Walks the appended block by the offsets in the XML, reading each array and the
// byte count in front of it, against the hand made state.
TEST_F(VtkTest, TheAppendedBlockHoldsEachArrayWhereItsOffsetSays) {
    const fs::path path = directory_ / "f.vtp";
    const cps::HostParticleState state = small_state();
    cps::write_vtk_frame(path, state, cps::VtkFrameInfo{7u, 0.5, true});
    const Bytes bytes = read_all(path);
    const std::string text(bytes.begin(), bytes.end());

    const std::size_t underscore = text.find("<AppendedData encoding=\"raw\">\n    _");
    ASSERT_NE(underscore, std::string::npos);
    const std::size_t base = text.find('_', underscore) + 1;

    // The value of offset="..." after the text naming an array.
    const auto block = [&](const char* name) {
        const std::size_t at = text.find(name);
        EXPECT_NE(at, std::string::npos) << name;
        const std::size_t quote = text.find("offset=\"", at) + 8;
        return base + static_cast<std::size_t>(std::stoull(text.substr(quote)));
    };

    const std::size_t points = block("<Points>");
    const std::size_t velocity = block("Name=\"velocity\"");
    const std::size_t mass = block("Name=\"mass\"");
    const std::size_t id = block("Name=\"id\"");
    const std::size_t connectivity = block("Name=\"connectivity\"");
    const std::size_t offsets = block("Name=\"offsets\"");

    constexpr std::size_t HEADER = sizeof(uint64_t);
    EXPECT_EQ(points, base);
    EXPECT_EQ(load<uint64_t>(bytes, points), 48u);
    EXPECT_EQ(velocity, points + HEADER + 48u);
    EXPECT_EQ(load<uint64_t>(bytes, velocity), 48u);
    EXPECT_EQ(mass, velocity + HEADER + 48u);
    EXPECT_EQ(load<uint64_t>(bytes, mass), 16u);
    EXPECT_EQ(id, mass + HEADER + 16u);
    EXPECT_EQ(load<uint64_t>(bytes, id), 16u);
    EXPECT_EQ(connectivity, id + HEADER + 16u);
    EXPECT_EQ(load<uint64_t>(bytes, connectivity), 32u);
    EXPECT_EQ(offsets, connectivity + HEADER + 32u);
    EXPECT_EQ(load<uint64_t>(bytes, offsets), 32u);

    EXPECT_EQ(load_array<float>(bytes, points + HEADER, 12),
              (std::vector<float>{-0.5f, -0.25f, 0.0f, 0.5f, -0.5f, 0.25f, -0.5f, 0.75f, -0.25f,
                                  0.5f, 0.5f, 1.0f}));
    EXPECT_EQ(load_array<float>(bytes, velocity + HEADER, 12),
              (std::vector<float>{0.125f, 0.5f, -0.0f, -0.125f, 0.0f, 0.0625f, 0.0f, 0.375f, 4.0f,
                                  2.0f, -0.375f, -0.0625f}));
    EXPECT_EQ(load_array<float>(bytes, mass + HEADER, 4), state.mass);
    EXPECT_EQ(load_array<uint32_t>(bytes, id + HEADER, 4), state.id);
    EXPECT_EQ(load_array<int64_t>(bytes, connectivity + HEADER, 4),
              (std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(load_array<int64_t>(bytes, offsets + HEADER, 4), (std::vector<int64_t>{1, 2, 3, 4}));

    // Nothing is left between the last array and the closing tags.
    const std::string tail = "\n  </AppendedData>\n</VTKFile>\n";
    EXPECT_EQ(bytes.size(), offsets + HEADER + 32u + tail.size());
    EXPECT_EQ(text.substr(offsets + HEADER + 32u), tail);
}

TEST_F(VtkTest, AnEmptyStateIsAValidEmptyFrame) {
    const fs::path path = directory_ / "f.vtp";
    cps::HostParticleState state;
    state.resize(0);
    cps::write_vtk_frame(path, state, cps::VtkFrameInfo{});
    EXPECT_NE(text_of(path).find("NumberOfPoints=\"0\""), std::string::npos);
}

TEST_F(VtkTest, RefusesInconsistentArraysNamingTheFile) {
    const fs::path path = directory_ / "f.vtp";
    cps::HostParticleState mass_short = small_state();
    mass_short.mass.pop_back();
    cps::HostParticleState ids_short = small_state();
    ids_short.id.pop_back();
    for (const cps::HostParticleState& state : {mass_short, ids_short}) {
        try {
            cps::write_vtk_frame(path, state, cps::VtkFrameInfo{});
            FAIL() << "expected a refusal";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("f.vtp"), std::string::npos) << error.what();
        }
    }
    EXPECT_FALSE(fs::exists(path));
}

TEST_F(VtkTest, CreatesItsDirectoryAndLeavesNoTemporaryBehind) {
    const fs::path path = directory_ / "a" / "b" / "f.vtp";
    cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{});
    EXPECT_TRUE(fs::exists(path));
    int entries = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(path.parent_path())) {
        ++entries;
        EXPECT_EQ(entry.path().extension().string(), ".vtp");
    }
    EXPECT_EQ(entries, 1);
}

// An existing frame stays whole until the new one replaces it in one rename: a
// directory blocking the temporary's name stops the write before the rename, and the
// old file is still there and still complete.
TEST_F(VtkTest, AFailedWriteLeavesTheOldFrameInPlace) {
    const fs::path path = directory_ / "f.vtp";
    cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{1u, 1.0, false});
    const Bytes before = read_all(path);

    fs::create_directories(path.string() + ".tmp");
    EXPECT_THROW(cps::write_vtk_frame(path, small_state(), cps::VtkFrameInfo{2u, 2.0, false}),
                 std::runtime_error);
    EXPECT_EQ(read_all(path), before);
}

TEST_F(VtkTest, NamesFramesWithAZeroPaddedStep) {
    EXPECT_EQ(cps::vtk_frame_path("d", 42u).filename().string(), "frame_000000042.vtp");
    EXPECT_EQ(cps::vtk_frame_path("d", 12345678901ull).filename().string(), "frame_12345678901.vtp");
}

// ---------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------

TEST_F(VtkTest, TheIndexListsEveryFrameWithItsTime) {
    const fs::path path = directory_ / "series.pvd";
    cps::write_vtk_series_index(path, {{0u, 0.0, "frame_000000000.vtp"},
                                       {10u, 0.1, "frame_000000010.vtp"}});
    EXPECT_EQ(text_of(path),
              "<?xml version=\"1.0\"?>\n"
              "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" "
              "header_type=\"UInt64\">\n"
              "  <Collection>\n"
              "    <DataSet timestep=\"0\" part=\"0\" file=\"frame_000000000.vtp\"/>\n"
              "    <DataSet timestep=\"0.10000000000000001\" part=\"0\" "
              "file=\"frame_000000010.vtp\"/>\n"
              "  </Collection>\n"
              "</VTKFile>\n");
}

TEST_F(VtkTest, TheIndexReadsBackExactly) {
    const fs::path path = directory_ / "series.pvd";
    const double time = 0.1 + 0.2;
    cps::write_vtk_series_index(path, {{10u, time, "frame_000000010.vtp"}});
    const std::vector<cps::VtkSeriesEntry> entries = cps::read_vtk_series_index(path);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].step, 10u);
    EXPECT_EQ(entries[0].time, time);
    EXPECT_EQ(entries[0].file, "frame_000000010.vtp");
}

TEST_F(VtkTest, ReadingAForeignIndexIsRefusedNamingTheFile) {
    const fs::path path = directory_ / "series.pvd";
    const std::string open = "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\">\n<Collection>\n";
    const std::string close = "</Collection>\n</VTKFile>\n";
    for (const std::string& body :
         {std::string("not xml at all\n"),
          open,  // cut off
          open + "<DataSet timestep=\"x\" part=\"0\" file=\"frame_1.vtp\"/>\n" + close,
          open + "<DataSet timestep=\"1\" part=\"0\" file=\"../../etc/passwd\"/>\n" + close,
          open + "<Something/>\n" + close}) {
        write_text(path, body);
        try {
            cps::read_vtk_series_index(path);
            ADD_FAILURE() << "accepted: " << body;
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("series.pvd"), std::string::npos)
                << error.what();
        }
    }
    EXPECT_THROW(cps::read_vtk_series_index(directory_ / "missing.pvd"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// The series
// ---------------------------------------------------------------------------

TEST_F(VtkTest, EveryFrameIsInTheIndexAsSoonAsItIsWritten) {
    cps::VtkSeries series = cps::VtkSeries::fresh(directory_);
    EXPECT_TRUE(indexed_steps().empty());
    frame_at(series, 0);
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{0u}));
    frame_at(series, 5);
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{0u, 5u}));
    EXPECT_TRUE(series.has_frame(5));
    EXPECT_FALSE(series.has_frame(4));
    frame_at(series, 5);
    EXPECT_EQ(series.entries().size(), 2u);
}

// Only what the series owns goes: a file the user put next to it stays.
TEST_F(VtkTest, AFreshSeriesReplacesTheOldOneAndNothingElse) {
    {
        cps::VtkSeries old = cps::VtkSeries::fresh(directory_);
        frame_at(old, 0);
        frame_at(old, 5);
    }
    write_text(directory_ / "notes.txt", "mine");
    write_text(directory_ / "frame_notes.vtp", "mine too");
    write_text(directory_ / "frame_000000099.vtp.tmp", "a killed write");

    cps::VtkSeries series = cps::VtkSeries::fresh(directory_);
    EXPECT_TRUE(indexed_steps().empty());
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(directory_, 0)));
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(directory_, 5)));
    EXPECT_FALSE(fs::exists(directory_ / "frame_000000099.vtp.tmp"));
    EXPECT_TRUE(fs::exists(directory_ / "notes.txt"));
    EXPECT_TRUE(fs::exists(directory_ / "frame_notes.vtp"));
}

TEST_F(VtkTest, AResumeKeepsFramesUpToTheStepAndDropsThoseBeyond) {
    {
        cps::VtkSeries old = cps::VtkSeries::fresh(directory_);
        frame_at(old, 0);
        frame_at(old, 5);
        frame_at(old, 10);
        frame_at(old, 15);
    }
    const Bytes kept = read_all(cps::vtk_frame_path(directory_, 5));

    cps::VtkSeries series = cps::VtkSeries::resumed(directory_, 10);
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{0u, 5u, 10u}));
    EXPECT_TRUE(series.has_frame(10));
    EXPECT_FALSE(series.has_frame(15));
    EXPECT_EQ(read_all(cps::vtk_frame_path(directory_, 5)), kept);
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(directory_, 15)));

    frame_at(series, 20);
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{0u, 5u, 10u, 20u}));
    EXPECT_EQ(cps::read_vtk_series_index(series.index_path())[1].time, 1.25);
}

// A frame renamed into place just before the kill, the index not yet rewritten, is on
// the disk and past the snapshot. The index cannot be what finds it.
TEST_F(VtkTest, AResumeDropsAFrameTheIndexNeverNamed) {
    {
        cps::VtkSeries old = cps::VtkSeries::fresh(directory_);
        frame_at(old, 0);
    }
    cps::write_vtk_frame(cps::vtk_frame_path(directory_, 8), small_state(),
                         cps::VtkFrameInfo{8u, 2.0, true});
    write_text(directory_ / "series.pvd.tmp", "a killed write");
    cps::VtkSeries series = cps::VtkSeries::resumed(directory_, 5);
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(directory_, 8)));
    EXPECT_FALSE(fs::exists(directory_ / "series.pvd.tmp"));
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{0u}));
}

TEST_F(VtkTest, AResumeDropsAnEntryWhoseFrameIsGone) {
    {
        cps::VtkSeries old = cps::VtkSeries::fresh(directory_);
        frame_at(old, 0);
        frame_at(old, 5);
    }
    fs::remove(cps::vtk_frame_path(directory_, 0));
    testing::internal::CaptureStderr();
    cps::VtkSeries series = cps::VtkSeries::resumed(directory_, 5);
    const std::string warning = testing::internal::GetCapturedStderr();
    EXPECT_NE(warning.find("frame_000000000.vtp"), std::string::npos) << warning;
    EXPECT_EQ(indexed_steps(), (std::vector<uint64_t>{5u}));
}

TEST_F(VtkTest, AResumeWithNoSeriesStartsOne) {
    const fs::path empty = directory_ / "new";
    cps::VtkSeries series = cps::VtkSeries::resumed(empty, 100);
    EXPECT_TRUE(series.entries().empty());
    EXPECT_TRUE(fs::exists(empty / cps::VTK_SERIES_FILE_NAME));
}

// The same promise a refused resume makes to the diagnostics file.
TEST_F(VtkTest, AResumeRefusesAForeignIndexAndChangesNothing) {
    {
        cps::VtkSeries old = cps::VtkSeries::fresh(directory_);
        frame_at(old, 9);
    }
    write_text(directory_ / cps::VTK_SERIES_FILE_NAME, "somebody else's file\n");
    EXPECT_THROW(cps::VtkSeries::resumed(directory_, 5), std::runtime_error);
    EXPECT_EQ(text_of(directory_ / cps::VTK_SERIES_FILE_NAME), "somebody else's file\n");
    EXPECT_TRUE(fs::exists(cps::vtk_frame_path(directory_, 9)));
}

// ---------------------------------------------------------------------------
// The golden fixture pins the format
// ---------------------------------------------------------------------------

void write_fixture_series(const fs::path& directory) {
    cps::VtkSeries series = cps::VtkSeries::fresh(directory);
    series.add_frame(small_state(), cps::VtkFrameInfo{0u, 0.0, true});
    series.add_frame(small_state(), cps::VtkFrameInfo{10u, 0.5, true});
}

fs::path fixture_directory() {
    return fs::path(CPS_PROJECT_ROOT) / "tests" / "data" / "vtk_4p";
}

TEST_F(VtkTest, TheWriterReproducesTheFixtureByteForByte) {
    write_fixture_series(directory_);
    for (const char* name : {"series.pvd", "frame_000000000.vtp", "frame_000000010.vtp"}) {
        SCOPED_TRACE(name);
        ASSERT_TRUE(fs::exists(fixture_directory() / name));
        EXPECT_EQ(read_all(directory_ / name), read_all(fixture_directory() / name));
    }
}

TEST(VtkFixtures, DISABLED_Regenerate) { write_fixture_series(fixture_directory()); }

}  // namespace
