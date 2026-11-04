#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "cps/core/config.hpp"
#include "cps/core/initial_conditions.hpp"
#include "cps/core/physics_fingerprint.hpp"
#include "cps/core/snapshot.hpp"

namespace fs = std::filesystem;

namespace {

using Bytes = std::vector<unsigned char>;

Bytes read_all(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_all(const fs::path& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
T load(const Bytes& bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void store(Bytes& bytes, std::size_t offset, const T& value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

std::size_t entry_offset(std::size_t index) {
    return sizeof(cps::SnapshotHeader) + index * sizeof(cps::SnapshotArrayEntry);
}

// Index of the named entry in a file's array table.
std::size_t find_entry(const Bytes& bytes, const char* name) {
    const auto header = load<cps::SnapshotHeader>(bytes, 0);
    for (std::size_t i = 0; i < header.array_count; ++i) {
        const auto entry = load<cps::SnapshotArrayEntry>(bytes, entry_offset(i));
        if (std::strncmp(entry.name, name, sizeof(entry.name)) == 0) {
            return i;
        }
    }
    ADD_FAILURE() << "no array named " << name;
    return 0;
}

void rename_entry(Bytes& bytes, std::size_t index, const char* name) {
    auto entry = load<cps::SnapshotArrayEntry>(bytes, entry_offset(index));
    std::memset(entry.name, 0, sizeof(entry.name));
    std::memcpy(entry.name, name, std::strlen(name));
    store(bytes, entry_offset(index), entry);
}

// Writes the version 1 layout by hand: the sixteen byte header and the seven
// float arrays. The writer no longer produces this format, so the tests keep
// their own copy of it.
Bytes cps1_bytes(const cps::HostParticleState& state, uint32_t step, uint32_t flags) {
    cps::SnapshotHeaderV1 header;
    header.count = static_cast<uint32_t>(state.size());
    header.step = step;
    header.flags = flags;
    Bytes bytes(sizeof(header));
    std::memcpy(bytes.data(), &header, sizeof(header));
    for (const std::vector<float>* block :
         {&state.x, &state.y, &state.z, &state.vx, &state.vy, &state.vz, &state.mass}) {
        const auto* begin = reinterpret_cast<const unsigned char*>(block->data());
        bytes.insert(bytes.end(), begin, begin + block->size() * sizeof(float));
    }
    return bytes;
}

// Golden fixtures.
//
// tests/data/snapshot_v1_4p.bin and tests/data/snapshot_v2_4p_id.bin hold
// fixture_state() at step 42. The CPS1 file is cps1_bytes(fixture_state(),
// 42, 0); the CPS2 file is write_snapshot(fixture_snapshot()), whose physics
// record comes from nbody_reference(), and whose ids are a shuffled permutation.
// Every value is exactly representable, one velocity is negative zero, and
// nothing depends on a random generator or libm, so both files are the same on
// every platform. To regenerate them after a deliberate format change, run
//
//   unit_tests_host --gtest_also_run_disabled_tests
//                   --gtest_filter='SnapshotFixtures.DISABLED_Regenerate'
//
// and say in the commit why the bytes moved.
//
// tests/data/snapshot_v2_4p.bin is the same snapshot from before CPS2 had an id
// array. No writer produces that layout now, so it is frozen and nothing
// regenerates it: it stands for every such file in the world.
fs::path fixture_path(const char* name) {
    return fs::path(CPS_PROJECT_ROOT) / "tests" / "data" / name;
}

cps::HostParticleState fixture_state() {
    cps::HostParticleState state;
    state.x = {-0.5f, 0.5f, -0.5f, 0.5f};
    state.y = {-0.5f, -0.5f, 0.5f, 0.5f};
    state.z = {0.0f, 0.25f, -0.25f, 0.0f};
    state.vx = {0.125f, -0.125f, 0.0f, 0.0f};
    state.vy = {0.0f, 0.0f, 0.375f, -0.375f};
    state.vz = {-0.0f, 0.0625f, 0.0f, -0.0625f};
    state.mass = {0.25f, 0.25f, 0.25f, 0.25f};
    state.id = {0u, 1u, 2u, 3u};
    return state;
}

// Reference configurations for the fingerprint. Every field in the list is
// set by hand so the golden values do not move when a default does. The
// fluid reference fixes the smoothing radius and particle mass, so deriving
// them needs no libm call.
cps::Config nbody_reference() {
    cps::Config config;
    config.name = "fingerprint_reference";
    config.solver = cps::SolverType::NBody;
    config.variant = cps::NBodyVariant::Fused;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = 4;
    config.dt = 1.0e-3f;
    config.softening = 0.05f;
    config.gravitational_constant = 1.0f;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    return config;
}

cps::Config sph_reference() {
    cps::Config config = nbody_reference();
    config.solver = cps::SolverType::Sph;
    config.variant = cps::NBodyVariant::Tiled;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = 4096;
    config.dt = 1.0e-4f;
    config.sph.smoothing_radius = 0.03125f;
    config.sph.particle_mass = 0.0625f;
    config.sph.rest_density = 1000.0f;
    config.sph.gas_constant = 1000.0f;
    config.sph.viscosity = 0.2f;
    config.sph.boundary_damping = 0.4f;
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    return config;
}

// The same fluid with both resolution parameters derived from the block.
cps::Config sph_derived_reference() {
    cps::Config config = sph_reference();
    config.sph.smoothing_radius = 0.0f;
    config.sph.particle_mass = 0.0f;
    return config;
}

cps::Snapshot fixture_snapshot() {
    const cps::Config config = nbody_reference();
    const std::vector<cps::PhysicsField> fields = cps::physics_fields(config);
    cps::Snapshot snapshot;
    snapshot.step = 42;
    snapshot.time = 42.0 * static_cast<double>(config.dt);
    snapshot.flags = cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY;
    snapshot.state = fixture_state();
    snapshot.state.id = {2u, 0u, 3u, 1u};
    snapshot.fingerprint = cps::physics_fingerprint(fields);
    snapshot.fingerprint_version = cps::PHYSICS_FINGERPRINT_VERSION;
    snapshot.physics = cps::physics_json(fields);
    return snapshot;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

void expect_same_state(const cps::HostParticleState& a, const cps::HostParticleState& b) {
    EXPECT_TRUE(same_bits(a.x, b.x));
    EXPECT_TRUE(same_bits(a.y, b.y));
    EXPECT_TRUE(same_bits(a.z, b.z));
    EXPECT_TRUE(same_bits(a.vx, b.vx));
    EXPECT_TRUE(same_bits(a.vy, b.vy));
    EXPECT_TRUE(same_bits(a.vz, b.vz));
    EXPECT_TRUE(same_bits(a.mass, b.mass));
    EXPECT_EQ(a.id, b.id);
}

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

    // A valid four particle CPS2 file as bytes, ready to be damaged.
    Bytes valid_bytes() {
        const fs::path path = directory_ / "valid.bin";
        cps::write_snapshot(path, fixture_snapshot());
        return read_all(path);
    }

    // Writes bytes to a file and returns the reader's error, or "" if it read.
    std::string read_error(const Bytes& bytes, const char* name = "damaged.bin") {
        const fs::path path = directory_ / name;
        write_all(path, bytes);
        try {
            cps::read_snapshot(path);
        } catch (const std::runtime_error& error) {
            return error.what();
        }
        return "";
    }

    fs::path directory_;
    static int counter_;
};

int SnapshotTest::counter_ = 0;

// Every value has to come back with the same bits it went in with. A snapshot
// that is merely close is useless: the resume path claims bit identity.
TEST_F(SnapshotTest, RoundTripsEveryValueExactly) {
    cps::Snapshot original = make_random_snapshot(1000, 4200u, 0u);
    original.time = 4.2;
    original.fingerprint = 0x0123456789abcdefull;
    original.fingerprint_version = 7;
    original.physics = "{\"anything\":\"at all\"}";
    const fs::path path = directory_ / "snap.bin";
    cps::write_snapshot(path, original);

    const cps::Snapshot restored = cps::read_snapshot(path);
    ASSERT_EQ(restored.state.size(), original.state.size());
    EXPECT_EQ(restored.format_version, 2u);
    EXPECT_EQ(restored.step, original.step);
    EXPECT_EQ(restored.time, original.time);
    EXPECT_EQ(restored.flags, original.flags);
    EXPECT_EQ(restored.fingerprint, original.fingerprint);
    EXPECT_EQ(restored.fingerprint_version, original.fingerprint_version);
    EXPECT_EQ(restored.physics, original.physics);
    expect_same_state(restored.state, original.state);
}

// The step is 64 bits end to end. Version 1 stored 32, so a run past four
// billion steps wrapped silently.
TEST_F(SnapshotTest, KeepsAStepBeyondThirtyTwoBits) {
    cps::Snapshot original = make_random_snapshot(4, 1u, 0u);
    original.step = 0x123456789ull;
    const fs::path path = directory_ / "far.bin";
    cps::write_snapshot(path, original);
    EXPECT_EQ(cps::read_snapshot(path).step, 0x123456789ull);
}

TEST_F(SnapshotTest, PreservesTheHalfStepVelocityFlag) {
    const cps::Snapshot original =
        make_random_snapshot(16, 7u, cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY);
    const fs::path path = directory_ / "flagged.bin";
    cps::write_snapshot(path, original);
    const cps::Snapshot restored = cps::read_snapshot(path);
    EXPECT_TRUE((restored.flags & cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u);
}

TEST_F(SnapshotTest, FileLengthFollowsFromTheLayout) {
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{64}, std::size_t{101}}) {
        cps::Snapshot original = make_random_snapshot(count, 1u, 0u);
        original.physics = "12345";
        const fs::path path = directory_ / "sized.bin";
        cps::write_snapshot(path, original);

        auto padded = [](std::uintmax_t bytes) { return (bytes + 7u) / 8u * 8u; };
        const std::uintmax_t expected =
            sizeof(cps::SnapshotHeader) + 9 * sizeof(cps::SnapshotArrayEntry) +
            cps::SNAPSHOT_ARRAY_COUNT * padded(count * sizeof(float)) +
            padded(count * sizeof(uint32_t)) + padded(5);
        EXPECT_EQ(fs::file_size(path), expected) << count << " particles";
    }
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
    const std::string message = read_error(Bytes(64, 0));
    EXPECT_NE(message.find("not a snapshot"), std::string::npos) << message;
    EXPECT_NE(message.find("damaged.bin"), std::string::npos) << message;
}

TEST_F(SnapshotTest, RejectsAnUnknownFormatVersion) {
    Bytes bytes = valid_bytes();
    store<uint32_t>(bytes, offsetof(cps::SnapshotHeader, version), 3u);
    const std::string message = read_error(bytes);
    EXPECT_NE(message.find("format version 3"), std::string::npos) << message;
}

// A header that claims a vast particle count must fail on the size checks,
// before anything is allocated from the count. Without them this asks for
// tens of terabytes and dies of bad_alloc, or worse, gets it.
TEST_F(SnapshotTest, RejectsAHugeCountInATinyFileQuickly) {
    Bytes bytes = valid_bytes();
    const uint64_t huge = uint64_t{1} << 40;
    store<uint64_t>(bytes, offsetof(cps::SnapshotHeader, count), huge);
    for (const char* name : {"x", "y", "z", "vx", "vy", "vz", "mass"}) {
        const std::size_t at = entry_offset(find_entry(bytes, name));
        store<uint64_t>(bytes, at + offsetof(cps::SnapshotArrayEntry, element_count), huge);
    }

    const auto start = std::chrono::steady_clock::now();
    const std::string message = read_error(bytes);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    EXPECT_NE(message.find("lies outside the payload"), std::string::npos) << message;
    EXPECT_LT(ms, 1000.0);

    // The same count with no table at all fails on the missing arrays.
    Bytes header_only(bytes.begin(), bytes.begin() + sizeof(cps::SnapshotHeader));
    store<uint32_t>(header_only, offsetof(cps::SnapshotHeader, array_count), 0u);
    EXPECT_NE(read_error(header_only).find("missing the required array"), std::string::npos);
}

TEST_F(SnapshotTest, RejectsATruncatedSnapshot) {
    const fs::path path = directory_ / "cut.bin";
    cps::write_snapshot(path, make_random_snapshot(100, 5u, 0u));
    const Bytes full = read_all(path);
    for (std::size_t length : {full.size() / 2, std::size_t{200}, std::size_t{40}, std::size_t{2}}) {
        const Bytes cut(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(length));
        const std::string message = read_error(cut);
        EXPECT_FALSE(message.empty()) << "a file cut to " << length << " bytes was accepted";
        EXPECT_NE(message.find("damaged.bin"), std::string::npos) << message;
    }
}

TEST_F(SnapshotTest, RejectsAnArrayThatRunsPastTheEndOfTheFile) {
    const Bytes valid = valid_bytes();
    const std::size_t x = entry_offset(find_entry(valid, "x"));

    Bytes beyond = valid;
    store<uint64_t>(beyond, x + offsetof(cps::SnapshotArrayEntry, offset), valid.size() + 8);
    EXPECT_NE(read_error(beyond).find("'x'"), std::string::npos);

    // The start is inside the file but the end is not.
    Bytes straddling = valid;
    store<uint64_t>(straddling, x + offsetof(cps::SnapshotArrayEntry, offset), valid.size() - 4);
    EXPECT_NE(read_error(straddling).find("lies outside the payload"), std::string::npos);

    // An offset pointing back into the header is just as wrong.
    Bytes inside_header = valid;
    store<uint64_t>(inside_header, x + offsetof(cps::SnapshotArrayEntry, offset), 8);
    EXPECT_NE(read_error(inside_header).find("lies outside the payload"), std::string::npos);
}

TEST_F(SnapshotTest, RejectsAMissingRequiredArray) {
    Bytes bytes = valid_bytes();
    rename_entry(bytes, find_entry(bytes, "mass"), "mass_old");
    const std::string message = read_error(bytes);
    EXPECT_NE(message.find("missing the required array 'mass'"), std::string::npos) << message;

    Bytes no_physics = valid_bytes();
    rename_entry(no_physics, find_entry(no_physics, "physics"), "notes");
    EXPECT_NE(read_error(no_physics).find("'physics'"), std::string::npos);
}

TEST_F(SnapshotTest, RejectsAWrongDtypeOrElementCount) {
    Bytes dtype = valid_bytes();
    const std::size_t vy = entry_offset(find_entry(dtype, "vy"));
    store<uint32_t>(dtype, vy + offsetof(cps::SnapshotArrayEntry, dtype),
                    static_cast<uint32_t>(cps::SnapshotDtype::F64));
    EXPECT_NE(read_error(dtype).find("'vy' must be f32"), std::string::npos);

    Bytes elements = valid_bytes();
    const std::size_t z = entry_offset(find_entry(elements, "z"));
    store<uint64_t>(elements, z + offsetof(cps::SnapshotArrayEntry, element_count), 3);
    EXPECT_NE(read_error(elements).find("'z' must be f32 with 4 elements"), std::string::npos);

    Bytes unknown = valid_bytes();
    store<uint32_t>(unknown, vy + offsetof(cps::SnapshotArrayEntry, dtype), 99u);
    EXPECT_NE(read_error(unknown).find("unknown dtype code 99"), std::string::npos);
}

TEST_F(SnapshotTest, RejectsADuplicateArrayName) {
    Bytes bytes = valid_bytes();
    rename_entry(bytes, find_entry(bytes, "vx"), "x");
    EXPECT_NE(read_error(bytes).find("'x' more than once"), std::string::npos);
}

// A newer writer adds arrays this reader has never heard of, such as tracer.
// They must be skipped, not rejected, or every addition would be a format
// break.
TEST_F(SnapshotTest, SkipsAnUnknownExtraArray) {
    const Bytes valid = valid_bytes();
    const auto header = load<cps::SnapshotHeader>(valid, 0);
    const std::size_t old_table_end = entry_offset(header.array_count);
    const std::size_t shift = sizeof(cps::SnapshotArrayEntry);

    // Rebuild the file with one more table entry, pointing at four u32 values
    // appended after the existing payload.
    Bytes extended(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(old_table_end));
    for (std::size_t i = 0; i < header.array_count; ++i) {
        auto entry = load<cps::SnapshotArrayEntry>(extended, entry_offset(i));
        entry.offset += shift;
        store(extended, entry_offset(i), entry);
    }
    cps::SnapshotArrayEntry extra;
    std::memcpy(extra.name, "tracer", 6);
    extra.dtype = static_cast<uint32_t>(cps::SnapshotDtype::U32);
    extra.element_count = 4;
    extra.offset = valid.size() + shift;
    extended.resize(extended.size() + shift);
    store(extended, old_table_end, extra);
    extended.insert(extended.end(), valid.begin() + static_cast<std::ptrdiff_t>(old_table_end),
                    valid.end());
    const uint32_t values[4] = {3, 1, 0, 2};
    const auto* value_bytes = reinterpret_cast<const unsigned char*>(values);
    extended.insert(extended.end(), value_bytes, value_bytes + sizeof(values));

    auto new_header = header;
    new_header.array_count = header.array_count + 1;
    const std::size_t payload_start = old_table_end + shift;
    new_header.payload_checksum =
        cps::fnv1a_64(extended.data() + payload_start, extended.size() - payload_start);
    store(extended, 0, new_header);

    const fs::path path = directory_ / "extended.bin";
    write_all(path, extended);
    cps::Snapshot restored;
    ASSERT_NO_THROW(restored = cps::read_snapshot(path));
    expect_same_state(restored.state, fixture_snapshot().state);
    EXPECT_EQ(restored.physics, fixture_snapshot().physics);
}

TEST_F(SnapshotTest, RejectsUnknownFlagBits) {
    Bytes bytes = valid_bytes();
    store<uint32_t>(bytes, offsetof(cps::SnapshotHeader, flags),
                    cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY | (1u << 1));
    EXPECT_NE(read_error(bytes).find("unknown flag bits"), std::string::npos);

    cps::Snapshot flagged = make_random_snapshot(2, 1u, 1u << 5);
    EXPECT_THROW(cps::write_snapshot(directory_ / "flagged.bin", flagged), std::runtime_error);
}

TEST_F(SnapshotTest, RejectsAFlippedPayloadByte) {
    const Bytes valid = valid_bytes();
    // One byte inside the x array, and one inside the physics record.
    for (const char* name : {"x", "physics"}) {
        Bytes bytes = valid;
        const auto entry = load<cps::SnapshotArrayEntry>(bytes, entry_offset(find_entry(bytes, name)));
        bytes[static_cast<std::size_t>(entry.offset) + 1] ^= 0x01u;
        const std::string message = read_error(bytes);
        EXPECT_NE(message.find("checksum"), std::string::npos) << name << ": " << message;
    }
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

// Past 999999999 steps the name grows a digit, so the name order and the step
// order disagree: "snap_1000000000" sorts before "snap_999999999". The newest
// snapshot is the one with the highest step, whatever its width.
TEST_F(SnapshotTest, OrdersSnapshotsByStepNumberAcrossNameWidths) {
    for (uint64_t step : {uint64_t{999999999}, uint64_t{1000000000}, uint64_t{20}}) {
        cps::Snapshot snapshot = make_random_snapshot(2, 1u, 0u);
        snapshot.step = step;
        cps::write_snapshot(cps::snapshot_path_for_step(directory_, step), snapshot);
    }
    // Junk that matches the pattern but cannot be a step is ignored.
    write_all(directory_ / "snap_99999999999999999999999.bin", Bytes(8, 0));
    const fs::path newest = cps::latest_snapshot(directory_);
    EXPECT_EQ(newest.filename().string(), "snap_1000000000.bin");
    EXPECT_EQ(cps::read_snapshot(newest).step, 1000000000u);
}

TEST_F(SnapshotTest, ReportsNoSnapshotForAnEmptyDirectory) {
    EXPECT_TRUE(cps::latest_snapshot(directory_).empty());
    EXPECT_TRUE(cps::latest_snapshot(directory_ / "missing").empty());
}

// Zero padded names keep the lexical and numeric orders in step up to nine
// digits, which matters when a human looks at the directory. Wider steps get
// wider names rather than being cut.
TEST_F(SnapshotTest, NamesSnapshotsWithAZeroPaddedStep) {
    EXPECT_EQ(cps::snapshot_path_for_step("dir", 42u).filename().string(), "snap_000000042.bin");
    EXPECT_EQ(cps::snapshot_path_for_step("dir", 123456789u).filename().string(),
              "snap_123456789.bin");
    EXPECT_EQ(cps::snapshot_path_for_step("dir", 12345678901ull).filename().string(),
              "snap_12345678901.bin");
}

TEST_F(SnapshotTest, RejectsInconsistentArrayLengths) {
    cps::Snapshot broken = make_random_snapshot(10, 1u, 0u);
    broken.state.vz.pop_back();
    EXPECT_THROW(cps::write_snapshot(directory_ / "bad.bin", broken), std::runtime_error);

    cps::Snapshot too_few_ids = make_random_snapshot(10, 1u, 0u);
    too_few_ids.state.id.pop_back();
    EXPECT_THROW(cps::write_snapshot(directory_ / "bad.bin", too_few_ids), std::runtime_error);
}

// ---------------------------------------------------------------------------
// The id array
// ---------------------------------------------------------------------------

TEST(HostParticleState, ResizeNumbersTheParticlesInOrder) {
    cps::HostParticleState state;
    state.resize(4);
    EXPECT_EQ(state.id, (std::vector<uint32_t>{0u, 1u, 2u, 3u}));
}

TEST_F(SnapshotTest, TheWriterEmitsIdAsU32WithOneElementPerParticle) {
    const Bytes bytes = valid_bytes();
    const auto entry = load<cps::SnapshotArrayEntry>(bytes, entry_offset(find_entry(bytes, "id")));
    EXPECT_EQ(entry.dtype, static_cast<uint32_t>(cps::SnapshotDtype::U32));
    EXPECT_EQ(entry.element_count, 4u);
    EXPECT_EQ(load<uint32_t>(bytes, static_cast<std::size_t>(entry.offset)), 2u);
    EXPECT_EQ(load<uint32_t>(bytes, static_cast<std::size_t>(entry.offset) + 12), 1u);
}

// With the entry renamed, this is what a file from before the id array reads as.
TEST_F(SnapshotTest, AFileWithNoIdArrayGetsIdsInFileOrder) {
    Bytes bytes = valid_bytes();
    rename_entry(bytes, find_entry(bytes, "id"), "not_id");
    const fs::path path = directory_ / "no_id.bin";
    write_all(path, bytes);
    EXPECT_EQ(cps::read_snapshot(path).state.id, (std::vector<uint32_t>{0u, 1u, 2u, 3u}));
}

TEST_F(SnapshotTest, RejectsAnIdArrayOfTheWrongDtypeOrLength) {
    Bytes dtype = valid_bytes();
    const std::size_t at = entry_offset(find_entry(dtype, "id"));
    store<uint32_t>(dtype, at + offsetof(cps::SnapshotArrayEntry, dtype),
                    static_cast<uint32_t>(cps::SnapshotDtype::I32));
    EXPECT_NE(read_error(dtype).find("'id' must be u32 with 4 elements"), std::string::npos);

    Bytes length = valid_bytes();
    store<uint64_t>(length, at + offsetof(cps::SnapshotArrayEntry, element_count), 3);
    EXPECT_NE(read_error(length).find("'id' must be u32 with 4 elements"), std::string::npos);
}

// Rewrites the ids and the checksum over them, the way a writer with a bug would
// produce a file that is intact but wrong.
class BadIdArray : public SnapshotTest {
   protected:
    std::string error_with_ids(const std::vector<uint32_t>& ids) {
        Bytes bytes = valid_bytes();
        const auto header = load<cps::SnapshotHeader>(bytes, 0);
        const auto entry =
            load<cps::SnapshotArrayEntry>(bytes, entry_offset(find_entry(bytes, "id")));
        std::memcpy(bytes.data() + entry.offset, ids.data(), ids.size() * sizeof(uint32_t));
        const std::size_t payload = entry_offset(header.array_count);
        auto updated = header;
        updated.payload_checksum = cps::fnv1a_64(bytes.data() + payload, bytes.size() - payload);
        store(bytes, 0, updated);
        return read_error(bytes);
    }
};

TEST_F(BadIdArray, RejectsAValueNotBelowTheCount) {
    const std::string message = error_with_ids({0u, 1u, 2u, 4u});
    EXPECT_NE(message.find("'id' holds 4 at index 3"), std::string::npos) << message;
    EXPECT_NE(message.find("particle count 4"), std::string::npos) << message;
    EXPECT_NE(message.find("damaged.bin"), std::string::npos) << message;
}

TEST_F(BadIdArray, RejectsADuplicate) {
    const std::string message = error_with_ids({1u, 0u, 1u, 3u});
    EXPECT_NE(message.find("'id' holds 1 more than once"), std::string::npos) << message;
    EXPECT_NE(message.find("index 2"), std::string::npos) << message;
}

TEST_F(BadIdArray, AcceptsAnyPermutation) { EXPECT_EQ(error_with_ids({3u, 2u, 1u, 0u}), ""); }

TEST_F(SnapshotTest, TheChecksumCoversTheIdArray) {
    Bytes bytes = valid_bytes();
    const auto entry = load<cps::SnapshotArrayEntry>(bytes, entry_offset(find_entry(bytes, "id")));
    bytes[static_cast<std::size_t>(entry.offset)] ^= 0x01u;
    EXPECT_NE(read_error(bytes).find("checksum"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Version 1 compatibility
// ---------------------------------------------------------------------------

TEST_F(SnapshotTest, ReadsTheCps1GoldenFixture) {
    const fs::path path = fixture_path("snapshot_v1_4p.bin");
    ASSERT_TRUE(fs::exists(path)) << path;
    EXPECT_EQ(read_all(path), cps1_bytes(fixture_state(), 42u, 0u));

    const cps::Snapshot snapshot = cps::read_snapshot(path);
    EXPECT_EQ(snapshot.format_version, 1u);
    EXPECT_EQ(snapshot.step, 42u);
    EXPECT_EQ(snapshot.flags, 0u);
    EXPECT_TRUE(std::isnan(snapshot.time));
    EXPECT_TRUE(snapshot.physics.empty());
    expect_same_state(snapshot.state, fixture_state());
}

TEST_F(SnapshotTest, RejectsACps1FileWhoseSizeDisagreesWithItsCount) {
    const Bytes valid = cps1_bytes(fixture_state(), 42u, 0u);

    Bytes short_file(valid.begin(), valid.end() - 4);
    EXPECT_NE(read_error(short_file).find("size mismatch"), std::string::npos);

    Bytes long_file = valid;
    long_file.push_back(0);
    EXPECT_NE(read_error(long_file).find("size mismatch"), std::string::npos);

    // Sixteen bytes claiming four billion particles would otherwise ask for
    // seven arrays of sixteen gigabytes each before reading a single one.
    Bytes huge(valid.begin(), valid.begin() + sizeof(cps::SnapshotHeaderV1));
    store<uint32_t>(huge, offsetof(cps::SnapshotHeaderV1, count), 0xFFFFFFFFu);
    const auto start = std::chrono::steady_clock::now();
    const std::string message = read_error(huge);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    EXPECT_NE(message.find("size mismatch"), std::string::npos) << message;
    EXPECT_LT(ms, 1000.0);
}

TEST_F(SnapshotTest, RejectsUnknownFlagBitsInACps1File) {
    EXPECT_NE(read_error(cps1_bytes(fixture_state(), 1u, 1u << 1)).find("unknown flag bits"),
              std::string::npos);
}

// ---------------------------------------------------------------------------
// The version 2 golden fixture pins the format
// ---------------------------------------------------------------------------

// Any change to the header, the table, the padding, the checksum, the
// fingerprint encoding or the physics record shows up here as a byte
// difference. If the change is deliberate, regenerate the fixtures (see the
// comment on fixture_path) and say so.
TEST_F(SnapshotTest, WriterReproducesTheCps2FixtureByteForByte) {
    const fs::path golden = fixture_path("snapshot_v2_4p_id.bin");
    ASSERT_TRUE(fs::exists(golden)) << golden;
    const fs::path path = directory_ / "fresh.bin";
    cps::write_snapshot(path, fixture_snapshot());
    const Bytes expected = read_all(golden);
    const Bytes actual = read_all(path);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i], expected[i]) << "first difference at byte " << i;
    }
}

TEST_F(SnapshotTest, ReadsTheCps2FixtureWithIds) {
    const cps::Snapshot snapshot = cps::read_snapshot(fixture_path("snapshot_v2_4p_id.bin"));
    const cps::Snapshot expected = fixture_snapshot();
    EXPECT_EQ(snapshot.step, 42u);
    EXPECT_EQ(snapshot.physics, expected.physics);
    expect_same_state(snapshot.state, expected.state);
}

// A file from before the id array. fixture_state() numbers its particles in file
// order, which is what this file has to come back with.
TEST_F(SnapshotTest, ReadsTheCps2GoldenFixture) {
    const cps::Snapshot snapshot = cps::read_snapshot(fixture_path("snapshot_v2_4p.bin"));
    const cps::Snapshot expected = fixture_snapshot();
    EXPECT_EQ(snapshot.format_version, 2u);
    EXPECT_EQ(snapshot.step, 42u);
    EXPECT_EQ(std::bit_cast<uint64_t>(snapshot.time), std::bit_cast<uint64_t>(expected.time));
    EXPECT_EQ(snapshot.flags, cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY);
    EXPECT_EQ(snapshot.fingerprint, cps::physics_fingerprint(nbody_reference()));
    EXPECT_EQ(snapshot.fingerprint_version, cps::PHYSICS_FINGERPRINT_VERSION);
    EXPECT_EQ(snapshot.physics, expected.physics);
    expect_same_state(snapshot.state, fixture_state());
}

TEST(SnapshotFixtures, DISABLED_Regenerate) {
    const fs::path directory = fixture_path("snapshot_v1_4p.bin").parent_path();
    fs::create_directories(directory);
    write_all(directory / "snapshot_v1_4p.bin", cps1_bytes(fixture_state(), 42u, 0u));
    cps::write_snapshot(directory / "snapshot_v2_4p_id.bin", fixture_snapshot());
}

// ---------------------------------------------------------------------------
// Physics fingerprint
// ---------------------------------------------------------------------------

// Pinned so that a change to the field list or its encoding cannot slip in
// unnoticed: it would make every existing snapshot refuse to resume. If the
// change is deliberate, bump PHYSICS_FINGERPRINT_VERSION and update these.
TEST(PhysicsFingerprint, MatchesTheGoldenValues) {
    EXPECT_EQ(cps::physics_fingerprint(nbody_reference()), 0x453ccb8480b7407bull);
    EXPECT_EQ(cps::physics_fingerprint(sph_reference()), 0x7eed893ee83cf351ull);
}

TEST(PhysicsFingerprint, RecordIsCanonicalJsonInListOrder) {
    EXPECT_EQ(cps::physics_json(cps::physics_fields(nbody_reference())),
              "{\"fingerprint_version\":1,\"fields\":{\"solver\":\"nbody\",\"variant\":\"fused\","
              "\"num_particles\":\"4\",\"dt\":\"0.001\",\"softening\":\"0.05\","
              "\"gravitational_constant\":\"1\",\"external_gravity.x\":\"0\","
              "\"external_gravity.y\":\"-9.81\",\"external_gravity.z\":\"0\"}}");
}

TEST(PhysicsFingerprint, ListsTheFluidGroupOnlyForTheFluidSolver) {
    auto has = [](const cps::Config& config, const std::string& name) {
        for (const cps::PhysicsField& field : cps::physics_fields(config)) {
            if (field.name == name) {
                return true;
            }
        }
        return false;
    };
    EXPECT_FALSE(has(nbody_reference(), "sph.viscosity"));
    EXPECT_TRUE(has(sph_reference(), "sph.viscosity"));
    EXPECT_TRUE(has(sph_reference(), "sph.domain_max.z"));
    EXPECT_FALSE(has(sph_reference(), "sph.fluid_max.x"));
    EXPECT_FALSE(has(sph_reference(), "seed"));
}

TEST(PhysicsFingerprint, ComparisonNamesEachDifferingFieldWithBothValues) {
    cps::Config resumed = nbody_reference();
    resumed.dt = 2.0e-3f;
    resumed.variant = cps::NBodyVariant::Tiled;
    const std::string recorded = cps::physics_json(cps::physics_fields(nbody_reference()));
    const std::vector<cps::PhysicsDifference> differences =
        cps::compare_physics(recorded, cps::physics_fields(resumed));
    ASSERT_EQ(differences.size(), 2u);
    EXPECT_EQ(differences[0].name, "variant");
    EXPECT_EQ(differences[0].snapshot_value, "fused");
    EXPECT_EQ(differences[0].config_value, "tiled");
    EXPECT_EQ(differences[1].name, "dt");
    EXPECT_EQ(cps::describe_differences(differences),
              "variant is fused in the snapshot and tiled in the configuration; "
              "dt is 0.001 in the snapshot and 0.002 in the configuration");

    EXPECT_TRUE(cps::compare_physics(recorded, cps::physics_fields(nbody_reference())).empty());
    EXPECT_THROW(cps::compare_physics("not json", cps::physics_fields(resumed)),
                 std::runtime_error);
}

// A record from another version of the list compares field by field: a field
// only one side knows is reported as absent on the other.
TEST(PhysicsFingerprint, ComparisonReportsFieldsOnlyOneSideKnows) {
    const std::string recorded =
        "{\"fingerprint_version\":9,\"fields\":{\"solver\":\"nbody\",\"variant\":\"fused\","
        "\"num_particles\":\"4\",\"dt\":\"0.001\",\"softening\":\"0.05\","
        "\"gravitational_constant\":\"1\",\"external_gravity.x\":\"0\","
        "\"external_gravity.y\":\"-9.81\",\"tidal_field\":\"0.5\"}}";
    const std::vector<cps::PhysicsDifference> differences =
        cps::compare_physics(recorded, cps::physics_fields(nbody_reference()));
    ASSERT_EQ(differences.size(), 2u);
    EXPECT_EQ(differences[0].name, "external_gravity.z");
    EXPECT_EQ(differences[0].snapshot_value, "(absent)");
    EXPECT_EQ(differences[1].name, "tidal_field");
    EXPECT_EQ(differences[1].config_value, "(absent)");
}

enum class Base { NBody, Sph, SphDerived };

struct FingerprintCase {
    const char* label;
    Base base;
    void (*change)(cps::Config&);
    bool fingerprint_moves;
};

void PrintTo(const FingerprintCase& value, std::ostream* out) { *out << value.label; }

class FingerprintField : public ::testing::TestWithParam<FingerprintCase> {};

TEST_P(FingerprintField, MovesExactlyWhenThePhysicsMoves) {
    const FingerprintCase& c = GetParam();
    const cps::Config base = c.base == Base::NBody ? nbody_reference()
                             : c.base == Base::Sph ? sph_reference()
                                                   : sph_derived_reference();
    cps::Config changed = base;
    c.change(changed);
    const uint64_t before = cps::physics_fingerprint(base);
    const uint64_t after = cps::physics_fingerprint(changed);
    if (c.fingerprint_moves) {
        EXPECT_NE(before, after) << c.label << " changes the physics but not the fingerprint";
    } else {
        EXPECT_EQ(before, after) << c.label << " is not physics but moves the fingerprint";
    }
}

// The fingerprint records the mass the solver uses, which for a derived fluid is
// the calibrated one.
TEST(PhysicsFields, ListTheCalibratedMass) {
    const cps::Config config = sph_derived_reference();
    const float calibrated = cps::derive_sph_parameters(config).particle_mass;
    bool found = false;
    for (const cps::PhysicsField& field : cps::physics_fields(config)) {
        if (field.name == "sph.particle_mass") {
            found = true;
            EXPECT_EQ(field.bits, std::bit_cast<uint32_t>(calibrated));
        }
    }
    EXPECT_TRUE(found);
}

INSTANTIATE_TEST_SUITE_P(
    Fields, FingerprintField,
    ::testing::Values(
        // Included, general group.
        FingerprintCase{"solver", Base::NBody,
                        [](cps::Config& c) { c.solver = cps::SolverType::Sph; }, true},
        FingerprintCase{"variant", Base::NBody,
                        [](cps::Config& c) { c.variant = cps::NBodyVariant::Naive; }, true},
        FingerprintCase{"num_particles", Base::NBody, [](cps::Config& c) { c.num_particles = 5; },
                        true},
        FingerprintCase{"dt", Base::NBody, [](cps::Config& c) { c.dt = 2.0e-3f; }, true},
        FingerprintCase{"dt_one_ulp", Base::NBody,
                        [](cps::Config& c) { c.dt = std::nextafter(c.dt, 1.0f); }, true},
        FingerprintCase{"softening", Base::NBody, [](cps::Config& c) { c.softening = 0.01f; },
                        true},
        FingerprintCase{"gravitational_constant", Base::NBody,
                        [](cps::Config& c) { c.gravitational_constant = 2.0f; }, true},
        FingerprintCase{"external_gravity_x", Base::NBody,
                        [](cps::Config& c) { c.external_gravity.x = 1.0f; }, true},
        FingerprintCase{"external_gravity_y", Base::NBody,
                        [](cps::Config& c) { c.external_gravity.y = -9.8f; }, true},
        FingerprintCase{"external_gravity_z", Base::NBody,
                        [](cps::Config& c) { c.external_gravity.z = 1.0f; }, true},
        FingerprintCase{"external_gravity_negative_zero", Base::NBody,
                        [](cps::Config& c) { c.external_gravity.x = -0.0f; }, true},
        FingerprintCase{"sph_dt", Base::Sph, [](cps::Config& c) { c.dt = 2.0e-4f; }, true},
        FingerprintCase{"sph_num_particles", Base::Sph,
                        [](cps::Config& c) { c.num_particles = 4000; }, true},
        // Included, fluid group.
        FingerprintCase{"smoothing_radius", Base::Sph,
                        [](cps::Config& c) { c.sph.smoothing_radius = 0.04f; }, true},
        FingerprintCase{"particle_mass", Base::Sph,
                        [](cps::Config& c) { c.sph.particle_mass = 0.07f; }, true},
        FingerprintCase{"rest_density", Base::Sph,
                        [](cps::Config& c) { c.sph.rest_density = 999.0f; }, true},
        FingerprintCase{"gas_constant", Base::Sph,
                        [](cps::Config& c) { c.sph.gas_constant = 500.0f; }, true},
        FingerprintCase{"viscosity", Base::Sph, [](cps::Config& c) { c.sph.viscosity = 0.1f; },
                        true},
        FingerprintCase{"boundary_damping", Base::Sph,
                        [](cps::Config& c) { c.sph.boundary_damping = 0.5f; }, true},
        FingerprintCase{"domain_min_x", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_min.x = -0.1f; }, true},
        FingerprintCase{"domain_min_y", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_min.y = -0.1f; }, true},
        FingerprintCase{"domain_min_z", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_min.z = -0.1f; }, true},
        FingerprintCase{"domain_max_x", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_max.x = 0.9f; }, true},
        FingerprintCase{"domain_max_y", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_max.y = 0.6f; }, true},
        FingerprintCase{"domain_max_z", Base::Sph,
                        [](cps::Config& c) { c.sph.domain_max.z = 0.5f; }, true},
        // A derived resolution moves with the fluid block and the count,
        // because that is what the solver then uses.
        FingerprintCase{"derived_smoothing_radius_follows_the_block", Base::SphDerived,
                        [](cps::Config& c) { c.sph.fluid_max.x = 0.35f; }, true},
        FingerprintCase{"derived_resolution_follows_the_count", Base::SphDerived,
                        [](cps::Config& c) { c.num_particles = 8192; }, true},
        // Excluded: run control.
        FingerprintCase{"steps", Base::NBody, [](cps::Config& c) { c.steps = 99; }, false},
        FingerprintCase{"snapshot_interval", Base::NBody,
                        [](cps::Config& c) { c.snapshot_interval = 7; }, false},
        FingerprintCase{"snapshot_dir", Base::NBody,
                        [](cps::Config& c) { c.snapshot_dir = "elsewhere"; }, false},
        FingerprintCase{"diagnostics_interval", Base::NBody,
                        [](cps::Config& c) { c.diagnostics_interval = 3; }, false},
        FingerprintCase{"diagnostics_path", Base::NBody,
                        [](cps::Config& c) { c.diagnostics_path = "other.csv"; }, false},
        FingerprintCase{"vtk_interval", Base::NBody,
                        [](cps::Config& c) { c.vtk_interval = 3; }, false},
        FingerprintCase{"vtk_dir", Base::NBody, [](cps::Config& c) { c.vtk_dir = "elsewhere"; },
                        false},
        FingerprintCase{"name", Base::NBody, [](cps::Config& c) { c.name = "renamed"; }, false},
        FingerprintCase{"block_size", Base::NBody, [](cps::Config& c) { c.block_size = 64; },
                        false},
        FingerprintCase{"allow_unstable_dt", Base::Sph,
                        [](cps::Config& c) { c.allow_unstable_dt = true; }, false},
        FingerprintCase{"sph_block_size", Base::Sph, [](cps::Config& c) { c.block_size = 64; },
                        false},
        FingerprintCase{"tile_size", Base::NBody, [](cps::Config& c) { c.nbody.tile_size = 64; },
                        false},
        FingerprintCase{"sph_tile_size", Base::Sph, [](cps::Config& c) { c.nbody.tile_size = 64; },
                        false},
        // Excluded: initial state only.
        FingerprintCase{"seed", Base::NBody, [](cps::Config& c) { c.seed = 1u; }, false},
        FingerprintCase{"initial_condition", Base::NBody,
                        [](cps::Config& c) { c.initial_condition = cps::InitialCondition::TwoBody; },
                        false},
        FingerprintCase{"plummer_radius", Base::NBody,
                        [](cps::Config& c) { c.plummer_radius = 2.0f; }, false},
        FingerprintCase{"total_mass", Base::NBody, [](cps::Config& c) { c.total_mass = 3.0f; },
                        false},
        FingerprintCase{"two_body_separation", Base::NBody,
                        [](cps::Config& c) { c.two_body_separation = 2.0f; }, false},
        FingerprintCase{"sph_seed", Base::Sph, [](cps::Config& c) { c.seed = 1u; }, false},
        FingerprintCase{"jitter", Base::Sph, [](cps::Config& c) { c.sph.jitter = 0.1f; }, false},
        FingerprintCase{"fluid_min", Base::Sph,
                        [](cps::Config& c) { c.sph.fluid_min = {0.05f, 0.05f, 0.05f}; }, false},
        FingerprintCase{"fluid_max", Base::Sph,
                        [](cps::Config& c) { c.sph.fluid_max = {0.35f, 0.3f, 0.3f}; }, false},
        // Excluded: the fluid group, for the N-body solver, which never reads it.
        FingerprintCase{"nbody_ignores_viscosity", Base::NBody,
                        [](cps::Config& c) { c.sph.viscosity = 0.1f; }, false},
        FingerprintCase{"nbody_ignores_fluid_block", Base::NBody,
                        [](cps::Config& c) { c.sph.fluid_max.x = 0.35f; }, false},
        FingerprintCase{"nbody_ignores_domain", Base::NBody,
                        [](cps::Config& c) { c.sph.domain_max.x = 2.0f; }, false}),
    [](const ::testing::TestParamInfo<FingerprintCase>& info) {
        return std::string(info.param.label);
    });

}  // namespace
