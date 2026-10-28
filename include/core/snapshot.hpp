#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cps {

// Binary snapshot format, version 2 ("CPS2").
//
// Everything is little endian, floats are IEEE 754, and the file is three
// regions in a row: a fixed 64 byte header, a table with one 40 byte entry per
// array, and the payload that the table points into.
//
//   offset  size  header field
//        0     4  magic                "CPS2"
//        4     4  version              2
//        8     4  flags                unknown bits are rejected on read
//       12     4  array_count          entries in the table
//       16     8  step                 step index the state belongs to
//       24     8  time                 simulated time, double
//       32     8  count                particles
//       40     8  fingerprint          physics_fingerprint() of the run
//       48     8  payload_checksum     FNV-1a 64 of every byte after the table
//       56     4  fingerprint_version  PHYSICS_FINGERPRINT_VERSION of the writer
//       60     4  reserved             zero
//
//   offset  size  array table entry
//        0    16  name                 NUL padded, at most 15 characters
//       16     4  dtype                a SnapshotDtype code
//       20     4  reserved             written as zero, ignored on read
//       24     8  element_count
//       32     8  offset               from the start of the file
//
// The writer emits x, y, z, vx, vy, vz and mass (f32, count elements each) and
// physics (u8: the canonical physics JSON from physics_json()), each starting
// on an eight byte boundary with zero padding in between. The table is what
// keeps this the last format bump: a later version adds arrays such as id,
// type or density as new entries, and this reader skips names it does not
// know, so older builds keep loading newer files.
//
// The flags word carries the one piece of information a reader cannot infer
// from the numbers, namely whether the stored velocities are synchronised with
// the positions or offset by half a step. The fused kernel advances a leapfrog
// velocity, so resuming it with the wrong convention would silently shift the
// trajectory by half a step and quietly break the bit identity test.
//
// Accelerations are not stored: they are a deterministic function of the
// positions, and recomputing them on load is both cheaper than storing them
// and a guarantee that the resumed run enters the loop in exactly the state the
// original run was in.
//
// Version 1 ("CPS1") is still read, never written: a sixteen byte header of
// magic, count, step and flags (uint32 each) followed by the seven float
// arrays, with no table, no time and no record of the configuration.

inline constexpr uint32_t SNAPSHOT_MAGIC = 0x32535043u;     // "CPS2" in little endian
inline constexpr uint32_t SNAPSHOT_MAGIC_V1 = 0x31535043u;  // "CPS1" in little endian
inline constexpr uint32_t SNAPSHOT_VERSION = 2;
inline constexpr uint32_t SNAPSHOT_FLAG_HALF_STEP_VELOCITY = 1u << 0;
inline constexpr uint32_t SNAPSHOT_KNOWN_FLAGS = SNAPSHOT_FLAG_HALF_STEP_VELOCITY;
inline constexpr int SNAPSHOT_ARRAY_COUNT = 7;  // the particle arrays, not counting physics
inline constexpr std::size_t SNAPSHOT_ARRAY_NAME_BYTES = 16;

// Explicit values: they are on disk, so they are never renumbered.
enum class SnapshotDtype : uint32_t {
    U8 = 1,
    U16 = 2,
    U32 = 3,
    U64 = 4,
    I8 = 5,
    I16 = 6,
    I32 = 7,
    I64 = 8,
    F32 = 9,
    F64 = 10,
};

// Bytes per element, or zero for a code this build does not know.
std::size_t snapshot_dtype_size(uint32_t code);

struct SnapshotHeader {
    uint32_t magic = SNAPSHOT_MAGIC;
    uint32_t version = SNAPSHOT_VERSION;
    uint32_t flags = 0;
    uint32_t array_count = 0;
    uint64_t step = 0;
    double time = 0.0;
    uint64_t count = 0;
    uint64_t fingerprint = 0;
    uint64_t payload_checksum = 0;
    uint32_t fingerprint_version = 0;
    uint32_t reserved = 0;
};

static_assert(sizeof(SnapshotHeader) == 64, "the snapshot header must stay 64 bytes");
static_assert(offsetof(SnapshotHeader, magic) == 0);
static_assert(offsetof(SnapshotHeader, version) == 4);
static_assert(offsetof(SnapshotHeader, flags) == 8);
static_assert(offsetof(SnapshotHeader, array_count) == 12);
static_assert(offsetof(SnapshotHeader, step) == 16);
static_assert(offsetof(SnapshotHeader, time) == 24);
static_assert(offsetof(SnapshotHeader, count) == 32);
static_assert(offsetof(SnapshotHeader, fingerprint) == 40);
static_assert(offsetof(SnapshotHeader, payload_checksum) == 48);
static_assert(offsetof(SnapshotHeader, fingerprint_version) == 56);
static_assert(offsetof(SnapshotHeader, reserved) == 60);

struct SnapshotArrayEntry {
    char name[SNAPSHOT_ARRAY_NAME_BYTES] = {};
    uint32_t dtype = 0;
    uint32_t reserved = 0;
    uint64_t element_count = 0;
    uint64_t offset = 0;
};

static_assert(sizeof(SnapshotArrayEntry) == 40, "an array table entry must stay 40 bytes");
static_assert(offsetof(SnapshotArrayEntry, name) == 0);
static_assert(offsetof(SnapshotArrayEntry, dtype) == 16);
static_assert(offsetof(SnapshotArrayEntry, reserved) == 20);
static_assert(offsetof(SnapshotArrayEntry, element_count) == 24);
static_assert(offsetof(SnapshotArrayEntry, offset) == 32);

// The version 1 header, kept only so old files can still be read.
struct SnapshotHeaderV1 {
    uint32_t magic = SNAPSHOT_MAGIC_V1;
    uint32_t count = 0;
    uint32_t step = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(SnapshotHeaderV1) == 16, "the version 1 header is sixteen bytes");

// Host side particle state, structure of arrays to mirror the device layout.
struct HostParticleState {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;
    std::vector<float> mass;

    void resize(std::size_t n);
    std::size_t size() const { return x.size(); }
};

struct Snapshot {
    uint64_t step = 0;
    double time = 0.0;
    uint32_t flags = 0;
    HostParticleState state;

    // What the run's physics was. The writer stores these as given; the
    // engine fills them from the configuration.
    uint64_t fingerprint = 0;
    uint32_t fingerprint_version = 0;
    std::string physics;

    // Set by read_snapshot. A version 1 file has no time, fingerprint or
    // physics record: time is NaN and the other two are left empty.
    uint32_t format_version = SNAPSHOT_VERSION;
};

// 64 bit FNV-1a. Pass the previous result as the basis to hash in pieces.
inline constexpr uint64_t FNV1A_64_BASIS = 0xcbf29ce484222325ull;
uint64_t fnv1a_64(const void* data, std::size_t bytes, uint64_t basis = FNV1A_64_BASIS);

// Always writes version 2. Writes to a temporary file in the same directory
// and renames it into place. A rename within one filesystem is atomic, so a
// reader never observes a partially written snapshot even if the process is
// killed mid write.
void write_snapshot(const std::filesystem::path& path, const Snapshot& snapshot);

// Reads version 2 or version 1. Every size, offset and count in the file is
// checked against the file size before anything is allocated from it, so a
// corrupt or hostile file gives an error naming the file rather than a huge
// allocation. Throws std::runtime_error.
Snapshot read_snapshot(const std::filesystem::path& path);

// snap_ followed by the step, zero padded to at least nine digits.
std::filesystem::path snapshot_path_for_step(const std::filesystem::path& directory, uint64_t step);

// The snapshot with the highest step number in the directory, or an empty
// path when none exists. Steps are compared as numbers, so a wider name such
// as snap_1000000000.bin sorts after snap_999999999.bin. Used by the resume
// path to find where to pick up.
std::filesystem::path latest_snapshot(const std::filesystem::path& directory);

}  // namespace cps
