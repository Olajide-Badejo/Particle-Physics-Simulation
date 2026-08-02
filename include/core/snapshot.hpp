#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace cps {

// Binary snapshot format, version 1.
//
// The header is exactly sixteen bytes: a magic number, the particle count, the
// step index the state belongs to, and a flags word. The flags word carries
// the one piece of information a reader cannot infer from the numbers, namely
// whether the stored velocities are synchronised with the positions or offset
// by half a step. The fused kernel advances a leapfrog velocity, so resuming
// it with the wrong convention would silently shift the trajectory by half a
// step and quietly break the bit identity test.
//
// Arrays follow the header as seven contiguous float blocks of `count`
// elements each, in the order x, y, z, vx, vy, vz, mass. Accelerations are not
// stored: they are a deterministic function of the positions, and recomputing
// them on load is both cheaper than storing them and a guarantee that the
// resumed run enters the loop in exactly the state the original run was in.

inline constexpr uint32_t SNAPSHOT_MAGIC = 0x31535043u;  // "CPS1" in little endian
inline constexpr uint32_t SNAPSHOT_FLAG_HALF_STEP_VELOCITY = 1u << 0;
inline constexpr int SNAPSHOT_ARRAY_COUNT = 7;

struct SnapshotHeader {
    uint32_t magic = SNAPSHOT_MAGIC;
    uint32_t count = 0;
    uint32_t step = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(SnapshotHeader) == 16, "the snapshot header must stay sixteen bytes");

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
    uint32_t step = 0;
    uint32_t flags = 0;
    HostParticleState state;
};

// Writes to a temporary file in the same directory and renames it into place.
// A rename within one filesystem is atomic, so a reader never observes a
// partially written snapshot even if the process is killed mid write.
void write_snapshot(const std::filesystem::path& path, const Snapshot& snapshot);

Snapshot read_snapshot(const std::filesystem::path& path);

std::filesystem::path snapshot_path_for_step(const std::filesystem::path& directory, uint32_t step);

// Highest numbered snapshot in the directory, or an empty optional when none
// exists. Used by the resume path to find where to pick up.
std::filesystem::path latest_snapshot(const std::filesystem::path& directory);

}  // namespace cps
