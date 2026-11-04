#include "cps/core/snapshot.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

namespace cps {

// The file is the memory image of these structs and arrays, so the format is
// only what the header says it is on a little endian machine with IEEE floats.
static_assert(std::endian::native == std::endian::little,
              "the snapshot format is little endian and is read and written as raw memory");
static_assert(std::numeric_limits<float>::is_iec559, "snapshots store IEEE 754 single precision");
static_assert(std::numeric_limits<double>::is_iec559, "snapshots store IEEE 754 double precision");

namespace {

constexpr uint64_t FNV1A_64_PRIME = 0x100000001b3ull;

// The float arrays in the order the writer emits them. Id follows them.
constexpr std::array<const char*, SNAPSHOT_ARRAY_COUNT> PARTICLE_ARRAY_NAMES = {
    "x", "y", "z", "vx", "vy", "vz", "mass"};
constexpr const char* ID_ARRAY_NAME = "id";
constexpr const char* PHYSICS_ARRAY_NAME = "physics";

uint64_t round_up_to_eight(uint64_t value) { return (value + 7u) & ~uint64_t{7u}; }

std::string hex(uint64_t value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%016" PRIx64, value);
    return buffer;
}

std::vector<float>* particle_array(HostParticleState& state, std::size_t index) {
    std::array<std::vector<float>*, SNAPSHOT_ARRAY_COUNT> arrays = {
        &state.x, &state.y, &state.z, &state.vx, &state.vy, &state.vz, &state.mass};
    return arrays[index];
}

const std::vector<float>* particle_array(const HostParticleState& state, std::size_t index) {
    return particle_array(const_cast<HostParticleState&>(state), index);
}

// One array as the writer lays it out: where its bytes come from and where
// they land in the file.
struct PayloadBlock {
    SnapshotArrayEntry entry;
    const void* data = nullptr;
    uint64_t bytes = 0;
};

void write_bytes(std::FILE* file, const void* data, std::size_t bytes, const fs::path& path) {
    if (bytes == 0) {
        return;
    }
    if (std::fwrite(data, 1, bytes, file) != bytes) {
        throw std::runtime_error("short write while saving snapshot " + path.string());
    }
}

void read_bytes(std::FILE* file, void* data, std::size_t bytes, const fs::path& path,
                const char* what) {
    if (bytes == 0) {
        return;
    }
    if (std::fread(data, 1, bytes, file) != bytes) {
        throw std::runtime_error("snapshot " + path.string() + " is truncated: cannot read " +
                                 what);
    }
}

// The name up to its first NUL. Throws if it is empty, has no NUL, or has
// anything but NULs after the first one, since any of those means the table
// is not what a writer produced.
std::string entry_name(const SnapshotArrayEntry& entry, std::size_t index, const fs::path& path) {
    const char* begin = entry.name;
    const char* end = entry.name + SNAPSHOT_ARRAY_NAME_BYTES;
    const char* nul = std::find(begin, end, '\0');
    const bool padded = nul != end && std::all_of(nul, end, [](char c) { return c == '\0'; });
    if (nul == begin || !padded) {
        throw std::runtime_error("snapshot " + path.string() + ": array table entry " +
                                 std::to_string(index) + " has a malformed name");
    }
    return std::string(begin, nul);
}

// Ids are the one thing a resume trusts that the checksum cannot vouch for: a
// writer that got them wrong still checksums correctly.
void check_id_permutation(const std::vector<uint32_t>& id, const std::string& where) {
    std::vector<bool> seen(id.size(), false);
    for (std::size_t i = 0; i < id.size(); ++i) {
        const uint32_t value = id[i];
        if (value >= id.size()) {
            throw std::runtime_error(where + ": array 'id' holds " + std::to_string(value) +
                                     " at index " + std::to_string(i) +
                                     ", which is not below the particle count " +
                                     std::to_string(id.size()));
        }
        if (seen[value]) {
            throw std::runtime_error(where + ": array 'id' holds " + std::to_string(value) +
                                     " more than once, the second time at index " +
                                     std::to_string(i));
        }
        seen[value] = true;
    }
}

Snapshot read_version_1(std::FILE* file, uint64_t file_size, const fs::path& path) {
    if (file_size < sizeof(SnapshotHeaderV1)) {
        throw std::runtime_error("snapshot " + path.string() + " is truncated: " +
                                 std::to_string(file_size) + " bytes is less than the " +
                                 std::to_string(sizeof(SnapshotHeaderV1)) + " byte CPS1 header");
    }
    SnapshotHeaderV1 header{};
    read_bytes(file, &header, sizeof(header), path, "the CPS1 header");
    if ((header.flags & ~SNAPSHOT_KNOWN_FLAGS) != 0u) {
        throw std::runtime_error("snapshot " + path.string() + " sets unknown flag bits " +
                                 hex(header.flags & ~SNAPSHOT_KNOWN_FLAGS));
    }
    // The size is fully determined by the count, so checking it first is
    // what stops a corrupt count from turning into a multi gigabyte resize.
    const uint64_t expected = sizeof(SnapshotHeaderV1) + uint64_t{SNAPSHOT_ARRAY_COUNT} *
                                                             sizeof(float) * header.count;
    if (file_size != expected) {
        throw std::runtime_error("snapshot " + path.string() + " has a size mismatch: its CPS1 " +
                                 "header claims " + std::to_string(header.count) +
                                 " particles, which needs " + std::to_string(expected) +
                                 " bytes, but the file has " + std::to_string(file_size));
    }

    Snapshot snapshot;
    snapshot.format_version = 1;
    snapshot.step = header.step;
    snapshot.time = std::numeric_limits<double>::quiet_NaN();
    snapshot.flags = header.flags;
    snapshot.fingerprint_version = 0;
    snapshot.state.resize(header.count);
    for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
        std::vector<float>& block = *particle_array(snapshot.state, a);
        read_bytes(file, block.data(), block.size() * sizeof(float), path, PARTICLE_ARRAY_NAMES[a]);
    }
    return snapshot;
}

Snapshot read_version_2(std::FILE* file, uint64_t file_size, const fs::path& path) {
    const std::string where = "snapshot " + path.string();
    if (file_size < sizeof(SnapshotHeader)) {
        throw std::runtime_error(where + " is truncated: " + std::to_string(file_size) +
                                 " bytes is less than the " +
                                 std::to_string(sizeof(SnapshotHeader)) + " byte CPS2 header");
    }
    SnapshotHeader header{};
    read_bytes(file, &header, sizeof(header), path, "the CPS2 header");
    if (header.version != SNAPSHOT_VERSION) {
        throw std::runtime_error(where + " has format version " + std::to_string(header.version) +
                                 ", but this build reads CPS2 version " +
                                 std::to_string(SNAPSHOT_VERSION));
    }
    if ((header.flags & ~SNAPSHOT_KNOWN_FLAGS) != 0u) {
        throw std::runtime_error(where + " sets unknown flag bits " +
                                 hex(header.flags & ~SNAPSHOT_KNOWN_FLAGS));
    }
    if (header.reserved != 0u) {
        throw std::runtime_error(where + " has a nonzero reserved header field");
    }

    const uint64_t table_end =
        sizeof(SnapshotHeader) + uint64_t{header.array_count} * sizeof(SnapshotArrayEntry);
    if (table_end > file_size) {
        throw std::runtime_error(where + " is truncated: its array table of " +
                                 std::to_string(header.array_count) + " entries needs " +
                                 std::to_string(table_end) + " bytes, but the file has " +
                                 std::to_string(file_size));
    }
    std::vector<SnapshotArrayEntry> table(header.array_count);
    read_bytes(file, table.data(), table.size() * sizeof(SnapshotArrayEntry), path,
               "the array table");

    // Find the arrays this build needs, checking every entry on the way,
    // including the ones it is about to skip.
    std::array<const SnapshotArrayEntry*, SNAPSHOT_ARRAY_COUNT> particle_entries{};
    const SnapshotArrayEntry* id_entry = nullptr;
    const SnapshotArrayEntry* physics_entry = nullptr;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < table.size(); ++i) {
        const SnapshotArrayEntry& entry = table[i];
        const std::string name = entry_name(entry, i, path);
        if (!seen.insert(name).second) {
            throw std::runtime_error(where + " lists array '" + name + "' more than once");
        }
        const std::size_t element_size = snapshot_dtype_size(entry.dtype);
        if (element_size == 0) {
            throw std::runtime_error(where + ": array '" + name + "' has unknown dtype code " +
                                     std::to_string(entry.dtype));
        }
        // Written so that nothing overflows: each bound is checked before it
        // is used in the next product or sum.
        const bool in_bounds = entry.offset >= table_end && entry.offset <= file_size &&
                               entry.element_count <= (file_size - entry.offset) / element_size;
        if (!in_bounds) {
            throw std::runtime_error(where + ": array '" + name + "' of " +
                                     std::to_string(entry.element_count) +
                                     " elements at offset " + std::to_string(entry.offset) +
                                     " lies outside the payload, which runs from byte " +
                                     std::to_string(table_end) + " to " +
                                     std::to_string(file_size));
        }

        for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
            if (name == PARTICLE_ARRAY_NAMES[a]) {
                if (entry.dtype != static_cast<uint32_t>(SnapshotDtype::F32) ||
                    entry.element_count != header.count) {
                    throw std::runtime_error(
                        where + ": array '" + name + "' must be f32 with " +
                        std::to_string(header.count) + " elements, but has dtype code " +
                        std::to_string(entry.dtype) + " and " +
                        std::to_string(entry.element_count) + " elements");
                }
                particle_entries[a] = &entry;
            }
        }
        if (name == ID_ARRAY_NAME) {
            if (entry.dtype != static_cast<uint32_t>(SnapshotDtype::U32) ||
                entry.element_count != header.count) {
                throw std::runtime_error(where + ": array 'id' must be u32 with " +
                                         std::to_string(header.count) +
                                         " elements, but has dtype code " +
                                         std::to_string(entry.dtype) + " and " +
                                         std::to_string(entry.element_count) + " elements");
            }
            id_entry = &entry;
        }
        if (name == PHYSICS_ARRAY_NAME) {
            if (entry.dtype != static_cast<uint32_t>(SnapshotDtype::U8)) {
                throw std::runtime_error(where + ": array 'physics' must be u8, but has dtype code " +
                                         std::to_string(entry.dtype));
            }
            physics_entry = &entry;
        }
        // Any other name belongs to a newer writer and is skipped.
    }
    for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
        if (particle_entries[a] == nullptr) {
            throw std::runtime_error(where + " is missing the required array '" +
                                     std::string(PARTICLE_ARRAY_NAMES[a]) + "'");
        }
    }
    if (physics_entry == nullptr) {
        throw std::runtime_error(where + " is missing the required array 'physics'");
    }

    // Every count and offset is now known to lie inside the file, so the
    // allocations below are bounded by its size.
    std::vector<unsigned char> payload(static_cast<std::size_t>(file_size - table_end));
    read_bytes(file, payload.data(), payload.size(), path, "the payload");
    const uint64_t checksum = fnv1a_64(payload.data(), payload.size());
    if (checksum != header.payload_checksum) {
        throw std::runtime_error(where + " is corrupt: its payload checksum is " + hex(checksum) +
                                 " but the header records " + hex(header.payload_checksum));
    }

    Snapshot snapshot;
    snapshot.format_version = SNAPSHOT_VERSION;
    snapshot.step = header.step;
    snapshot.time = header.time;
    snapshot.flags = header.flags;
    snapshot.fingerprint = header.fingerprint;
    snapshot.fingerprint_version = header.fingerprint_version;
    snapshot.state.resize(static_cast<std::size_t>(header.count));
    for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
        std::vector<float>& block = *particle_array(snapshot.state, a);
        if (!block.empty()) {
            std::memcpy(block.data(), payload.data() + (particle_entries[a]->offset - table_end),
                        block.size() * sizeof(float));
        }
    }
    // With no id array the file order stands, which is what resize just set up.
    if (id_entry != nullptr) {
        if (!snapshot.state.id.empty()) {
            std::memcpy(snapshot.state.id.data(), payload.data() + (id_entry->offset - table_end),
                        snapshot.state.id.size() * sizeof(uint32_t));
        }
        check_id_permutation(snapshot.state.id, where);
    }
    const unsigned char* physics = payload.data() + (physics_entry->offset - table_end);
    snapshot.physics.assign(reinterpret_cast<const char*>(physics),
                            static_cast<std::size_t>(physics_entry->element_count));
    return snapshot;
}

}  // namespace

std::size_t snapshot_dtype_size(uint32_t code) {
    switch (static_cast<SnapshotDtype>(code)) {
        case SnapshotDtype::U8:
        case SnapshotDtype::I8:
            return 1;
        case SnapshotDtype::U16:
        case SnapshotDtype::I16:
            return 2;
        case SnapshotDtype::U32:
        case SnapshotDtype::I32:
        case SnapshotDtype::F32:
            return 4;
        case SnapshotDtype::U64:
        case SnapshotDtype::I64:
        case SnapshotDtype::F64:
            return 8;
    }
    return 0;
}

uint64_t fnv1a_64(const void* data, std::size_t bytes, uint64_t basis) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t hash = basis;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= FNV1A_64_PRIME;
    }
    return hash;
}

void HostParticleState::resize(std::size_t n) {
    x.assign(n, 0.0f);
    y.assign(n, 0.0f);
    z.assign(n, 0.0f);
    vx.assign(n, 0.0f);
    vy.assign(n, 0.0f);
    vz.assign(n, 0.0f);
    mass.assign(n, 0.0f);
    id.resize(n);
    std::iota(id.begin(), id.end(), uint32_t{0});
}

void write_snapshot(const fs::path& path, const Snapshot& snapshot) {
    const std::size_t count = snapshot.state.size();
    for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
        if (particle_array(snapshot.state, a)->size() != count) {
            throw std::runtime_error("snapshot arrays have inconsistent lengths");
        }
    }
    if (snapshot.state.id.size() != count) {
        throw std::runtime_error("snapshot arrays have inconsistent lengths: " +
                                 std::to_string(count) + " particles but " +
                                 std::to_string(snapshot.state.id.size()) + " ids");
    }
    if ((snapshot.flags & ~SNAPSHOT_KNOWN_FLAGS) != 0u) {
        throw std::runtime_error("cannot write unknown snapshot flag bits " +
                                 hex(snapshot.flags & ~SNAPSHOT_KNOWN_FLAGS));
    }

    // Lay out the table and payload. Each array starts on an eight byte
    // boundary so a reader can map it straight into a typed array.
    std::vector<PayloadBlock> blocks;
    auto add_block = [&blocks](const char* name, SnapshotDtype dtype, uint64_t elements,
                               const void* data) {
        PayloadBlock block;
        const std::size_t length = std::strlen(name);
        if (length == 0 || length >= SNAPSHOT_ARRAY_NAME_BYTES) {
            throw std::logic_error(std::string("snapshot array name '") + name +
                                   "' does not fit the table");
        }
        std::memcpy(block.entry.name, name, length);
        block.entry.dtype = static_cast<uint32_t>(dtype);
        block.entry.element_count = elements;
        block.data = data;
        block.bytes = elements * snapshot_dtype_size(block.entry.dtype);
        blocks.push_back(block);
    };
    for (std::size_t a = 0; a < SNAPSHOT_ARRAY_COUNT; ++a) {
        add_block(PARTICLE_ARRAY_NAMES[a], SnapshotDtype::F32, count,
                  particle_array(snapshot.state, a)->data());
    }
    add_block(ID_ARRAY_NAME, SnapshotDtype::U32, count, snapshot.state.id.data());
    add_block(PHYSICS_ARRAY_NAME, SnapshotDtype::U8, snapshot.physics.size(),
              snapshot.physics.data());

    const uint64_t table_end = sizeof(SnapshotHeader) + blocks.size() * sizeof(SnapshotArrayEntry);
    uint64_t cursor = table_end;
    static const std::array<unsigned char, 8> zeros{};
    uint64_t checksum = FNV1A_64_BASIS;
    for (PayloadBlock& block : blocks) {
        block.entry.offset = cursor;
        const uint64_t padding = round_up_to_eight(block.bytes) - block.bytes;
        checksum = fnv1a_64(block.data, static_cast<std::size_t>(block.bytes), checksum);
        checksum = fnv1a_64(zeros.data(), static_cast<std::size_t>(padding), checksum);
        cursor += block.bytes + padding;
    }

    SnapshotHeader header;
    header.flags = snapshot.flags;
    header.array_count = static_cast<uint32_t>(blocks.size());
    header.step = snapshot.step;
    header.time = snapshot.time;
    header.count = count;
    header.fingerprint = snapshot.fingerprint;
    header.payload_checksum = checksum;
    header.fingerprint_version = snapshot.fingerprint_version;

    const fs::path directory = path.parent_path();
    if (!directory.empty()) {
        fs::create_directories(directory);
    }

    // Write to a sibling temporary file and rename. Rename within a single
    // filesystem is atomic, so a reader either sees the previous snapshot or
    // the complete new one, never a half written file. Writing in place would
    // leave a corrupt snapshot behind if the run were interrupted during the
    // write, and that snapshot is exactly the one a resume would pick up.
    const fs::path temporary = path.string() + ".tmp";
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open " + temporary.string() + " for writing");
    }

    try {
        write_bytes(file, &header, sizeof(header), temporary);
        for (const PayloadBlock& block : blocks) {
            write_bytes(file, &block.entry, sizeof(block.entry), temporary);
        }
        for (const PayloadBlock& block : blocks) {
            write_bytes(file, block.data, static_cast<std::size_t>(block.bytes), temporary);
            write_bytes(file, zeros.data(),
                        static_cast<std::size_t>(round_up_to_eight(block.bytes) - block.bytes),
                        temporary);
        }
        if (std::fflush(file) != 0) {
            throw std::runtime_error("cannot flush " + temporary.string());
        }
    } catch (...) {
        std::fclose(file);
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
    // A failed close can mean the last buffered bytes never reached the disk,
    // so it is a failed write, not a detail.
    if (std::fclose(file) != 0) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw std::runtime_error("cannot close " + temporary.string() + " after writing it");
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        // Rename across an existing file fails on some filesystems; remove and
        // retry rather than leaving the temporary behind.
        std::error_code ignored;
        fs::remove(path, ignored);
        fs::rename(temporary, path, error);
        if (error) {
            fs::remove(temporary, ignored);
            throw std::runtime_error("cannot move snapshot into place at " + path.string() + ": " +
                                     error.message());
        }
    }
}

Snapshot read_snapshot(const fs::path& path) {
    std::error_code error;
    const uint64_t file_size = fs::file_size(path, error);
    if (error) {
        throw std::runtime_error("cannot open snapshot " + path.string() + ": " + error.message());
    }
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open snapshot " + path.string());
    }

    try {
        uint32_t magic = 0;
        if (file_size < sizeof(magic) || std::fread(&magic, sizeof(magic), 1, file) != 1) {
            throw std::runtime_error("snapshot " + path.string() +
                                     " is too short to hold a magic number");
        }
        std::rewind(file);
        Snapshot snapshot;
        if (magic == SNAPSHOT_MAGIC) {
            snapshot = read_version_2(file, file_size, path);
        } else if (magic == SNAPSHOT_MAGIC_V1) {
            snapshot = read_version_1(file, file_size, path);
        } else {
            throw std::runtime_error(path.string() + " is not a snapshot of this simulation: its " +
                                     "magic number is " + hex(magic) + ", not CPS2 or CPS1");
        }
        std::fclose(file);
        return snapshot;
    } catch (...) {
        std::fclose(file);
        throw;
    }
}

fs::path snapshot_path_for_step(const fs::path& directory, uint64_t step) {
    char name[64];
    std::snprintf(name, sizeof(name), "snap_%09" PRIu64 ".bin", step);
    return directory / name;
}

fs::path latest_snapshot(const fs::path& directory) {
    if (!fs::exists(directory)) {
        return {};
    }
    static const std::regex pattern(R"(snap_(\d+)\.bin)");
    fs::path best;
    uint64_t best_step = 0;
    bool found = false;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        // A number too large for 64 bits is not a step this code wrote.
        const std::string digits = match[1].str();
        uint64_t step = 0;
        const auto [end, parse_error] =
            std::from_chars(digits.data(), digits.data() + digits.size(), step);
        if (parse_error != std::errc() || end != digits.data() + digits.size()) {
            continue;
        }
        // Equal steps under different paddings are broken by name, so the
        // choice does not depend on the directory iteration order.
        const bool better = !found || step > best_step ||
                            (step == best_step && entry.path().filename() > best.filename());
        if (better) {
            best_step = step;
            best = entry.path();
            found = true;
        }
    }
    return best;
}

}  // namespace cps
