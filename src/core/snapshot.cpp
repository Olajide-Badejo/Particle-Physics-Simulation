#include "core/snapshot.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace cps {
namespace {

void write_block(std::FILE* file, const std::vector<float>& block, const fs::path& path) {
    if (block.empty()) {
        return;
    }
    const std::size_t written = std::fwrite(block.data(), sizeof(float), block.size(), file);
    if (written != block.size()) {
        throw std::runtime_error("short write while saving snapshot " + path.string());
    }
}

void read_block(std::FILE* file, std::vector<float>& block, std::size_t count,
                const fs::path& path) {
    block.resize(count);
    if (count == 0) {
        return;
    }
    const std::size_t read = std::fread(block.data(), sizeof(float), count, file);
    if (read != count) {
        throw std::runtime_error("truncated snapshot " + path.string());
    }
}

}  // namespace

void HostParticleState::resize(std::size_t n) {
    x.assign(n, 0.0f);
    y.assign(n, 0.0f);
    z.assign(n, 0.0f);
    vx.assign(n, 0.0f);
    vy.assign(n, 0.0f);
    vz.assign(n, 0.0f);
    mass.assign(n, 0.0f);
}

void write_snapshot(const fs::path& path, const Snapshot& snapshot) {
    const std::size_t count = snapshot.state.size();
    if (snapshot.state.y.size() != count || snapshot.state.z.size() != count ||
        snapshot.state.vx.size() != count || snapshot.state.vy.size() != count ||
        snapshot.state.vz.size() != count || snapshot.state.mass.size() != count) {
        throw std::runtime_error("snapshot arrays have inconsistent lengths");
    }

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
        SnapshotHeader header;
        header.magic = SNAPSHOT_MAGIC;
        header.count = static_cast<uint32_t>(count);
        header.step = snapshot.step;
        header.flags = snapshot.flags;
        if (std::fwrite(&header, sizeof(header), 1, file) != 1) {
            throw std::runtime_error("cannot write snapshot header to " + temporary.string());
        }
        write_block(file, snapshot.state.x, temporary);
        write_block(file, snapshot.state.y, temporary);
        write_block(file, snapshot.state.z, temporary);
        write_block(file, snapshot.state.vx, temporary);
        write_block(file, snapshot.state.vy, temporary);
        write_block(file, snapshot.state.vz, temporary);
        write_block(file, snapshot.state.mass, temporary);
        if (std::fflush(file) != 0) {
            throw std::runtime_error("cannot flush " + temporary.string());
        }
    } catch (...) {
        std::fclose(file);
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
    std::fclose(file);

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        // Rename across an existing file fails on some filesystems; remove and
        // retry rather than leaving the temporary behind.
        std::error_code ignored;
        fs::remove(path, ignored);
        fs::rename(temporary, path, error);
        if (error) {
            throw std::runtime_error("cannot move snapshot into place at " + path.string() + ": " +
                                     error.message());
        }
    }
}

Snapshot read_snapshot(const fs::path& path) {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open snapshot " + path.string());
    }

    Snapshot snapshot;
    try {
        SnapshotHeader header{};
        if (std::fread(&header, sizeof(header), 1, file) != 1) {
            throw std::runtime_error("cannot read snapshot header from " + path.string());
        }
        if (header.magic != SNAPSHOT_MAGIC) {
            throw std::runtime_error(path.string() + " is not a snapshot of this simulation");
        }
        snapshot.step = header.step;
        snapshot.flags = header.flags;
        const std::size_t count = header.count;
        read_block(file, snapshot.state.x, count, path);
        read_block(file, snapshot.state.y, count, path);
        read_block(file, snapshot.state.z, count, path);
        read_block(file, snapshot.state.vx, count, path);
        read_block(file, snapshot.state.vy, count, path);
        read_block(file, snapshot.state.vz, count, path);
        read_block(file, snapshot.state.mass, count, path);
    } catch (...) {
        std::fclose(file);
        throw;
    }
    std::fclose(file);
    return snapshot;
}

fs::path snapshot_path_for_step(const fs::path& directory, uint32_t step) {
    char name[64];
    std::snprintf(name, sizeof(name), "snap_%09u.bin", step);
    return directory / name;
}

fs::path latest_snapshot(const fs::path& directory) {
    if (!fs::exists(directory)) {
        return {};
    }
    static const std::regex pattern(R"(snap_(\d+)\.bin)");
    fs::path best;
    long long best_step = -1;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        const long long step = std::stoll(match[1].str());
        if (step > best_step) {
            best_step = step;
            best = entry.path();
        }
    }
    return best;
}

}  // namespace cps
