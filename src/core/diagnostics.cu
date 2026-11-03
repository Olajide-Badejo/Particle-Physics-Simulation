#include "cps/core/diagnostics.hpp"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "cps/physics/gravity.hpp"

namespace cps {
namespace {

constexpr int DIAG_BLOCK = 256;
constexpr int DIAG_TILE = 128;

// Block level reduction of four doubles. The tree is a fixed shape and the
// partials are summed on the host in block order afterwards, so the whole
// reduction is deterministic. An atomic accumulation would have been shorter
// and would have made two runs of the same configuration disagree in the last
// few bits, which is precisely what the bit identity test forbids.
__device__ void block_reduce4(double* shared, int lane, double& a, double& b, double& c,
                              double& d) {
    shared[lane] = a;
    shared[DIAG_BLOCK + lane] = b;
    shared[2 * DIAG_BLOCK + lane] = c;
    shared[3 * DIAG_BLOCK + lane] = d;
    __syncthreads();
    for (int stride = DIAG_BLOCK / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
            shared[DIAG_BLOCK + lane] += shared[DIAG_BLOCK + lane + stride];
            shared[2 * DIAG_BLOCK + lane] += shared[2 * DIAG_BLOCK + lane + stride];
            shared[3 * DIAG_BLOCK + lane] += shared[3 * DIAG_BLOCK + lane + stride];
        }
        __syncthreads();
    }
    a = shared[0];
    b = shared[DIAG_BLOCK];
    c = shared[2 * DIAG_BLOCK];
    d = shared[3 * DIAG_BLOCK];
}

__global__ void kinetic_momentum_kernel(const float* __restrict__ vx, const float* __restrict__ vy,
                                        const float* __restrict__ vz,
                                        const float* __restrict__ ax, const float* __restrict__ ay,
                                        const float* __restrict__ az,
                                        const float* __restrict__ mass, int count, float half_dt,
                                        double* __restrict__ partials) {
    __shared__ double shared[4 * DIAG_BLOCK];

    const int lane = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);

    double kinetic = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;

    for (int i = blockIdx.x * blockDim.x + lane; i < count; i += stride) {
        const double m = static_cast<double>(mass[i]);
        // half_dt is zero when the velocity is already synchronised with the
        // positions, so the common path costs one fused multiply add.
        const double ux = static_cast<double>(vx[i]) + static_cast<double>(half_dt) * ax[i];
        const double uy = static_cast<double>(vy[i]) + static_cast<double>(half_dt) * ay[i];
        const double uz = static_cast<double>(vz[i]) + static_cast<double>(half_dt) * az[i];
        kinetic += 0.5 * m * (ux * ux + uy * uy + uz * uz);
        px += m * ux;
        py += m * uy;
        pz += m * uz;
    }

    block_reduce4(shared, lane, kinetic, px, py, pz);

    if (lane == 0) {
        partials[4 * blockIdx.x + 0] = kinetic;
        partials[4 * blockIdx.x + 1] = px;
        partials[4 * blockIdx.x + 2] = py;
        partials[4 * blockIdx.x + 3] = pz;
    }
}

// All pairs softened potential. Every unordered pair is visited twice, once
// from each end, so the caller halves the result. Self interaction is excluded
// explicitly: with softening it would contribute a finite and entirely
// spurious self energy of G m^2 / eps per particle.
__global__ void potential_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                 const float* __restrict__ z, const float* __restrict__ mass,
                                 int count, float softening_sq, double* __restrict__ partials) {
    __shared__ float tile_x[DIAG_TILE];
    __shared__ float tile_y[DIAG_TILE];
    __shared__ float tile_z[DIAG_TILE];
    __shared__ float tile_mass[DIAG_TILE];
    __shared__ double shared[4 * DIAG_BLOCK];

    const int lane = static_cast<int>(threadIdx.x);
    const int i = blockIdx.x * blockDim.x + lane;
    const bool active = i < count;

    const float xi = active ? x[i] : 0.0f;
    const float yi = active ? y[i] : 0.0f;
    const float zi = active ? z[i] : 0.0f;
    const float mi = active ? mass[i] : 0.0f;

    double potential = 0.0;

    for (int tile_start = 0; tile_start < count; tile_start += DIAG_TILE) {
        __syncthreads();
        for (int t = lane; t < DIAG_TILE; t += static_cast<int>(blockDim.x)) {
            const int source = tile_start + t;
            if (source < count) {
                tile_x[t] = x[source];
                tile_y[t] = y[source];
                tile_z[t] = z[source];
                tile_mass[t] = mass[source];
            } else {
                tile_x[t] = 0.0f;
                tile_y[t] = 0.0f;
                tile_z[t] = 0.0f;
                tile_mass[t] = 0.0f;
            }
        }
        __syncthreads();

        if (active) {
            for (int t = 0; t < DIAG_TILE; ++t) {
                const int j = tile_start + t;
                if (j >= count || j == i) {
                    continue;
                }
                potential += pair_potential(xi, yi, zi, tile_x[t], tile_y[t], tile_z[t], mi,
                                            tile_mass[t], softening_sq);
            }
        }
    }

    double unused_b = 0.0;
    double unused_c = 0.0;
    double unused_d = 0.0;
    block_reduce4(shared, lane, potential, unused_b, unused_c, unused_d);

    if (lane == 0) {
        partials[blockIdx.x] = potential;
    }
}

}  // namespace

DiagnosticsReducer::DiagnosticsReducer(int count, int block_size)
    : count_(count), block_size_(block_size) {
    if (count <= 0) {
        throw std::runtime_error("the diagnostics reducer needs at least one particle");
    }
    // The grid is capped so the partial buffer stays small; the kernel loops
    // with a grid stride to cover any particle count.
    partial_blocks_ = grid_for(count, DIAG_BLOCK);
    const int capped = partial_blocks_ > 4096 ? 4096 : partial_blocks_;
    partial_blocks_ = capped;
    partials_.allocate(static_cast<std::size_t>(4 * partial_blocks_));
    partials1_.allocate(static_cast<std::size_t>(grid_for(count, DIAG_BLOCK)));
}

void DiagnosticsReducer::reduce_kinetic_and_momentum(const ParticleSystem& particles,
                                                     double& kinetic, double& px, double& py,
                                                     double& pz, float half_dt) {
    const int grid = partial_blocks_;
    kinetic_momentum_kernel<<<grid, DIAG_BLOCK>>>(
        particles.vx(), particles.vy(), particles.vz(), particles.ax(), particles.ay(),
        particles.az(), particles.mass(), count_, half_dt, partials_.get());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> host(static_cast<std::size_t>(4 * grid));
    CUDA_CHECK(cudaMemcpy(host.data(), partials_.get(), host.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

    kinetic = 0.0;
    px = 0.0;
    py = 0.0;
    pz = 0.0;
    for (int b = 0; b < grid; ++b) {
        kinetic += host[static_cast<std::size_t>(4 * b + 0)];
        px += host[static_cast<std::size_t>(4 * b + 1)];
        py += host[static_cast<std::size_t>(4 * b + 2)];
        pz += host[static_cast<std::size_t>(4 * b + 3)];
    }
}

double DiagnosticsReducer::reduce_potential(const ParticleSystem& particles, float softening,
                                            float gravitational_constant) {
    const int grid = grid_for(count_, DIAG_BLOCK);
    potential_kernel<<<grid, DIAG_BLOCK>>>(particles.x(), particles.y(), particles.z(),
                                           particles.mass(), count_, softening * softening,
                                           partials1_.get());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> host(static_cast<std::size_t>(grid));
    CUDA_CHECK(cudaMemcpy(host.data(), partials1_.get(), host.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int b = 0; b < grid; ++b) {
        sum += host[static_cast<std::size_t>(b)];
    }
    return 0.5 * static_cast<double>(gravitational_constant) * sum;
}

Diagnostics DiagnosticsReducer::compute(const ParticleSystem& particles, long long step,
                                        double time, float softening,
                                        float gravitational_constant, bool include_potential,
                                        float half_dt) {
    Diagnostics row;
    row.step = step;
    row.time = time;
    reduce_kinetic_and_momentum(particles, row.kinetic, row.px, row.py, row.pz, half_dt);
    row.potential =
        include_potential ? reduce_potential(particles, softening, gravitational_constant) : 0.0;
    row.total = row.kinetic + row.potential;
    row.momentum_magnitude = std::sqrt(row.px * row.px + row.py * row.py + row.pz * row.pz);
    return row;
}

DiagnosticsWriter::DiagnosticsWriter(const std::string& path) {
    if (path.empty()) {
        return;
    }
    file_ = std::fopen(path.c_str(), "w");
    if (file_ == nullptr) {
        throw std::runtime_error("cannot open diagnostics file " + path);
    }
    std::fprintf(file_, "%s\n", DIAGNOSTICS_HEADER);
}

DiagnosticsWriter::DiagnosticsWriter(const std::string& path, long long resume_step) {
    if (path.empty()) {
        return;
    }

    std::string text;
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::ostringstream buffer;
            buffer << in.rdbuf();
            text = buffer.str();
        }
    }

    if (text.empty()) {
        file_ = std::fopen(path.c_str(), "w");
        if (file_ == nullptr) {
            throw std::runtime_error("cannot open diagnostics file " + path);
        }
        std::fprintf(file_, "%s\n", DIAGNOSTICS_HEADER);
        return;
    }

    // Split into complete lines. Whatever follows the last newline is a row
    // a crash cut short, and is dropped with the rows the resume rewrites.
    std::vector<std::string> lines;
    std::size_t start = 0;
    bool dropped = false;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            // A header cut short of its newline is still a header.
            if (lines.empty() && text.compare(start, std::string::npos, DIAGNOSTICS_HEADER) == 0) {
                lines.emplace_back(DIAGNOSTICS_HEADER);
            }
            dropped = true;
            break;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }

    if (lines.empty() || lines.front() != DIAGNOSTICS_HEADER) {
        const std::string first = lines.empty() ? text.substr(0, 80) : lines.front();
        throw std::runtime_error("diagnostics file " + path + " starts with \"" + first +
                                 "\" rather than the expected header \"" + DIAGNOSTICS_HEADER +
                                 "\", so a resumed run cannot append to it; move it aside or " +
                                 "choose another diagnostics path");
    }

    std::string kept = lines.front() + "\n";
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const std::string& line = lines[i];
        long long step = 0;
        const char* first = line.data();
        const char* last = first + line.size();
        const auto [end, error] = std::from_chars(first, last, step);
        if (error != std::errc() || end == last || *end != ',') {
            throw std::runtime_error("diagnostics file " + path + " has a row at line " +
                                     std::to_string(i + 1) + " whose step cannot be read: \"" +
                                     line.substr(0, 80) + "\"");
        }
        if (step >= resume_step) {
            dropped = true;
            continue;
        }
        kept += line;
        kept += '\n';
    }

    if (dropped) {
        const std::string temporary = path + ".tmp";
        std::FILE* out = std::fopen(temporary.c_str(), "wb");
        if (out == nullptr) {
            throw std::runtime_error("cannot write diagnostics file " + temporary);
        }
        const bool written = std::fwrite(kept.data(), 1, kept.size(), out) == kept.size();
        const bool closed = std::fclose(out) == 0;
        std::error_code ignored;
        if (!written || !closed) {
            std::filesystem::remove(temporary, ignored);
            throw std::runtime_error("cannot write diagnostics file " + temporary);
        }
        std::error_code renamed;
        std::filesystem::rename(temporary, path, renamed);
        if (renamed) {
            std::filesystem::remove(temporary, ignored);
            throw std::runtime_error("cannot replace diagnostics file " + path + ": " +
                                     renamed.message());
        }
    }

    file_ = std::fopen(path.c_str(), "a");
    if (file_ == nullptr) {
        throw std::runtime_error("cannot open diagnostics file " + path);
    }
}

DiagnosticsWriter::~DiagnosticsWriter() { close(); }

void DiagnosticsWriter::write(const Diagnostics& row) {
    if (file_ == nullptr) {
        return;
    }
    // Seventeen significant digits round trips a double exactly, which matters
    // because the drift gate is evaluated from this file rather than from the
    // process that produced it.
    std::fprintf(file_, "%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", row.step,
                 row.time, row.kinetic, row.potential, row.total, row.px, row.py, row.pz,
                 row.momentum_magnitude);
}

void DiagnosticsWriter::flush() {
    if (file_ != nullptr) {
        std::fflush(file_);
    }
}

void DiagnosticsWriter::close() {
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

}  // namespace cps
