#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/simulation_engine.hpp"
#include "solvers/nbody_solver.hpp"
#include "utils/cuda_helpers.hpp"
#include "utils/progress.hpp"

#ifndef CPS_GIT_COMMIT
#define CPS_GIT_COMMIT "unknown"
#endif

namespace {

namespace fs = std::filesystem;

// Timing protocol.
//
// Three things ruin a GPU timing number: the first launch of a kernel, which
// pays for module load and clock ramp; the host queueing work faster than the
// device retires it, which measures the host; and a card that has not settled
// at its sustained clock. So each measurement warms up, then times a run of
// steps long enough to be worth measuring with CUDA events on the same stream
// as the work, and repeats that whole run several times and keeps the fastest.
// The fastest rather than the mean because everything that perturbs a
// measurement makes it slower, never faster, so the minimum is the estimate
// least contaminated by whatever else the machine was doing.
constexpr int WARMUP_STEPS = 5;
constexpr int TRIALS = 3;
constexpr double TARGET_TRIAL_MS = 120.0;
constexpr int MIN_TIMED_STEPS = 3;
constexpr int MAX_TIMED_STEPS = 400;

// The conventional operation count for one softened gravitational interaction,
// as used by Nyland, Harris and Prins in GPU Gems 3. Counting it any other way
// would make the numbers here incomparable with every other N-body result in
// the literature, so the convention is followed and stated rather than
// recounted.
constexpr double FLOPS_PER_INTERACTION = 20.0;

struct BenchmarkRow {
    std::string solver;
    std::string variant;
    int num_particles = 0;
    int block_size = 0;
    int tile_size = 0;
    int timed_steps = 0;
    double ms_per_step = 0.0;
    double steps_per_second = 0.0;
    double interactions_per_second = 0.0;
    double gflops = 0.0;
    double model_bytes_per_step = 0.0;
    double model_gb_s = 0.0;
    double model_bandwidth_fraction = 0.0;
    int fast_math = 0;
    uint32_t seed = 0;
    std::string gpu_name;
    int cuda_runtime = 0;
    int cuda_driver = 0;
    std::string commit;
    std::string timestamp;
};

const char* CSV_HEADER =
    "solver,variant,num_particles,block_size,tile_size,timed_steps,ms_per_step,steps_per_second,"
    "interactions_per_second,gflops,model_bytes_per_step,model_gb_s,model_bandwidth_fraction,"
    "fast_math,seed,gpu_name,cuda_runtime,cuda_driver,commit,timestamp";

// Compulsory global memory traffic for one step, counted from the algorithm
// rather than measured.
//
// This is the traffic the step cannot avoid: what it must read from and write
// to global memory at least once. It deliberately excludes the tile reads,
// which are the whole point of the tiled kernel and which the caches and the
// shared memory staging are there to absorb. Quoted as a model, and the
// profiler numbers in the debug report are what check it.
double model_bytes_per_step(const cps::Config& config) {
    const double n = static_cast<double>(config.num_particles);
    if (config.solver == cps::SolverType::Sph) {
        // hash, sort, bounds, gather, density and force passes, counted as the
        // arrays each one touches once.
        return n * 4.0 * 44.0;
    }
    if (config.variant == cps::NBodyVariant::Fused) {
        // Reads x, y, z, mass, and the three velocities; writes three
        // positions, three velocities and three accelerations.
        return n * 4.0 * (7.0 + 9.0);
    }
    // Split path: the force kernel reads four arrays and writes three, the
    // drift kernel reads nine and writes six, the closing kick reads six and
    // writes three.
    return n * 4.0 * ((4.0 + 3.0) + (9.0 + 6.0) + (6.0 + 3.0));
}

std::string now_iso8601() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_value{};
#if defined(_WIN32)
    gmtime_s(&tm_value, &t);
#else
    gmtime_r(&t, &tm_value);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm_value);
    return buffer;
}

cps::Config nbody_config(cps::NBodyVariant variant, int count) {
    cps::Config config;
    config.name = "bench_nbody";
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.steps = 0;
    config.dt = 1.0e-3f;
    config.softening = 0.01f;
    config.gravitational_constant = 1.0f;
    config.total_mass = 1.0f;
    config.plummer_radius = 1.0f;
    config.seed = 20260802u;
    config.block_size = 256;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    return config;
}

cps::Config sph_config(int count) {
    cps::Config config;
    config.name = "bench_sph";
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = count;
    config.steps = 0;
    config.dt = 1.0e-4f;
    config.seed = 20260802u;
    config.block_size = 256;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    config.sph.rest_density = 1000.0f;
    config.sph.gas_constant = 1000.0f;
    config.sph.viscosity = 0.2f;
    config.sph.boundary_damping = 0.4f;
    return config;
}

// One measurement. Returns milliseconds per step.
double measure(const cps::Config& config, int& timed_steps_out) {
    cps::SimulationEngine engine(config);
    engine.initialize();

    for (int i = 0; i < WARMUP_STEPS; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calibrate the run length so every point costs about the same wall time,
    // rather than timing a fixed number of steps that is far too many at two
    // hundred thousand particles and far too few at ten thousand.
    cps::CudaTimer probe;
    probe.start();
    engine.step();
    const double single_ms = static_cast<double>(probe.stop());

    int timed_steps = MIN_TIMED_STEPS;
    if (single_ms > 0.0) {
        timed_steps = static_cast<int>(std::lround(TARGET_TRIAL_MS / single_ms));
    }
    timed_steps = std::clamp(timed_steps, MIN_TIMED_STEPS, MAX_TIMED_STEPS);
    timed_steps_out = timed_steps;

    double best_ms = 0.0;
    for (int trial = 0; trial < TRIALS; ++trial) {
        cps::CudaTimer timer;
        timer.start();
        for (int step = 0; step < timed_steps; ++step) {
            engine.step();
        }
        const double elapsed = static_cast<double>(timer.stop());
        if (trial == 0 || elapsed < best_ms) {
            best_ms = elapsed;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return best_ms / static_cast<double>(timed_steps);
}

BenchmarkRow make_row(const cps::Config& config, double ms_per_step, int timed_steps,
                      const cps::DeviceInfo& info) {
    BenchmarkRow row;
    row.solver = cps::to_string(config.solver);
    row.variant =
        config.solver == cps::SolverType::NBody ? cps::to_string(config.variant) : std::string("na");
    row.num_particles = config.num_particles;
    row.block_size = config.block_size;
    row.tile_size = config.variant == cps::NBodyVariant::Naive ? 0 : cps::DEFAULT_TILE_SIZE;
    if (config.solver == cps::SolverType::Sph) {
        row.tile_size = 0;
    }
    row.timed_steps = timed_steps;
    row.ms_per_step = ms_per_step;
    row.steps_per_second = ms_per_step > 0.0 ? 1000.0 / ms_per_step : 0.0;

    const double n = static_cast<double>(config.num_particles);
    const double seconds = ms_per_step * 1.0e-3;
    if (config.solver == cps::SolverType::NBody) {
        const double interactions = n * n;
        row.interactions_per_second = seconds > 0.0 ? interactions / seconds : 0.0;
        row.gflops = row.interactions_per_second * FLOPS_PER_INTERACTION / 1.0e9;
    } else {
        row.interactions_per_second = seconds > 0.0 ? n / seconds : 0.0;
        row.gflops = 0.0;
    }

    row.model_bytes_per_step = model_bytes_per_step(config);
    row.model_gb_s = seconds > 0.0 ? row.model_bytes_per_step / seconds / 1.0e9 : 0.0;
    row.model_bandwidth_fraction =
        info.peak_bandwidth_gb_s > 0.0 ? row.model_gb_s / info.peak_bandwidth_gb_s : 0.0;

    row.fast_math = CPS_FAST_MATH;
    row.seed = config.seed;
    row.gpu_name = info.name;
    row.cuda_runtime = info.runtime_version;
    row.cuda_driver = info.driver_version;
    row.commit = CPS_GIT_COMMIT;
    row.timestamp = now_iso8601();
    return row;
}

std::string row_key(const std::string& solver, const std::string& variant, int count,
                    int block_size, int tile_size, int fast_math) {
    std::ostringstream out;
    out << solver << '|' << variant << '|' << count << '|' << block_size << '|' << tile_size << '|'
        << fast_math;
    return out.str();
}

// Reads the keys already present so an interrupted sweep can be restarted
// without repeating the expensive points it already has.
std::set<std::string> read_existing_keys(const fs::path& path) {
    std::set<std::string> keys;
    std::ifstream stream(path);
    if (!stream) {
        return keys;
    }
    std::string line;
    bool first = true;
    while (std::getline(stream, line)) {
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::string field;
        std::istringstream parts(line);
        while (std::getline(parts, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < 14) {
            continue;
        }
        keys.insert(row_key(fields[0], fields[1], std::stoi(fields[2]), std::stoi(fields[3]),
                            std::stoi(fields[4]), std::stoi(fields[13])));
    }
    return keys;
}

void append_row(const fs::path& path, const BenchmarkRow& row, bool write_header) {
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("cannot append to " + path.string());
    }
    if (write_header) {
        stream << CSV_HEADER << "\n";
    }
    stream << row.solver << ',' << row.variant << ',' << row.num_particles << ',' << row.block_size
           << ',' << row.tile_size << ',' << row.timed_steps << ',' << row.ms_per_step << ','
           << row.steps_per_second << ',' << row.interactions_per_second << ',' << row.gflops << ','
           << row.model_bytes_per_step << ',' << row.model_gb_s << ','
           << row.model_bandwidth_fraction << ',' << row.fast_math << ',' << row.seed << ','
           << row.gpu_name << ',' << row.cuda_runtime << ',' << row.cuda_driver << ',' << row.commit
           << ',' << row.timestamp << "\n";
}

struct PlannedPoint {
    cps::Config config;
    std::string solver;
    std::string variant;
    int tile_size = 0;
    // Relative cost, used only to project the total up front.
    double weight = 0.0;
};

void print_usage() {
    std::printf(
        "bench_all: sweep the kernel ladder and record the results\n"
        "\n"
        "options:\n"
        "  --output <path>   results CSV (default experiments/results/benchmarks.csv)\n"
        "  --force           remeasure points that are already in the CSV\n"
        "  --quick           a short sweep for smoke testing the driver itself\n"
        "  --tile-sweep      also sweep the tile width at a fixed particle count\n"
        "  --help            print this message\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path output = fs::path(CPS_PROJECT_ROOT) / "experiments" / "results" / "benchmarks.csv";
        bool force = false;
        bool quick = false;
        bool tile_sweep = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                output = argv[++i];
            } else if (arg == "--force") {
                force = true;
            } else if (arg == "--quick") {
                quick = true;
            } else if (arg == "--tile-sweep") {
                tile_sweep = true;
            } else if (arg == "--help") {
                print_usage();
                return 0;
            } else {
                std::fprintf(stderr, "unknown option %s\n", arg.c_str());
                print_usage();
                return 2;
            }
        }

        const cps::DeviceInfo info = cps::query_device(0);
        cps::print_device_info(info, stdout);
        std::printf("\nbuild: commit %s, fast math %s\n", CPS_GIT_COMMIT,
                    CPS_FAST_MATH ? "on" : "off");

        const std::vector<int> nbody_counts =
            quick ? std::vector<int>{4096, 8192}
                  : std::vector<int>{10000, 20000, 50000, 100000, 150000, 200000};
        const std::vector<int> sph_counts = quick ? std::vector<int>{4096, 8192}
                                                  : std::vector<int>{10000, 25000, 50000, 75000,
                                                                     100000};

        std::vector<PlannedPoint> plan;
        for (cps::NBodyVariant variant :
             {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
            for (int count : nbody_counts) {
                PlannedPoint point;
                point.config = nbody_config(variant, count);
                point.solver = "nbody";
                point.variant = cps::to_string(variant);
                point.tile_size = variant == cps::NBodyVariant::Naive ? 0 : cps::DEFAULT_TILE_SIZE;
                point.weight = static_cast<double>(count) * count;
                plan.push_back(point);
            }
        }
        for (int count : sph_counts) {
            PlannedPoint point;
            point.config = sph_config(count);
            point.solver = "sph";
            point.variant = "na";
            point.tile_size = 0;
            // The fluid step is close to linear in the particle count, and far
            // cheaper per particle than the all pairs sum.
            point.weight = static_cast<double>(count) * 3000.0;
            plan.push_back(point);
        }

        const std::set<std::string> existing = force ? std::set<std::string>{}
                                                     : read_existing_keys(output);

        std::vector<PlannedPoint> todo;
        double total_weight = 0.0;
        for (const PlannedPoint& point : plan) {
            const std::string key = row_key(point.solver, point.variant, point.config.num_particles,
                                            point.config.block_size, point.tile_size, CPS_FAST_MATH);
            if (existing.count(key) > 0) {
                continue;
            }
            todo.push_back(point);
            total_weight += point.weight;
        }

        std::printf("\nplanned points: %zu of %zu (%zu already present in %s)\n", todo.size(),
                    plan.size(), plan.size() - todo.size(), output.string().c_str());

        if (todo.empty() && !tile_sweep) {
            std::printf("nothing to do\n");
            return 0;
        }

        // Calibrate against one cheap point so the projection up front is
        // based on this machine rather than on a guess.
        double seconds_per_unit_weight = 0.0;
        if (!todo.empty()) {
            cps::Config probe = nbody_config(cps::NBodyVariant::Tiled, 8192);
            int probe_steps = 0;
            const double probe_ms = measure(probe, probe_steps);
            const double probe_weight = 8192.0 * 8192.0;
            // Each point costs the warm up, the calibration step, and three
            // timed runs of about the target duration.
            seconds_per_unit_weight = (probe_ms * 1.0e-3) / probe_weight;
            const double projected =
                total_weight * seconds_per_unit_weight * (WARMUP_STEPS + 1) +
                static_cast<double>(todo.size()) * TRIALS * TARGET_TRIAL_MS * 1.0e-3;
            std::printf("calibration: %.4f ms/step at 8192 tiled particles\n", probe_ms);
            std::printf("projected sweep total: about %s\n",
                        cps::format_duration(projected).c_str());
        }
        std::fflush(stdout);

        fs::create_directories(output.parent_path());
        bool need_header = !fs::exists(output) || fs::file_size(output) == 0;

        cps::ProgressBar bar("benchmark sweep", static_cast<long long>(todo.size()));
        long long done = 0;

        for (const PlannedPoint& point : todo) {
            int timed_steps = 0;
            const double ms = measure(point.config, timed_steps);
            const BenchmarkRow row = make_row(point.config, ms, timed_steps, info);
            append_row(output, row, need_header);
            need_header = false;

            std::printf("  %-5s %-6s N=%-7d %9.4f ms/step  %8.1f steps/s", row.solver.c_str(),
                        row.variant.c_str(), row.num_particles, row.ms_per_step,
                        row.steps_per_second);
            if (point.solver == "nbody") {
                std::printf("  %8.1f GFLOP/s", row.gflops);
            } else {
                std::printf("  %8.2f Mparticle/s", row.interactions_per_second / 1.0e6);
            }
            std::printf("\n");
            std::fflush(stdout);

            bar.update(++done);
        }
        bar.finish();

        // The tile width sweep is a diagnostic rather than part of the ladder.
        // The specification fixes the tile at thirty two; whether that is the
        // best choice on this card is an empirical question, and the answer
        // belongs in the debug report with numbers behind it.
        if (tile_sweep) {
            const fs::path tile_output = output.parent_path() / "tile_sweep.csv";
            // The header is written through the same append path as the rows.
            // Holding a separate truncating stream open here while append_row
            // opens the file independently corrupts the result: the truncating
            // stream flushes its buffer at offset zero when it is destroyed,
            // overwriting rows that were appended in the meantime.
            std::error_code remove_error;
            fs::remove(tile_output, remove_error);
            bool tile_header_needed = true;
            std::printf("\ntile width sweep at 100000 particles\n");
            for (int tile : {32, 64, 128, 256}) {
                for (cps::NBodyVariant variant :
                     {cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
                    cps::Config config = nbody_config(variant, 100000);
                    // The engine uses the default tile width, so the sweep
                    // drives the kernels directly through a dedicated harness
                    // rather than through the engine.
                    cps::SimulationEngine engine(config);
                    engine.initialize();

                    cps::ParticleSystem& particles = engine.particles();
                    const int count = config.num_particles;

                    auto run_once = [&]() {
                        if (variant == cps::NBodyVariant::Tiled) {
                            cps::compute_forces_tiled(
                                particles.x(), particles.y(), particles.z(), particles.mass(),
                                particles.ax(), particles.ay(), particles.az(), count,
                                config.softening, config.gravitational_constant, config.block_size,
                                tile);
                        } else {
                            cps::nbody_fused_step(
                                particles.x(), particles.y(), particles.z(), particles.x_next(),
                                particles.y_next(), particles.z_next(), particles.vx(),
                                particles.vy(), particles.vz(), particles.ax(), particles.ay(),
                                particles.az(), particles.mass(), count, config.softening,
                                config.gravitational_constant, config.dt, config.block_size, tile);
                            particles.swap_position_buffers();
                        }
                    };

                    for (int i = 0; i < WARMUP_STEPS; ++i) {
                        run_once();
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    const int steps = 20;
                    double best = 0.0;
                    for (int trial = 0; trial < TRIALS; ++trial) {
                        cps::CudaTimer timer;
                        timer.start();
                        for (int i = 0; i < steps; ++i) {
                            run_once();
                        }
                        const double elapsed = static_cast<double>(timer.stop());
                        if (trial == 0 || elapsed < best) {
                            best = elapsed;
                        }
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    const double ms = best / steps;
                    BenchmarkRow row = make_row(config, ms, steps, info);
                    row.tile_size = tile;
                    append_row(tile_output, row, tile_header_needed);
                    tile_header_needed = false;
                    std::printf("  tile %3d  %-6s  %9.4f ms/step  %8.1f GFLOP/s\n", tile,
                                cps::to_string(variant).c_str(), ms, row.gflops);
                    std::fflush(stdout);
                }
            }
            std::printf("wrote %s\n", tile_output.string().c_str());
        }

        std::printf("\nwrote %s\n", output.string().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
