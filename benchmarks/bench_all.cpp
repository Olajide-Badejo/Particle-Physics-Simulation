#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cps/core/bench_rows.hpp"
#include "cps/core/initial_conditions.hpp"
#include "cps/core/simulation_engine.hpp"
#include "cps/version.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "cps/utils/cuda_helpers.hpp"
#include "cps/utils/progress.hpp"

namespace {

namespace fs = std::filesystem;
namespace bench = cps::bench;

// Timing protocol.
//
// Three things ruin a GPU timing number: the first launch of a kernel, which
// pays for module load and clock ramp; the host queueing work faster than the
// device retires it, which measures the host; and a card that has not settled
// at its sustained clock. So each configuration warms up, then times a run of
// steps long enough to be worth measuring with CUDA events on the same stream
// as the work, and repeats that run for a number of trials.
//
// A fourth thing ruins a comparison: session drift. The same two binaries have
// measured several percent apart when run in separate blocks and agreed when
// interleaved, and more trials do not remove that, because the drift is
// confounded with which variant ran when. So the configurations
// being compared are measured in rounds: every configuration's first trial,
// then every configuration's second trial, and so on. Each engine stays alive
// between its trials. The order inside a round rotates, so no configuration
// always follows the same neighbour.
//
// A group is the set of configurations compared this way: the three variants at
// one particle count, the fluid sizes, or the tile widths. Rows are written when
// a group finishes, so an interrupted sweep restarts at the group it lost.
//
// The row reports the median of the trials, which a single interrupted trial
// cannot move, with the dispersion alongside so a reader can tell a difference
// from noise.
constexpr int WARMUP_STEPS = 20;
constexpr int TRIALS = 10;
constexpr double TARGET_TRIAL_MS = 120.0;
constexpr int MIN_TIMED_STEPS = 20;
constexpr int MAX_TIMED_STEPS = 400;

struct Protocol {
    // Recorded on every row. check_provenance.py accepts only "full".
    const char* name;
    int trials;
    int warmup_steps;
    int min_timed_steps;
    double target_trial_ms;
};

constexpr Protocol FULL_PROTOCOL{"full", TRIALS, WARMUP_STEPS, MIN_TIMED_STEPS, TARGET_TRIAL_MS};
// Exercises every path of the driver in seconds. Its numbers mean nothing.
constexpr Protocol QUICK_PROTOCOL{"quick", 3, 5, 3, 20.0};

struct BenchmarkRow {
    std::string solver;
    std::string variant;
    int num_particles = 0;
    int block_size = 0;
    int tile_size = 0;
    int timed_steps = 0;
    double ms_per_step = 0.0;
    int trials = 0;
    double ms_min = 0.0;
    double ms_median = 0.0;
    double ms_mean = 0.0;
    double ms_stddev = 0.0;
    double ms_iqr = 0.0;
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
    int dirty = 0;
    std::string protocol;
    std::string timestamp;
};

const char* CSV_HEADER =
    "solver,variant,num_particles,block_size,tile_size,timed_steps,ms_per_step,trials,ms_min,"
    "ms_median,ms_mean,ms_stddev,ms_iqr,steps_per_second,interactions_per_second,gflops,"
    "model_bytes_per_step,model_gb_s,model_bandwidth_fraction,fast_math,seed,gpu_name,"
    "cuda_runtime,cuda_driver,commit,dirty,protocol,timestamp";

const char* TRIALS_HEADER =
    "solver,variant,num_particles,block_size,tile_size,fast_math,round,timed_steps,ms_per_step,"
    "commit,dirty,protocol,timestamp";

// A tree the hash cannot describe is treated the same as one with local edits.
bool tree_is_dirty() {
    return CPS_GIT_DIRTY != 0 || std::strcmp(CPS_GIT_HASH, "unknown") == 0;
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

std::string variant_name(const cps::Config& config) {
    return config.solver == cps::SolverType::NBody ? cps::to_string(config.variant)
                                                   : std::string("na");
}

int default_tile_size(const cps::Config& config) {
    if (config.solver == cps::SolverType::Sph || config.variant == cps::NBodyVariant::Naive) {
        return 0;
    }
    return cps::DEFAULT_TILE_SIZE;
}

// One configuration under measurement, with the engine that stays alive across
// its rounds.
struct Contender {
    cps::Config config;
    int tile_size = 0;
    std::unique_ptr<cps::SimulationEngine> engine;
    std::function<void()> step;
    double single_step_ms = 0.0;
    int timed_steps = 0;
    std::vector<double> trial_ms;
};

Contender make_contender(const cps::Config& config) {
    Contender contender;
    contender.config = config;
    contender.tile_size = default_tile_size(config);
    contender.engine = std::make_unique<cps::SimulationEngine>(config);
    contender.engine->initialize();
    cps::SimulationEngine* engine = contender.engine.get();
    contender.step = [engine]() { engine->step(); };
    return contender;
}

// The steps a trial runs, for a step that costs step_ms. Calibrating the run
// length makes every point cost about the same wall time, rather than timing a
// fixed number of steps that is far too many at two hundred thousand particles
// and far too few at ten thousand.
int steps_per_trial(double step_ms, const Protocol& protocol) {
    int steps = protocol.min_timed_steps;
    if (step_ms > 0.0) {
        steps = static_cast<int>(std::lround(protocol.target_trial_ms / step_ms));
    }
    return std::clamp(steps, protocol.min_timed_steps, MAX_TIMED_STEPS);
}

void warm_up_and_calibrate(Contender& contender, const Protocol& protocol) {
    for (int i = 0; i < protocol.warmup_steps; ++i) {
        contender.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::CudaTimer probe;
    probe.start();
    contender.step();
    contender.single_step_ms = static_cast<double>(probe.stop());
    contender.timed_steps = steps_per_trial(contender.single_step_ms, protocol);
}

// Measures a group in rounds. See the protocol above.
void measure_group(std::vector<Contender>& group, const Protocol& protocol) {
    for (Contender& contender : group) {
        warm_up_and_calibrate(contender, protocol);
    }
    const std::size_t size = group.size();
    for (int round = 0; round < protocol.trials; ++round) {
        for (std::size_t position = 0; position < size; ++position) {
            Contender& contender = group[(position + static_cast<std::size_t>(round)) % size];
            cps::CudaTimer timer;
            timer.start();
            for (int step = 0; step < contender.timed_steps; ++step) {
                contender.step();
            }
            const double elapsed = static_cast<double>(timer.stop());
            contender.trial_ms.push_back(elapsed / static_cast<double>(contender.timed_steps));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // A timing of a blown-up fluid would still look like a timing.
    for (Contender& contender : group) {
        contender.engine->check_device_errors();
    }
}

BenchmarkRow make_row(const Contender& contender, const cps::DeviceInfo& info,
                      const Protocol& protocol) {
    const cps::Config& config = contender.config;
    const std::vector<double>& trials = contender.trial_ms;

    BenchmarkRow row;
    row.solver = cps::to_string(config.solver);
    row.variant = variant_name(config);
    row.num_particles = config.num_particles;
    row.block_size = config.block_size;
    row.tile_size = contender.tile_size;
    row.timed_steps = contender.timed_steps;

    row.trials = static_cast<int>(trials.size());
    row.ms_min = *std::min_element(trials.begin(), trials.end());
    row.ms_median = bench::median(trials);
    row.ms_mean = bench::mean(trials);
    row.ms_stddev = bench::standard_deviation(trials);
    row.ms_iqr = bench::interquartile_range(trials);

    const double ms = row.ms_median;
    row.ms_per_step = ms;
    row.steps_per_second = bench::steps_per_second(ms);
    row.interactions_per_second =
        bench::interactions_per_second(config.solver, config.num_particles, ms);
    row.gflops = bench::gflops(config.solver, config.num_particles, ms);
    row.model_bytes_per_step =
        bench::model_bytes_per_step(config.solver, config.variant, config.num_particles);
    row.model_gb_s = bench::model_gb_s(row.model_bytes_per_step, ms);
    row.model_bandwidth_fraction =
        bench::model_bandwidth_fraction(row.model_gb_s, info.peak_bandwidth_gb_s);

    row.fast_math = CPS_FAST_MATH;
    row.seed = config.seed;
    row.gpu_name = info.name;
    row.cuda_runtime = info.runtime_version;
    row.cuda_driver = info.driver_version;
    row.commit = CPS_GIT_STAMP;
    row.dirty = tree_is_dirty() ? 1 : 0;
    row.protocol = protocol.name;
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

std::string config_key(const cps::Config& config) {
    return row_key(cps::to_string(config.solver), variant_name(config), config.num_particles,
                   config.block_size, default_tile_size(config), CPS_FAST_MATH);
}

// Splits one CSV line. The driver never writes a quoted field.
std::vector<std::string> split_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream parts(line);
    while (std::getline(parts, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

int column_index(const std::vector<std::string>& header, const std::string& name) {
    const auto found = std::find(header.begin(), header.end(), name);
    if (found == header.end()) {
        throw std::runtime_error("no column named " + name);
    }
    return static_cast<int>(found - header.begin());
}

// What an earlier run left in the output file. A sweep may only continue rows
// from the same commit, a clean tree and the same protocol, because a table
// mixing them would not be one measurement. When it cannot, problem says why.
struct ExistingRows {
    std::set<std::string> keys;
    std::string problem;
};

ExistingRows read_existing_rows(const fs::path& path, const Protocol& protocol) {
    ExistingRows existing;
    std::ifstream stream(path);
    if (!stream) {
        return existing;
    }
    std::string line;
    if (!std::getline(stream, line)) {
        return existing;
    }
    if (line != CSV_HEADER) {
        existing.problem = "it has a different set of columns";
        return existing;
    }
    const std::vector<std::string> header = split_fields(line);
    const int commit = column_index(header, "commit");
    const int dirty = column_index(header, "dirty");
    const int protocol_column = column_index(header, "protocol");
    const int solver = column_index(header, "solver");
    const int variant = column_index(header, "variant");
    const int num_particles = column_index(header, "num_particles");
    const int block_size = column_index(header, "block_size");
    const int tile_size = column_index(header, "tile_size");
    const int fast_math = column_index(header, "fast_math");
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_fields(line);
        if (fields.size() != header.size()) {
            existing.problem = "it has a truncated row";
            return existing;
        }
        if (fields[dirty] != "0" || fields[commit] != CPS_GIT_STAMP) {
            existing.problem = "it holds rows from commit " + fields[commit] +
                               ", and this build is " + CPS_GIT_STAMP;
            return existing;
        }
        if (fields[protocol_column] != protocol.name) {
            existing.problem = "it holds rows from the " + fields[protocol_column] + " protocol";
            return existing;
        }
        existing.keys.insert(row_key(fields[solver], fields[variant], std::stoi(fields[num_particles]),
                                     std::stoi(fields[block_size]), std::stoi(fields[tile_size]),
                                     std::stoi(fields[fast_math])));
    }
    return existing;
}

// The checkpoints an earlier run already measured.
std::set<std::string> read_checkpoints(const fs::path& path) {
    std::set<std::string> checkpoints;
    std::ifstream stream(path);
    std::string line;
    bool first = true;
    while (std::getline(stream, line)) {
        if (!first && !line.empty()) {
            checkpoints.insert(line.substr(0, line.find(',')));
        }
        first = false;
    }
    return checkpoints;
}

std::ofstream open_for_append(const fs::path& path, const std::string& header) {
    const bool need_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream stream(path, std::ios::app);
    if (!stream) {
        throw std::runtime_error("cannot append to " + path.string());
    }
    if (need_header) {
        stream << header << "\n";
    }
    return stream;
}

void write_row_fields(std::ostream& stream, const BenchmarkRow& row) {
    stream << row.solver << ',' << row.variant << ',' << row.num_particles << ',' << row.block_size
           << ',' << row.tile_size << ',' << row.timed_steps << ',' << row.ms_per_step << ','
           << row.trials << ',' << row.ms_min << ',' << row.ms_median << ',' << row.ms_mean << ','
           << row.ms_stddev << ',' << row.ms_iqr << ',' << row.steps_per_second << ','
           << row.interactions_per_second << ',' << row.gflops << ',' << row.model_bytes_per_step
           << ',' << row.model_gb_s << ',' << row.model_bandwidth_fraction << ',' << row.fast_math
           << ',' << row.seed << ',' << row.gpu_name << ',' << row.cuda_runtime << ','
           << row.cuda_driver << ',' << row.commit << ',' << row.dirty << ',' << row.protocol << ','
           << row.timestamp;
}

void append_row(const fs::path& path, const BenchmarkRow& row) {
    std::ofstream stream = open_for_append(path, CSV_HEADER);
    write_row_fields(stream, row);
    stream << "\n";
}

void append_checkpoint(const fs::path& path, const std::string& checkpoint,
                       const BenchmarkRow& row) {
    std::ofstream stream = open_for_append(path, std::string("checkpoint,") + CSV_HEADER);
    stream << checkpoint << ',';
    write_row_fields(stream, row);
    stream << "\n";
}

// Every trial of a group, round by round in the order they ran, so the file
// shows the protocol and paired statistics can be computed from it.
void append_trials(const fs::path& path, const std::vector<Contender>& group,
                   const Protocol& protocol) {
    std::ofstream stream = open_for_append(path, TRIALS_HEADER);
    const std::string timestamp = now_iso8601();
    const std::size_t size = group.size();
    for (int round = 0; round < protocol.trials; ++round) {
        for (std::size_t position = 0; position < size; ++position) {
            const Contender& contender = group[(position + static_cast<std::size_t>(round)) % size];
            const cps::Config& config = contender.config;
            stream << cps::to_string(config.solver) << ',' << variant_name(config) << ','
                   << config.num_particles << ',' << config.block_size << ','
                   << contender.tile_size << ',' << CPS_FAST_MATH << ',' << (round + 1) << ','
                   << contender.timed_steps << ',' << contender.trial_ms[round] << ','
                   << CPS_GIT_STAMP << ',' << (tree_is_dirty() ? 1 : 0) << ',' << protocol.name
                   << ',' << timestamp << "\n";
        }
    }
}

struct PlannedConfig {
    cps::Config config;
    // Relative cost, used only to project the total up front.
    double weight = 0.0;
};

struct PlannedGroup {
    std::string label;
    std::vector<PlannedConfig> configs;
};

std::vector<PlannedGroup> plan_ladder(bool quick) {
    const std::vector<int> nbody_counts =
        quick ? std::vector<int>{4096, 8192}
              : std::vector<int>{10000, 20000, 50000, 100000, 150000, 200000};
    const std::vector<int> sph_counts =
        quick ? std::vector<int>{4096, 8192} : std::vector<int>{10000, 25000, 50000, 75000, 100000};

    std::vector<PlannedGroup> plan;
    for (int count : nbody_counts) {
        PlannedGroup group;
        group.label = "nbody N=" + std::to_string(count);
        for (cps::NBodyVariant variant :
             {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
            group.configs.push_back(
                {nbody_config(variant, count), static_cast<double>(count) * count});
        }
        plan.push_back(std::move(group));
    }
    PlannedGroup fluid;
    fluid.label = "sph";
    for (int count : sph_counts) {
        // The fluid step is close to linear in the particle count, and far
        // cheaper per particle than the all pairs sum.
        fluid.configs.push_back({sph_config(count), static_cast<double>(count) * 3000.0});
    }
    plan.push_back(std::move(fluid));
    return plan;
}

// The tile width sweep is a diagnostic rather than part of the ladder. Which
// width is the best choice on this card is an empirical question, and the
// answer belongs in the debug report with numbers behind it.
//
// The engine uses the default tile width, so the sweep drives the kernels
// directly rather than through engine.step(). It keeps the engine for its state.
// Both variants time a whole step, the sequence engine.step() runs, so a width
// can be compared across variants as well as within one.
std::vector<Contender> make_tile_sweep_group(int count) {
    std::vector<Contender> group;
    for (int tile : {32, 64, 128, 256}) {
        for (cps::NBodyVariant variant : {cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
            Contender contender = make_contender(nbody_config(variant, count));
            contender.tile_size = tile;
            cps::ParticleSystem* particles = &contender.engine->particles();
            const cps::Config config = contender.config;
            contender.step = [particles, config, variant, count, tile]() {
                if (variant == cps::NBodyVariant::Tiled) {
                    cps::integrate_verlet_position(
                        particles->x(), particles->y(), particles->z(), particles->vx(),
                        particles->vy(), particles->vz(), particles->ax(), particles->ay(),
                        particles->az(), count, config.dt, config.block_size);
                    cps::compute_forces_tiled(particles->x(), particles->y(), particles->z(),
                                              particles->mass(), particles->ax(), particles->ay(),
                                              particles->az(), count, config.softening,
                                              config.gravitational_constant, config.block_size,
                                              tile);
                    cps::integrate_verlet_velocity(particles->vx(), particles->vy(), particles->vz(),
                                                   particles->ax(), particles->ay(), particles->az(),
                                                   count, config.dt, config.block_size);
                } else {
                    cps::nbody_fused_step(
                        particles->x(), particles->y(), particles->z(), particles->x_next(),
                        particles->y_next(), particles->z_next(), particles->vx(), particles->vy(),
                        particles->vz(), particles->ax(), particles->ay(), particles->az(),
                        particles->mass(), count, config.softening, config.gravitational_constant,
                        config.dt, config.block_size, tile);
                    particles->swap_position_buffers();
                }
            };
            group.push_back(std::move(contender));
        }
    }
    return group;
}

// Wall time of one configuration under the protocol, for a step of step_ms.
double configuration_cost_ms(double step_ms, const Protocol& protocol) {
    const int steps = steps_per_trial(step_ms, protocol);
    return step_ms * (protocol.warmup_steps + 1 + protocol.trials * steps);
}

// Constructing an engine runs every refusal the solver has, including the fluid
// time step limits, so a sweep cannot die on a bad size hours in.
void check_plan(const std::vector<PlannedGroup>& plan, bool print) {
    for (const PlannedGroup& group : plan) {
        for (const PlannedConfig& point : group.configs) {
            const cps::SimulationEngine engine(point.config);
            if (print && point.config.solver == cps::SolverType::Sph) {
                const cps::SphDerivedParameters derived = cps::derive_sph_parameters(point.config);
                const cps::SphTimeStepLimits limits =
                    cps::sph_time_step_limits(point.config, derived.smoothing_radius);
                std::printf("  sph N=%-7d dt %g, %s limit %g\n", point.config.num_particles,
                            static_cast<double>(point.config.dt), limits.smallest_name(),
                            limits.smallest());
            }
        }
    }
}

void print_row(const BenchmarkRow& row) {
    std::printf("  %-5s %-6s N=%-7d tile %-3d %9.4f ms/step  iqr %7.4f  %8.1f steps/s",
                row.solver.c_str(), row.variant.c_str(), row.num_particles, row.tile_size,
                row.ms_per_step, row.ms_iqr, row.steps_per_second);
    if (row.solver == "nbody") {
        std::printf("  %8.1f GFLOP/s", row.gflops);
    } else {
        std::printf("  %8.2f Mparticle/s", row.interactions_per_second / 1.0e6);
    }
    std::printf("\n");
    std::fflush(stdout);
}

void print_usage() {
    std::printf(
        "bench_all: sweep the kernel ladder and record the results\n"
        "\n"
        "Writes the summary rows to the output file and, next to it, every trial to\n"
        "benchmark_trials.csv and the session drift check to benchmark_repeatability.csv.\n"
        "An interrupted sweep continues where it stopped, provided the rows in the\n"
        "output file came from this commit under this protocol. Otherwise it starts over.\n"
        "\n"
        "options:\n"
        "  --output <path>   results CSV (default experiments/results/benchmarks.csv)\n"
        "  --force           discard the output files and measure everything again\n"
        "  --quick           a short sweep for smoke testing the driver itself. Its rows are\n"
        "                    labelled quick, are not results, and go to the temporary\n"
        "                    directory unless --output says otherwise\n"
        "  --tile-sweep      also sweep the tile width, to tile_sweep.csv and\n"
        "                    tile_sweep_trials.csv next to the output\n"
        "  --allow-dirty     write rows from a tree with uncommitted code changes, or an\n"
        "                    unknown one, stamped dirty=1\n"
        "  --dry-run         check every planned configuration against the solver, print\n"
        "                    the plan, and exit without measuring\n"
        "  --version         print the version and commit, then exit\n"
        "  --help            print this message\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path results_dir = fs::path(CPS_PROJECT_ROOT) / "experiments" / "results";
        fs::path output;
        bool force = false;
        bool quick = false;
        bool tile_sweep = false;
        bool allow_dirty = false;
        bool dry_run = false;

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
            } else if (arg == "--allow-dirty") {
                allow_dirty = true;
            } else if (arg == "--dry-run") {
                dry_run = true;
            } else if (arg == "--version") {
                std::printf("%s\n", CPS_VERSION_STRING);
                return 0;
            } else if (arg == "--help") {
                print_usage();
                return 0;
            } else {
                std::fprintf(stderr, "unknown option %s\n", arg.c_str());
                print_usage();
                return 2;
            }
        }

        if (output.empty()) {
            output = quick ? fs::temp_directory_path() / "cps_bench_quick" / "benchmarks.csv"
                           : results_dir / "benchmarks.csv";
        }
        output = fs::absolute(output);
        const fs::path output_dir = output.parent_path();
        fs::create_directories(output_dir);

        if (quick && fs::equivalent(output_dir, results_dir)) {
            std::fprintf(stderr,
                         "error: --quick does not write into %s, because its rows are not "
                         "results and would replace the ones there\n",
                         results_dir.string().c_str());
            return 1;
        }
        if (tree_is_dirty() && !allow_dirty && !dry_run) {
            std::fprintf(stderr,
                         "error: this build is %s, so its rows could not be reproduced from "
                         "a commit. Commit the change and rebuild, or pass --allow-dirty to "
                         "write rows stamped dirty=1.\n",
                         CPS_VERSION_STRING);
            return 1;
        }

        const Protocol& protocol = quick ? QUICK_PROTOCOL : FULL_PROTOCOL;
        const int tile_count = quick ? 8192 : 100000;
        const int repeat_count = quick ? 8192 : 100000;
        const cps::Config repeat_config = nbody_config(cps::NBodyVariant::Tiled, repeat_count);

        const fs::path trials_path = output_dir / "benchmark_trials.csv";
        const fs::path repeat_path = output_dir / "benchmark_repeatability.csv";
        const fs::path tile_path = output_dir / "tile_sweep.csv";
        const fs::path tile_trials_path = output_dir / "tile_sweep_trials.csv";

        const cps::DeviceInfo info = cps::query_device(0);
        cps::print_device_info(info, stdout);
        std::printf("\nbuild: %s, fast math %s\n", CPS_VERSION_STRING,
                    CPS_FAST_MATH ? "on" : "off");
        std::printf("protocol: %s, %d rounds, %d warm up steps, at least %d timed steps\n",
                    protocol.name, protocol.trials, protocol.warmup_steps,
                    protocol.min_timed_steps);
        if (quick) {
            std::printf("QUICK SMOKE TEST of the driver: these numbers are not results\n");
        }
        if (tree_is_dirty()) {
            std::printf("DIRTY TREE: every row is stamped dirty=1\n");
        }

        const std::vector<PlannedGroup> plan = plan_ladder(quick);
        check_plan(plan, dry_run);
        if (dry_run) {
            for (const PlannedGroup& group : plan) {
                std::printf("  group %-14s %zu configurations\n", group.label.c_str(),
                            group.configs.size());
            }
            std::printf("every planned configuration is accepted; nothing was measured\n");
            return 0;
        }

        ExistingRows existing;
        if (!force) {
            existing = read_existing_rows(output, protocol);
        }
        std::vector<const PlannedGroup*> todo;
        std::size_t planned_points = 0;
        std::size_t present_points = 0;
        for (const PlannedGroup& group : plan) {
            std::size_t present = 0;
            for (const PlannedConfig& point : group.configs) {
                present += existing.keys.count(config_key(point.config));
            }
            if (present > 0 && present < group.configs.size() && existing.problem.empty()) {
                existing.problem = "group " + group.label + " is only partly there";
            }
            planned_points += group.configs.size();
            present_points += present;
        }
        // With no output file there is nothing to continue, so a trials or
        // repeatability file left beside it belongs to some other run.
        if (force || !existing.problem.empty() || !fs::exists(output)) {
            if (fs::exists(output)) {
                std::printf("\nstarting %s afresh: %s\n", output.string().c_str(),
                            force ? "--force was given" : existing.problem.c_str());
            }
            for (const fs::path& path : {output, trials_path, repeat_path}) {
                fs::remove(path);
            }
            existing.keys.clear();
            present_points = 0;
        }
        for (const PlannedGroup& group : plan) {
            if (existing.keys.count(config_key(group.configs.front().config)) == 0) {
                todo.push_back(&group);
            }
        }

        std::printf("\nplanned points: %zu of %zu (%zu already present in %s)\n",
                    planned_points - present_points, planned_points, present_points,
                    output.string().c_str());

        if (todo.empty() && !tile_sweep) {
            std::printf("nothing to do\n");
            return 0;
        }

        // Calibrate against one cheap point so the projection up front is
        // based on this machine rather than on a guess.
        {
            std::vector<Contender> probe;
            probe.push_back(make_contender(nbody_config(cps::NBodyVariant::Tiled, 8192)));
            warm_up_and_calibrate(probe.front(), protocol);
            const double ms_per_weight = probe.front().single_step_ms / (8192.0 * 8192.0);
            const auto cost_at = [&](double weight) {
                return configuration_cost_ms(weight * ms_per_weight, protocol);
            };
            double projected_ms = 3.0 * cost_at(static_cast<double>(repeat_count) * repeat_count);
            for (const PlannedGroup* group : todo) {
                for (const PlannedConfig& point : group->configs) {
                    projected_ms += cost_at(point.weight);
                }
            }
            if (tile_sweep) {
                projected_ms += 8.0 * cost_at(static_cast<double>(tile_count) * tile_count);
            }
            std::printf("calibration: %.4f ms/step at 8192 tiled particles\n",
                        probe.front().single_step_ms);
            std::printf("projected sweep total: about %s\n",
                        cps::format_duration(projected_ms * 1.0e-3).c_str());
        }
        std::fflush(stdout);

        // The same configuration at the start, middle and end of the sweep. How far
        // its medians move is the session drift, the smallest difference between
        // two rows of this sweep that means anything. It has its own file so it
        // never reaches a plot or a ladder table.
        std::set<std::string> checkpoints = read_checkpoints(repeat_path);
        std::vector<double> session_medians;
        const auto checkpoint = [&](const std::string& name) {
            if (checkpoints.count(name) > 0) {
                return;
            }
            checkpoints.insert(name);
            std::vector<Contender> group;
            group.push_back(make_contender(repeat_config));
            measure_group(group, protocol);
            const BenchmarkRow row = make_row(group.front(), info, protocol);
            append_checkpoint(repeat_path, name, row);
            session_medians.push_back(row.ms_median);
            std::printf("  repeatability %-6s N=%d %9.4f ms/step\n", name.c_str(),
                        row.num_particles, row.ms_per_step);
            std::fflush(stdout);
        };

        checkpoint("start");

        const std::size_t total_groups = plan.size() + (tile_sweep ? 1 : 0);
        cps::ProgressBar bar("benchmark sweep", static_cast<long long>(todo.size()));
        long long done = 0;
        for (std::size_t index = 0; index < plan.size(); ++index) {
            const PlannedGroup& planned = plan[index];
            if (std::find(todo.begin(), todo.end(), &planned) == todo.end()) {
                continue;
            }
            if (index >= total_groups / 2) {
                checkpoint("middle");
            }
            std::vector<Contender> group;
            for (const PlannedConfig& point : planned.configs) {
                group.push_back(make_contender(point.config));
            }
            measure_group(group, protocol);
            // Trials first: a crash between the two leaves a group to remeasure,
            // and gen_report_assets.py refuses the duplicate trials it would make.
            append_trials(trials_path, group, protocol);
            for (const Contender& contender : group) {
                const BenchmarkRow row = make_row(contender, info, protocol);
                append_row(output, row);
                print_row(row);
            }
            bar.update(++done);
        }
        bar.finish();

        if (tile_sweep) {
            checkpoint("middle");
            for (const fs::path& path : {tile_path, tile_trials_path}) {
                fs::remove(path);
            }
            std::printf("\ntile width sweep at %d particles\n", tile_count);
            std::vector<Contender> group = make_tile_sweep_group(tile_count);
            measure_group(group, protocol);
            append_trials(tile_trials_path, group, protocol);
            for (const Contender& contender : group) {
                const BenchmarkRow row = make_row(contender, info, protocol);
                append_row(tile_path, row);
                print_row(row);
            }
            std::printf("wrote %s\n", tile_path.string().c_str());
        }

        checkpoint("end");
        if (session_medians.size() == 3) {
            const auto [low, high] =
                std::minmax_element(session_medians.begin(), session_medians.end());
            std::printf("session drift: %.2f percent of the median\n",
                        100.0 * (*high - *low) / bench::median(session_medians));
        }

        std::printf("\nwrote %s\n", output.string().c_str());
        if (quick) {
            std::printf("QUICK SMOKE TEST of the driver: these numbers are not results\n");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
