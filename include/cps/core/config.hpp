#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace cps {

enum class SolverType { NBody, Sph };
enum class NBodyVariant { Naive, Tiled, Fused };
enum class InitialCondition { TwoBody, Plummer, DamBreak };

std::string to_string(SolverType value);
std::string to_string(NBodyVariant value);
std::string to_string(InitialCondition value);

SolverType solver_from_string(std::string_view text);
NBodyVariant variant_from_string(std::string_view text);
InitialCondition initial_condition_from_string(std::string_view text);

// Plain host vector type. The configuration layer is deliberately free of any
// CUDA dependency so that configuration parsing can be tested without a GPU.
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct SphConfig {
    // A smoothing radius or particle mass of zero means derive it: the radius
    // from the fluid volume and the particle count, the mass from a kernel sum
    // over the lattice actually laid out. Fixing either by hand while the
    // particle count moves is the fastest way to end up with a resolution the
    // kernel cannot support, so the derived value is the default.
    float smoothing_radius = 0.0f;
    float rest_density = 1000.0f;
    float gas_constant = 1000.0f;
    // Dynamic viscosity in Pa s. The kernel forms a = mu / rho_i sum m_j / rho_j
    // (v_j - v_i) lap W, so the default of 0.2 at the default rest density is a
    // kinematic viscosity of 2e-4 m^2/s, about 200 times that of water.
    float viscosity = 0.2f;
    float particle_mass = 0.0f;
    // Fraction of the normal velocity kept at a wall. The default of 0.4 keeps 16
    // percent of the normal kinetic energy per impact.
    float boundary_damping = 0.4f;
    Vec3 domain_min{0.0f, 0.0f, 0.0f};
    Vec3 domain_max{1.0f, 1.0f, 1.0f};
    // Axis aligned region filled with fluid at t = 0.
    Vec3 fluid_min{0.0f, 0.0f, 0.0f};
    Vec3 fluid_max{0.4f, 0.8f, 1.0f};
    float jitter = 0.05f;  // fraction of the particle spacing
};

struct Config {
    std::string name = "unnamed";
    SolverType solver = SolverType::NBody;
    NBodyVariant variant = NBodyVariant::Tiled;
    InitialCondition initial_condition = InitialCondition::Plummer;

    int num_particles = 10000;
    long long steps = 1000;
    float dt = 1.0e-3f;
    // Run control, not physics: turns the fluid solver's refusal of a time step
    // above its stability limits into a warning.
    bool allow_unstable_dt = false;
    float softening = 0.01f;
    float gravitational_constant = 1.0f;
    uint32_t seed = 20260802u;
    int block_size = 256;

    // Initial condition parameters for the gravitational cases.
    float total_mass = 1.0f;
    float plummer_radius = 1.0f;
    float two_body_separation = 1.0f;

    // External uniform acceleration, used by the fluid cases.
    Vec3 external_gravity{0.0f, -9.81f, 0.0f};

    long long snapshot_interval = 1000;
    std::string snapshot_dir = "snapshots";
    // Each diagnostics row costs an O(N^2) potential pass and two synchronous
    // copies, so the default samples sparsely rather than every step.
    long long diagnostics_interval = 100;
    std::string diagnostics_path = "experiments/results/diagnostics.csv";

    SphConfig sph{};

    static Config from_json_string(std::string_view text);
    static Config from_json_file(const std::filesystem::path& path);

    // Throws std::runtime_error with a message naming the offending field.
    // Parsing also rejects any key it does not read, at every level.
    void validate() const;

    // Canonical form: every field is written, keys are sorted, and floats are
    // printed with enough digits that parsing the output gives back the same
    // Config bit for bit.
    std::string to_json_string(int indent = 2) const;
};

}  // namespace cps
