#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "test_support_host.hpp"

namespace {

// Gate: the order of velocity Verlet, measured from its energy error.
//
// The single precision energy gate cannot see the order, because rounding
// dominates it. This one runs in double precision on the host, where the
// truncation error stands well clear of rounding at every step size used.
//
// The orbit matters. On an eccentric orbit the leading term of the modified
// Hamiltonian varies along the path, so the energy error is second order, as
// the method is. On a circular orbit rotational symmetry holds that term
// constant, so its contribution to the energy error cancels and what is left is
// fourth order. The limits are stated in docs/physics_validation.md.

constexpr double GRAVITATIONAL_CONSTANT = 1.0;
constexpr double BODY_MASS = 1.0;
constexpr double DURATION = 10.0;
constexpr double SLOPE_LIMIT = 0.15;

struct TwoBody {
    std::vector<double> x{0.0, 0.0};
    std::vector<double> y{0.0, 0.0};
    std::vector<double> z{0.0, 0.0};
    std::vector<double> vx{0.0, 0.0};
    std::vector<double> vy{0.0, 0.0};
    std::vector<double> vz{0.0, 0.0};
    std::vector<double> mass{BODY_MASS, BODY_MASS};
};

// The two bodies a distance apart on the x axis, each moving at speed along y.
TwoBody two_bodies(double separation, double speed) {
    TwoBody state;
    state.x = {-0.5 * separation, 0.5 * separation};
    state.vy = {-speed, speed};
    return state;
}

// Semi major axis one, starting at periapsis. The speed is the bare Kepler one:
// a softening of 0.01 changes the eccentricity only slightly, and the order does
// not depend on its exact value.
TwoBody eccentric_orbit(double eccentricity) {
    const double total_mass = 2.0 * BODY_MASS;
    const double separation = 1.0 - eccentricity;
    const double relative_speed = std::sqrt(GRAVITATIONAL_CONSTANT * total_mass *
                                            (1.0 + eccentricity) / (1.0 - eccentricity));
    return two_bodies(separation, 0.5 * relative_speed);
}

// Separation one, with the speed from the softened law so the orbit is circular.
TwoBody circular_orbit(double softening) {
    const double separation = 1.0;
    const double denominator = std::pow(separation * separation + softening * softening, 1.5);
    const double speed =
        separation * std::sqrt(GRAVITATIONAL_CONSTANT * BODY_MASS / (2.0 * denominator));
    return two_bodies(separation, speed);
}

double total_energy(const TwoBody& state, double softening) {
    double kinetic = 0.0;
    for (std::size_t i = 0; i < state.x.size(); ++i) {
        kinetic += 0.5 * state.mass[i] *
                   (state.vx[i] * state.vx[i] + state.vy[i] * state.vy[i] +
                    state.vz[i] * state.vz[i]);
    }
    return kinetic + cps_test::reference_potential(state.x, state.y, state.z, state.mass, softening,
                                                   GRAVITATIONAL_CONSTANT);
}

// Largest |E(t) - E(0)| / |E(0)| over velocity Verlet from t = 0 to DURATION.
double worst_energy_error(TwoBody state, double softening, double dt) {
    const double initial = total_energy(state, softening);
    const long long steps = std::llround(DURATION / dt);
    std::vector<double> ax, ay, az;
    cps_test::reference_accelerations(state.x, state.y, state.z, state.mass, softening,
                                      GRAVITATIONAL_CONSTANT, ax, ay, az);
    double worst = 0.0;
    for (long long step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < state.x.size(); ++i) {
            state.vx[i] += 0.5 * dt * ax[i];
            state.vy[i] += 0.5 * dt * ay[i];
            state.vz[i] += 0.5 * dt * az[i];
            state.x[i] += dt * state.vx[i];
            state.y[i] += dt * state.vy[i];
            state.z[i] += dt * state.vz[i];
        }
        cps_test::reference_accelerations(state.x, state.y, state.z, state.mass, softening,
                                          GRAVITATIONAL_CONSTANT, ax, ay, az);
        for (std::size_t i = 0; i < state.x.size(); ++i) {
            state.vx[i] += 0.5 * dt * ax[i];
            state.vy[i] += 0.5 * dt * ay[i];
            state.vz[i] += 0.5 * dt * az[i];
        }
        worst = std::max(worst, std::abs(total_energy(state, softening) - initial));
    }
    return worst / std::abs(initial);
}

// Least squares slope of log(max |dE/E|) against log(dt).
double fitted_slope(const TwoBody& orbit, double softening, const char* label) {
    const double steps[] = {64.0e-3, 32.0e-3, 16.0e-3, 8.0e-3, 4.0e-3};
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    double count = 0.0;
    for (const double dt : steps) {
        const double error = worst_energy_error(orbit, softening, dt);
        std::printf("%s: dt = %.3f, max |dE/E| = %.6e\n", label, dt, error);
        const double log_dt = std::log(dt);
        const double log_error = std::log(error);
        sum_x += log_dt;
        sum_y += log_error;
        sum_xx += log_dt * log_dt;
        sum_xy += log_dt * log_error;
        count += 1.0;
    }
    return (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);
}

TEST(VerletConvergence, AnEccentricOrbitShowsSecondOrder) {
    const double slope =
        fitted_slope(eccentric_orbit(0.5), 0.01, "eccentricity 0.5, softening 0.01");
    const double error = std::abs(slope - 2.0);
    std::printf("eccentric orbit: fitted slope %.3f, against 2 within %.2f\n", slope, SLOPE_LIMIT);
    cps_test::record_gate("verlet_energy_slope_error_eccentric", error, SLOPE_LIMIT, "less_than",
                          error < SLOPE_LIMIT);
    EXPECT_LT(error, SLOPE_LIMIT) << "fitted slope " << slope;
}

TEST(VerletConvergence, AnUnsoftenedCircularOrbitShowsFourthOrder) {
    const double slope = fitted_slope(circular_orbit(0.0), 0.0, "circular, softening 0");
    const double error = std::abs(slope - 4.0);
    std::printf("circular orbit: fitted slope %.3f, against 4 within %.2f\n", slope, SLOPE_LIMIT);
    cps_test::record_gate("verlet_energy_slope_error_circular", error, SLOPE_LIMIT, "less_than",
                          error < SLOPE_LIMIT);
    EXPECT_LT(error, SLOPE_LIMIT) << "fitted slope " << slope;
}

// Not a gate. The figure docs/physics_validation.md quotes beside the single
// precision energy gate: what truncation alone contributes at that gate's step
// size and softening, which shows the gate's value is rounding.
TEST(VerletConvergence, TruncationErrorAtTheSinglePrecisionGateStep) {
    const double softening = 0.01;
    const double error = worst_energy_error(circular_orbit(softening), softening, 1.0e-3);
    std::printf("truncation error: dt = 0.001, softening 0.01, circular, T = 10: %.3e\n", error);
    EXPECT_GT(error, 0.0);
}

}  // namespace
