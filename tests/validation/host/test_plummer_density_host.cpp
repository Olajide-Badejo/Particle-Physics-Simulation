#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "test_support_host.hpp"

namespace {

// Gate: the sampled radial density of the Plummer sphere is consistent with
// the analytic profile at a chi squared p value above 0.05.
//
// The cumulative mass profile of a Plummer sphere is
//
//     M(r) / M = r^3 / (r^2 + a^2)^{3/2}
//
// so the expected count in a shell is the total count times the difference of
// that function at the shell edges. A chi squared test over those shells asks
// the right question: not whether the sampler produced a plausible looking
// blob, but whether the deviations from the profile are the size that random
// sampling explains. A sampler with a subtly wrong exponent produces a blob
// that looks entirely reasonable and fails this immediately.

double plummer_enclosed_fraction(double r, double a) {
    const double ratio = r / a;
    const double denominator = std::pow(1.0 + ratio * ratio, 1.5);
    return (ratio * ratio * ratio) / denominator;
}

struct ChiSquareResult {
    double statistic = 0.0;
    int degrees_of_freedom = 0;
    double p_value = 0.0;
    std::vector<double> edges;
    std::vector<double> observed;
    std::vector<double> expected;
};

// Shell edges are chosen so that every shell holds the same expected count.
// Equal width shells would put almost nothing in the outermost ones, and a
// chi squared statistic built on expected counts of two or three is not valid.
ChiSquareResult radial_chi_square(const cps::HostParticleState& state, double a, int shells) {
    std::vector<double> radii;
    radii.reserve(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        radii.push_back(std::sqrt(static_cast<double>(state.x[i]) * state.x[i] +
                                  static_cast<double>(state.y[i]) * state.y[i] +
                                  static_cast<double>(state.z[i]) * state.z[i]));
    }

    // The generator clamps the sampled quantile at 0.999 so the sphere stays
    // finite, so the test is restricted to the same range. Including the
    // truncated tail would test the clamp rather than the sampler.
    const double quantile_limit = 0.999;
    const double outer_edge = a / std::sqrt(std::pow(quantile_limit, -2.0 / 3.0) - 1.0);

    ChiSquareResult result;
    result.edges.resize(static_cast<std::size_t>(shells) + 1);
    result.edges[0] = 0.0;
    for (int s = 1; s < shells; ++s) {
        // Invert the profile at equally spaced mass fractions.
        const double fraction =
            quantile_limit * static_cast<double>(s) / static_cast<double>(shells);
        result.edges[static_cast<std::size_t>(s)] =
            a / std::sqrt(std::pow(fraction, -2.0 / 3.0) - 1.0);
    }
    result.edges[static_cast<std::size_t>(shells)] = outer_edge;

    const double total = static_cast<double>(radii.size());
    result.observed.assign(static_cast<std::size_t>(shells), 0.0);
    result.expected.assign(static_cast<std::size_t>(shells), 0.0);

    for (double r : radii) {
        if (r >= outer_edge) {
            continue;
        }
        const auto it = std::upper_bound(result.edges.begin(), result.edges.end(), r);
        const std::size_t bin = static_cast<std::size_t>(it - result.edges.begin()) - 1;
        if (bin < result.observed.size()) {
            result.observed[bin] += 1.0;
        }
    }

    for (int s = 0; s < shells; ++s) {
        const double inner = plummer_enclosed_fraction(result.edges[static_cast<std::size_t>(s)], a);
        const double outer =
            plummer_enclosed_fraction(result.edges[static_cast<std::size_t>(s) + 1], a);
        result.expected[static_cast<std::size_t>(s)] = total * (outer - inner);
    }

    result.statistic = 0.0;
    for (int s = 0; s < shells; ++s) {
        const double e = result.expected[static_cast<std::size_t>(s)];
        if (e <= 0.0) {
            continue;
        }
        const double d = result.observed[static_cast<std::size_t>(s)] - e;
        result.statistic += d * d / e;
    }
    // One constraint is used up by matching the total count.
    result.degrees_of_freedom = shells - 1;
    result.p_value = cps_test::chi_square_p_value(result.statistic, result.degrees_of_freedom);
    return result;
}

cps::Config plummer_config(int count, uint32_t seed) {
    cps::Config config;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.total_mass = 1.0f;
    config.plummer_radius = 1.0f;
    config.gravitational_constant = 1.0f;
    config.seed = seed;
    return config;
}

TEST(PlummerDensity, RadialProfileIsConsistentWithTheAnalyticModel) {
    const int count = 100000;
    const int shells = 40;
    const cps::HostParticleState state = cps::make_plummer_sphere(plummer_config(count, 20260802u));
    const ChiSquareResult result = radial_chi_square(state, 1.0, shells);

    std::printf("plummer radial density: chi squared = %.3f on %d degrees of freedom, p = %.4f\n",
                result.statistic, result.degrees_of_freedom, result.p_value);

    const std::filesystem::path out = cps_test::results_directory();
    std::filesystem::create_directories(out);
    std::vector<double> centres;
    for (int s = 0; s < shells; ++s) {
        centres.push_back(0.5 * (result.edges[static_cast<std::size_t>(s)] +
                                 result.edges[static_cast<std::size_t>(s) + 1]));
    }
    std::FILE* file = std::fopen((out / "plummer_radial_density.csv").string().c_str(), "w");
    if (file != nullptr) {
        std::fprintf(file, "shell_inner,shell_outer,shell_centre,observed,expected\n");
        for (int s = 0; s < shells; ++s) {
            std::fprintf(file, "%.17g,%.17g,%.17g,%.17g,%.17g\n",
                         result.edges[static_cast<std::size_t>(s)],
                         result.edges[static_cast<std::size_t>(s) + 1],
                         centres[static_cast<std::size_t>(s)],
                         result.observed[static_cast<std::size_t>(s)],
                         result.expected[static_cast<std::size_t>(s)]);
        }
        std::fclose(file);
    }

    cps_test::record_gate("plummer_radial_density_chi_square_p_value", result.p_value, 0.05,
                          "greater_than", result.p_value > 0.05);

    EXPECT_GT(result.p_value, 0.05)
        << "the sampled radial density does not match the Plummer profile: chi squared "
        << result.statistic << " on " << result.degrees_of_freedom << " degrees of freedom";
}

// A single seed passing could be luck. Across several seeds the p values
// should be scattered over the unit interval rather than clustered near zero.
TEST(PlummerDensity, HoldsAcrossSeveralSeeds) {
    const int count = 40000;
    const int shells = 25;
    int passing = 0;
    for (uint32_t seed : {1u, 7u, 99u, 12345u, 20260802u}) {
        const cps::HostParticleState state = cps::make_plummer_sphere(plummer_config(count, seed));
        const ChiSquareResult result = radial_chi_square(state, 1.0, shells);
        std::printf("  seed %10u: chi squared %8.3f, p = %.4f\n", seed, result.statistic,
                    result.p_value);
        if (result.p_value > 0.05) {
            ++passing;
        }
    }
    // Allowing one of five to fall below 0.05 is consistent with the test
    // itself being a five percent test.
    EXPECT_GE(passing, 4);
}

// The chi squared machinery has to be able to reject. Feeding it a sphere with
// a deliberately wrong scale radius must fail, otherwise the test above proves
// nothing about the sampler.
TEST(PlummerDensity, TheTestRejectsAProfileWithTheWrongScaleRadius) {
    const int count = 40000;
    const cps::HostParticleState state = cps::make_plummer_sphere(plummer_config(count, 5u));
    // Compare against a model whose scale radius is ten percent too large.
    const ChiSquareResult wrong = radial_chi_square(state, 1.1, 25);
    std::printf("against a deliberately wrong scale radius: chi squared %.3f, p = %.6f\n",
                wrong.statistic, wrong.p_value);
    EXPECT_LT(wrong.p_value, 0.05);
}

TEST(PlummerDensity, TheChiSquarePValueMatchesKnownValues) {
    // Reference values from standard tables.
    EXPECT_NEAR(cps_test::chi_square_p_value(3.84146, 1), 0.05, 1.0e-4);
    EXPECT_NEAR(cps_test::chi_square_p_value(5.99146, 2), 0.05, 1.0e-4);
    EXPECT_NEAR(cps_test::chi_square_p_value(18.307, 10), 0.05, 1.0e-3);
    EXPECT_NEAR(cps_test::chi_square_p_value(9.48773, 4), 0.05, 1.0e-4);
    EXPECT_NEAR(cps_test::chi_square_p_value(0.0, 5), 1.0, 1.0e-9);
}

}  // namespace
