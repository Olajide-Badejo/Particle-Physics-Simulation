#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

#include "cps/core/initial_conditions.hpp"
#include "cps/physics/sph_kernels.hpp"
#include "test_support_host.hpp"
#include "validation/sph_fluid_config.hpp"

namespace {

using cps_test::fluid_config;

// The host half of the SPH gates: the kernel normalisation and the derived
// resolution need only the shared kernel formulas and the parameter
// derivation, so they run with no device. The density, neighbour search and
// running fluid gates need a GPU and stay in
// tests/validation/test_sph_validation.cpp.

// Composite Simpson quadrature of 4 pi integral W(r) r^2 dr over the support.
template <typename Kernel>
double integrate_kernel(Kernel kernel, double h, int intervals = 400000) {
    if (intervals % 2 != 0) {
        ++intervals;
    }
    const double step = h / static_cast<double>(intervals);
    double total = 0.0;
    for (int i = 0; i <= intervals; ++i) {
        const double r = static_cast<double>(i) * step;
        double weight = 2.0;
        if (i == 0 || i == intervals) {
            weight = 1.0;
        } else if (i % 2 == 1) {
            weight = 4.0;
        }
        total += weight * kernel(r) * r * r;
    }
    return 4.0 * cps::PI_D * total * step / 3.0;
}

TEST(SphValidation, SmoothingKernelsIntegrateToOne) {
    // Checked at the smoothing radius the shipped fifty thousand particle dam
    // break actually derives, not only at a convenient round number.
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(fluid_config(50000, 0));
    const double h = derived.smoothing_radius;
    std::printf("derived smoothing radius %.6g, particle mass %.6g, expected neighbours %.1f\n", h,
                static_cast<double>(derived.particle_mass), derived.expected_neighbour_count);

    const double poly6 = integrate_kernel(
        [h](double r) {
            return static_cast<double>(
                cps::poly6_kernel(static_cast<float>(r * r), static_cast<float>(h)));
        },
        h);
    const double spiky = integrate_kernel(
        [h](double r) {
            return static_cast<double>(
                cps::spiky_kernel(static_cast<float>(r), static_cast<float>(h)));
        },
        h);

    std::printf("kernel normalisation: poly6 = %.9f, spiky = %.9f\n", poly6, spiky);
    cps_test::record_gate("sph_poly6_kernel_normalisation_error", std::abs(poly6 - 1.0), 1.0e-3,
                          "less_than", std::abs(poly6 - 1.0) < 1.0e-3);
    cps_test::record_gate("sph_spiky_kernel_normalisation_error", std::abs(spiky - 1.0), 1.0e-3,
                          "less_than", std::abs(spiky - 1.0) < 1.0e-3);
    EXPECT_NEAR(poly6, 1.0, 1.0e-3);
    EXPECT_NEAR(spiky, 1.0, 1.0e-3);
}

TEST(SphValidation, TheDerivedResolutionPutsEnoughNeighboursInTheSupport) {
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(fluid_config(50000, 0));
    // Below about twenty neighbours the density estimate is too noisy to be
    // useful; far above forty the neighbour loop dominates the step for no
    // additional accuracy.
    EXPECT_GT(derived.expected_neighbour_count, 20.0);
    EXPECT_LT(derived.expected_neighbour_count, 60.0);
    EXPECT_GT(derived.smoothing_radius, derived.particle_spacing);
}

}  // namespace
