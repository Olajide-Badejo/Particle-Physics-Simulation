#include <gtest/gtest.h>

#include <cmath>

#include "cps/physics/gravity.hpp"
#include "test_support_host.hpp"

namespace {

using cps_test::random_cloud;

constexpr float SOFTENING = 0.02f;

// The host half of the force gate: properties of the shared pair expression
// the kernels are built from, checked on the host with no device involved.
// The comparison of the device kernel against the double precision reference
// needs a GPU and stays in tests/validation/test_force_antisymmetry.cpp.

// Newton's third law, pair by pair. The kernel only ever produces summed
// accelerations, so this checks the shared expression the kernel is built out
// of, which is where an asymmetry would have to originate.
TEST(ForceAntisymmetry, ThePairInteractionIsExactlyAntisymmetric) {
    const int count = 100;
    const cps::HostParticleState state = random_cloud(count, 4711u);
    const float softening_sq = SOFTENING * SOFTENING;

    double worst = 0.0;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            const std::size_t a = static_cast<std::size_t>(i);
            const std::size_t b = static_cast<std::size_t>(j);

            float aix = 0.0f, aiy = 0.0f, aiz = 0.0f;
            cps::accumulate_pair_acceleration(state.x[a], state.y[a], state.z[a], state.x[b],
                                              state.y[b], state.z[b], state.mass[b], softening_sq,
                                              aix, aiy, aiz);
            float ajx = 0.0f, ajy = 0.0f, ajz = 0.0f;
            cps::accumulate_pair_acceleration(state.x[b], state.y[b], state.z[b], state.x[a],
                                              state.y[a], state.z[a], state.mass[a], softening_sq,
                                              ajx, ajy, ajz);

            // Forces, not accelerations: m_i a_i must be exactly the negative
            // of m_j a_j.
            const double fx = static_cast<double>(state.mass[a]) * aix +
                              static_cast<double>(state.mass[b]) * ajx;
            const double fy = static_cast<double>(state.mass[a]) * aiy +
                              static_cast<double>(state.mass[b]) * ajy;
            const double fz = static_cast<double>(state.mass[a]) * aiz +
                              static_cast<double>(state.mass[b]) * ajz;
            const double residual = std::sqrt(fx * fx + fy * fy + fz * fz);
            const double magnitude =
                static_cast<double>(state.mass[a]) *
                std::sqrt(static_cast<double>(aix) * aix + static_cast<double>(aiy) * aiy +
                          static_cast<double>(aiz) * aiz);
            if (magnitude > 0.0) {
                worst = std::max(worst, residual / magnitude);
            }
        }
    }
    std::printf("worst pairwise antisymmetry residual over %d pairs: %.3e\n",
                count * (count - 1) / 2, worst);
    cps_test::record_gate("pairwise_force_antisymmetry_residual", worst, 1.0e-6, "less_than",
                          worst < 1.0e-6);
    EXPECT_LT(worst, 1.0e-6);
}

// Self interaction has to contribute nothing. The tiled kernel deliberately
// includes the diagonal term so its inner loop needs no branch, which is only
// safe because the numerator vanishes while the softened denominator stays
// finite.
TEST(ForceAntisymmetry, SelfInteractionContributesNothing) {
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    cps::accumulate_pair_acceleration(1.5f, -2.5f, 0.25f, 1.5f, -2.5f, 0.25f, 3.0f,
                                      SOFTENING * SOFTENING, ax, ay, az);
    EXPECT_FLOAT_EQ(ax, 0.0f);
    EXPECT_FLOAT_EQ(ay, 0.0f);
    EXPECT_FLOAT_EQ(az, 0.0f);
}

// Softening is what keeps a close pass finite. Without it the acceleration at
// contact is unbounded, one step throws the pair to infinity, and the run is
// destroyed. This checks the bound is where the softening length says it is.
TEST(ForceAntisymmetry, SofteningBoundsTheAccelerationAtContact) {
    const float eps = 0.1f;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    // Separation far below the softening length.
    cps::accumulate_pair_acceleration(0.0f, 0.0f, 0.0f, 1.0e-6f, 0.0f, 0.0f, 1.0f, eps * eps, ax,
                                      ay, az);
    EXPECT_TRUE(std::isfinite(ax));
    // a = m d / (d^2 + eps^2)^{3/2}, which for d far below eps is m d / eps^3.
    const float expected = 1.0e-6f / (eps * eps * eps);
    EXPECT_NEAR(ax, expected, expected * 1.0e-3f);
}

TEST(ForceAntisymmetry, TheForceFallsOffAsTheInverseSquareAtLargeSeparation) {
    const float eps = 0.001f;
    float a1x = 0.0f, a1y = 0.0f, a1z = 0.0f;
    float a2x = 0.0f, a2y = 0.0f, a2z = 0.0f;
    cps::accumulate_pair_acceleration(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, eps * eps, a1x, a1y,
                                      a1z);
    cps::accumulate_pair_acceleration(0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, eps * eps, a2x, a2y,
                                      a2z);
    // Doubling the separation must quarter the acceleration.
    EXPECT_NEAR(a1x / a2x, 4.0f, 1.0e-3f);
}

// The potential and the force have to come from the same expression. Pairing a
// softened force with an unsoftened potential would break the energy gate for
// a reason that has nothing to do with the integrator, and it is an easy
// mistake to make because both look right in isolation.
TEST(ForceAntisymmetry, ThePotentialIsConsistentWithTheForce) {
    const double eps = 0.05;
    const double m1 = 1.5, m2 = 2.5;
    const double r = 0.7;
    // The separation is passed through a float, so the step cannot be made
    // arbitrarily small: below about 1e-4 the rounding of r plus delta
    // dominates the difference. A step of 1e-3 leaves the truncation error of
    // the central difference well below the tolerance while staying far above
    // the representable spacing.
    const double delta = 1.0e-3;

    auto potential_at = [&](double separation) {
        return cps::pair_potential(0.0f, 0.0f, 0.0f, static_cast<float>(separation), 0.0f, 0.0f,
                                   static_cast<float>(m1), static_cast<float>(m2),
                                   static_cast<float>(eps * eps));
    };

    // U is negative and rises towards zero with separation, so its radial
    // derivative is the magnitude of the attractive force.
    const double numerical = (potential_at(r + delta) - potential_at(r - delta)) / (2.0 * delta);

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    cps::accumulate_pair_acceleration(0.0f, 0.0f, 0.0f, static_cast<float>(r), 0.0f, 0.0f,
                                      static_cast<float>(m2), static_cast<float>(eps * eps), ax, ay,
                                      az);
    const double analytic_force = m1 * static_cast<double>(ax);

    std::printf("force from the potential gradient %.9g, force from the force law %.9g\n",
                numerical, analytic_force);
    EXPECT_NEAR(numerical, analytic_force, std::abs(analytic_force) * 1.0e-3);
}

}  // namespace
