#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "physics/sph_kernels.hpp"

namespace {

// Integrate a spherically symmetric kernel over its support:
//
//     I = 4 pi integral_0^h W(r) r^2 dr
//
// by composite Simpson's rule. A correctly normalised smoothing kernel gives
// exactly one. This is the single most useful check on the whole fluid solver,
// because a kernel whose integral is off by a few percent produces a density
// field that is off by the same few percent everywhere, and that error is
// invisible: the fluid still looks like a fluid, it just sits at the wrong
// rest density and the pressure term quietly fights gravity at the wrong
// strength.
template <typename Kernel>
double integrate_over_support(Kernel kernel, double h, int intervals = 200000) {
    // Simpson's rule needs an even number of intervals.
    if (intervals % 2 != 0) {
        ++intervals;
    }
    const double step = h / static_cast<double>(intervals);
    double total = 0.0;
    for (int i = 0; i <= intervals; ++i) {
        const double r = static_cast<double>(i) * step;
        const double value = kernel(r) * r * r;
        double weight = 2.0;
        if (i == 0 || i == intervals) {
            weight = 1.0;
        } else if (i % 2 == 1) {
            weight = 4.0;
        }
        total += weight * value;
    }
    total *= step / 3.0;
    return 4.0 * cps::PI_D * total;
}

constexpr double NORMALIZATION_TOLERANCE = 1.0e-3;

TEST(SphKernels, Poly6IntegratesToOne) {
    for (double h : {0.02, 0.05, 0.1, 0.5, 1.0}) {
        const double integral = integrate_over_support(
            [h](double r) {
                return static_cast<double>(
                    cps::poly6_kernel(static_cast<float>(r * r), static_cast<float>(h)));
            },
            h);
        EXPECT_NEAR(integral, 1.0, NORMALIZATION_TOLERANCE) << "smoothing radius " << h;
    }
}

TEST(SphKernels, SpikyIntegratesToOne) {
    for (double h : {0.02, 0.05, 0.1, 0.5, 1.0}) {
        const double integral = integrate_over_support(
            [h](double r) {
                return static_cast<double>(
                    cps::spiky_kernel(static_cast<float>(r), static_cast<float>(h)));
            },
            h);
        EXPECT_NEAR(integral, 1.0, NORMALIZATION_TOLERANCE) << "smoothing radius " << h;
    }
}

TEST(SphKernels, ViscosityKernelIntegratesToOne) {
    // Singular at the origin but integrable, so the quadrature starts just
    // above zero and the omitted shell is far below the tolerance.
    for (double h : {0.05, 0.1, 0.5}) {
        const int intervals = 400000;
        const double start = h * 1.0e-6;
        const double step = (h - start) / intervals;
        double total = 0.0;
        for (int i = 0; i <= intervals; ++i) {
            const double r = start + static_cast<double>(i) * step;
            const double value =
                static_cast<double>(
                    cps::viscosity_kernel(static_cast<float>(r), static_cast<float>(h))) *
                r * r;
            double weight = 2.0;
            if (i == 0 || i == intervals) {
                weight = 1.0;
            } else if (i % 2 == 1) {
                weight = 4.0;
            }
            total += weight * value;
        }
        total *= step / 3.0;
        EXPECT_NEAR(4.0 * cps::PI_D * total, 1.0, NORMALIZATION_TOLERANCE)
            << "smoothing radius " << h;
    }
}

TEST(SphKernels, AllKernelsVanishOutsideTheSupport) {
    const float h = 0.1f;
    EXPECT_FLOAT_EQ(cps::poly6_kernel(h * h * 1.001f, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::poly6_kernel(4.0f * h * h, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::spiky_kernel(h * 1.001f, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::spiky_gradient_magnitude(h * 1.001f, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::viscosity_laplacian(h * 1.001f, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::viscosity_kernel(h * 1.001f, h), 0.0f);
}

// Every kernel must reach zero at the support boundary rather than stepping
// off a cliff, because a discontinuity there injects an impulse into any
// particle pair that crosses it. The comparison has to be relative to the
// peak: these kernels carry a 1/h^6 normalisation, so at a smoothing radius of
// 0.1 their peak values are of order a million and an absolute tolerance says
// nothing at all.
TEST(SphKernels, KernelsApproachZeroAtTheSupportBoundary) {
    const float h = 0.1f;
    const float edge = h * 0.9999f;
    const float relative = 1.0e-3f;

    EXPECT_LT(cps::poly6_kernel(edge * edge, h), cps::poly6_kernel(0.0f, h) * relative);
    EXPECT_LT(cps::spiky_kernel(edge, h), cps::spiky_kernel(0.0f, h) * relative);
    EXPECT_LT(cps::spiky_gradient_magnitude(edge, h),
              cps::spiky_gradient_magnitude(0.5f * h, h) * relative);
    EXPECT_LT(cps::viscosity_laplacian(edge, h), cps::viscosity_laplacian(0.0f, h) * relative);

    // And they are exactly zero once past it, so the transition is continuous
    // rather than merely small.
    EXPECT_FLOAT_EQ(cps::poly6_kernel(h * h, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::spiky_kernel(h, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::spiky_gradient_magnitude(h, h), 0.0f);
    EXPECT_FLOAT_EQ(cps::viscosity_laplacian(h, h), 0.0f);
}

TEST(SphKernels, Poly6DecreasesMonotonicallyWithSeparation) {
    const float h = 0.1f;
    float previous = cps::poly6_kernel(0.0f, h);
    for (int i = 1; i <= 100; ++i) {
        const float r = static_cast<float>(i) * h / 100.0f;
        const float value = cps::poly6_kernel(r * r, h);
        EXPECT_LE(value, previous);
        previous = value;
    }
}

// This is the property poly6 does not have and the reason a second kernel
// exists at all. A vanishing gradient at contact means no restoring force
// exactly where the fluid needs one most, and particles pile up on top of one
// another.
TEST(SphKernels, SpikyGradientStaysFiniteAndLargeNearContact) {
    const float h = 0.1f;
    const float near_contact = cps::spiky_gradient_magnitude(1.0e-4f * h, h);
    const float mid = cps::spiky_gradient_magnitude(0.5f * h, h);
    EXPECT_GT(near_contact, 0.0f);
    EXPECT_GT(near_contact, mid);
    EXPECT_TRUE(std::isfinite(near_contact));

    // The analytic limit as the separation goes to zero is 45 / (pi h^4).
    const float expected = 45.0f / (cps::PI_F * h * h * h * h);
    EXPECT_NEAR(near_contact, expected, expected * 1.0e-3f);
}

TEST(SphKernels, ViscosityLaplacianIsNonNegativeInsideTheSupport) {
    const float h = 0.1f;
    for (int i = 0; i < 100; ++i) {
        const float r = static_cast<float>(i) * h / 100.0f;
        EXPECT_GE(cps::viscosity_laplacian(r, h), 0.0f);
    }
}

TEST(SphKernels, EquationOfStateIsClampedBelowRestDensity) {
    EXPECT_FLOAT_EQ(cps::equation_of_state(900.0f, 1000.0f, 50.0f), 0.0f);
    EXPECT_FLOAT_EQ(cps::equation_of_state(1000.0f, 1000.0f, 50.0f), 0.0f);
    EXPECT_FLOAT_EQ(cps::equation_of_state(1010.0f, 1000.0f, 50.0f), 500.0f);
}

TEST(SphKernels, KernelScalesAsTheInverseCubeOfTheSmoothingRadius) {
    // W(0, h) for poly6 is 315 / (64 pi h^3). Doubling h must divide the peak
    // by eight, which is the check that catches a mistyped exponent in the
    // normalisation constant.
    const float h = 0.1f;
    const float peak = cps::poly6_kernel(0.0f, h);
    const float peak_double = cps::poly6_kernel(0.0f, 2.0f * h);
    EXPECT_NEAR(peak / peak_double, 8.0f, 1.0e-3f);

    const float expected = 315.0f / (64.0f * cps::PI_F * h * h * h);
    EXPECT_NEAR(peak, expected, expected * 1.0e-5f);
}

}  // namespace
