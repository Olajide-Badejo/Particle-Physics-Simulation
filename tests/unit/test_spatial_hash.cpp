#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "physics/spatial_hash.hpp"

namespace {

TEST(SpatialHash, IsDeterministicForTheSameCell) {
    const uint32_t table = 100003u;
    for (int trial = 0; trial < 1000; ++trial) {
        const int ix = trial - 500;
        const int iy = 3 * trial - 71;
        const int iz = -2 * trial + 17;
        const uint32_t first = cps::spatial_hash(ix, iy, iz, table);
        const uint32_t second = cps::spatial_hash(ix, iy, iz, table);
        EXPECT_EQ(first, second);
    }
}

TEST(SpatialHash, StaysInsideTheTable) {
    const uint32_t table = 1543u;
    std::mt19937 rng(7u);
    for (int trial = 0; trial < 20000; ++trial) {
        const int ix = static_cast<int>(rng() % 4000) - 2000;
        const int iy = static_cast<int>(rng() % 4000) - 2000;
        const int iz = static_cast<int>(rng() % 4000) - 2000;
        EXPECT_LT(cps::spatial_hash(ix, iy, iz, table), table);
    }
}

// Truncation towards zero would fold the cell on either side of the origin
// onto each other, so every particle just below zero would look like a
// neighbour of every particle just above it.
TEST(SpatialHash, CellCoordinateFloorsRatherThanTruncates) {
    const float cell = 0.1f;
    EXPECT_EQ(cps::cell_coordinate(0.05f, cell), 0);
    EXPECT_EQ(cps::cell_coordinate(0.15f, cell), 1);
    EXPECT_EQ(cps::cell_coordinate(-0.05f, cell), -1);
    EXPECT_EQ(cps::cell_coordinate(-0.15f, cell), -2);
    EXPECT_NE(cps::cell_coordinate(-0.05f, cell), cps::cell_coordinate(0.05f, cell));
}

TEST(SpatialHash, CellCoordinateIsConsistentAcrossACellBoundary) {
    const float cell = 0.25f;
    EXPECT_EQ(cps::cell_coordinate(0.2499f, cell), 0);
    EXPECT_EQ(cps::cell_coordinate(0.2501f, cell), 1);
}

// The exact cell key is what protects the neighbour scan from hash collisions,
// so it has to be genuinely injective over the range the simulation uses.
TEST(SpatialHash, CellKeyIsUniqueOverAWideRange) {
    std::set<int64_t> seen;
    for (int ix = -30; ix <= 30; ++ix) {
        for (int iy = -30; iy <= 30; ++iy) {
            for (int iz = -30; iz <= 30; ++iz) {
                const int64_t key = cps::cell_key(ix, iy, iz);
                EXPECT_TRUE(seen.insert(key).second)
                    << "collision at " << ix << " " << iy << " " << iz;
            }
        }
    }
    EXPECT_EQ(seen.size(), 61u * 61u * 61u);
}

TEST(SpatialHash, CellKeyHandlesTheExtremesOfItsRange) {
    const int limit = (1 << 20) - 1;
    EXPECT_NE(cps::cell_key(limit, limit, limit), cps::cell_key(-limit, -limit, -limit));
    EXPECT_NE(cps::cell_key(limit, 0, 0), cps::cell_key(0, limit, 0));
    EXPECT_NE(cps::cell_key(0, 0, limit), cps::cell_key(0, limit, 0));
    EXPECT_EQ(cps::cell_key(0, 0, 0), cps::cell_key(0, 0, 0));
}

TEST(SpatialHash, NextPrimeReturnsAPrimeNotLessThanTheRequest) {
    const std::vector<uint32_t> requests = {1u, 2u, 3u, 4u, 10u, 100u, 1000u, 100000u, 1048576u};
    for (uint32_t request : requests) {
        const uint32_t prime = cps::next_prime(request);
        EXPECT_GE(prime, request);
        for (uint64_t d = 2; d * d <= prime; ++d) {
            EXPECT_NE(prime % static_cast<uint32_t>(d), 0u)
                << prime << " is divisible by " << d;
        }
    }
}

TEST(SpatialHash, NextPrimeIsExactOnKnownValues) {
    EXPECT_EQ(cps::next_prime(2u), 2u);
    EXPECT_EQ(cps::next_prime(3u), 3u);
    EXPECT_EQ(cps::next_prime(8u), 11u);
    EXPECT_EQ(cps::next_prime(100u), 101u);
}

// A hash that piles a fluid up into a handful of buckets turns the neighbour
// scan back into a linear search. This measures the occupancy the solver
// actually sees: a solid block of cells, table sized at twice the particle
// count, which is the arrangement the dam break starts from.
TEST(SpatialHash, SpreadsASolidBlockOfCellsAcrossTheTable) {
    const int side = 25;
    const int cell_count = side * side * side;
    const uint32_t table = cps::next_prime(static_cast<uint32_t>(2 * cell_count));

    std::map<uint32_t, int> occupancy;
    for (int ix = 0; ix < side; ++ix) {
        for (int iy = 0; iy < side; ++iy) {
            for (int iz = 0; iz < side; ++iz) {
                ++occupancy[cps::spatial_hash(ix, iy, iz, table)];
            }
        }
    }

    int worst = 0;
    for (const auto& entry : occupancy) {
        worst = std::max(worst, entry.second);
    }

    // With this load factor a well behaved hash leaves most buckets with one
    // cell and none with a long chain.
    const double distinct_fraction =
        static_cast<double>(occupancy.size()) / static_cast<double>(cell_count);
    EXPECT_GT(distinct_fraction, 0.75) << "too many cells share a bucket";
    EXPECT_LE(worst, 6) << "a bucket collected " << worst << " distinct cells";
}

TEST(SpatialHash, DistinctNeighbouringCellsGetDistinctKeys) {
    // The neighbour search visits twenty seven cells and relies on their keys
    // being different even when two of them collide in the table.
    std::set<int64_t> keys;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                keys.insert(cps::cell_key(100 + dx, -50 + dy, 7 + dz));
            }
        }
    }
    EXPECT_EQ(keys.size(), 27u);
}

TEST(SpatialHash, FiniteTestSeparatesNaNAndInfinityFromEveryFiniteValue) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const float bad : {nan, -nan, inf, -inf, std::numeric_limits<float>::signaling_NaN()}) {
        SCOPED_TRACE(bad);
        EXPECT_FALSE(cps::is_finite_component(bad));
        EXPECT_FALSE(cps::position_is_finite(bad, 0.0f, 0.0f));
        EXPECT_FALSE(cps::position_is_finite(0.0f, bad, 0.0f));
        EXPECT_FALSE(cps::position_is_finite(0.0f, 0.0f, bad));
    }
    for (const float good : {0.0f, -0.0f, 1.5f, -1.5f, std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::min(),
                             std::numeric_limits<float>::denorm_min()}) {
        SCOPED_TRACE(good);
        EXPECT_TRUE(cps::is_finite_component(good));
        EXPECT_TRUE(cps::position_is_finite(good, good, good));
    }
}

// A finite particle inside a domain the host check accepted must never be able
// to look in the sentinel cell: its coordinate reaches 2^20 - 3 at most, and
// the search around it 2^20 - 2. The sentinel is one past that, and every
// field of its key still fits in 21 bits.
TEST(SpatialHash, TheSentinelCellIsOutOfReachOfAnyAllowedSearch) {
    EXPECT_EQ(cps::NON_FINITE_CELL, cps::SPH_DOMAIN_CELL_LIMIT + 1);
    const int s = cps::NON_FINITE_CELL;
    EXPECT_EQ(cps::cell_key(s, s, s), std::numeric_limits<int64_t>::max());

    const float limit = static_cast<float>(cps::SPH_DOMAIN_CELL_LIMIT);
    const float highest_position = std::nextafter(limit, 0.0f);
    const int highest_cell = cps::cell_coordinate(highest_position, 1.0f);
    const int lowest_cell = cps::cell_coordinate(-highest_position, 1.0f);
    EXPECT_EQ(highest_cell, cps::SPH_DOMAIN_CELL_LIMIT - 1);
    EXPECT_EQ(lowest_cell, -cps::SPH_DOMAIN_CELL_LIMIT);
    EXPECT_LT(highest_cell + 1, s);

    // The extreme corners of the search keep their fields apart, so the
    // highest allowed key on each axis stays below the sentinel's and the
    // lowest stays positive.
    const int top = highest_cell + 1;
    const int bottom = lowest_cell - 1;
    EXPECT_LT(cps::cell_key(top, top, top), cps::cell_key(s, s, s));
    EXPECT_GT(cps::cell_key(bottom, bottom, bottom), 0);
    EXPECT_NE(cps::cell_key(top, bottom, bottom), cps::cell_key(bottom, top, bottom));
    EXPECT_NE(cps::cell_key(bottom, top, bottom), cps::cell_key(bottom, bottom, top));
}

}  // namespace
