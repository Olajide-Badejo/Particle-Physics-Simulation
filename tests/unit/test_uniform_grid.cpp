#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/physics/uniform_grid.hpp"

namespace {

// A grid from the origin to the given far corner.
cps::UniformGrid grid_to(float x, float y, float z, float support) {
    return cps::make_uniform_grid(0.0f, 0.0f, 0.0f, x, y, z, support);
}

// A grid of exactly these counts with a unit cell, which make_uniform_grid would
// not give: its margin takes a cell off a domain that is a whole number of supports.
cps::UniformGrid grid_of(int nx, int ny, int nz) {
    cps::UniformGrid grid;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.cells_per_layer = static_cast<uint32_t>(nx) * static_cast<uint32_t>(ny);
    grid.num_cells = grid.cells_per_layer * static_cast<uint32_t>(nz);
    return grid;
}

// The middle of a cell of a grid_of grid.
float centre(int cell) { return static_cast<float>(cell) + 0.5f; }

TEST(UniformGrid, TheDamBreakDomainGivesTheDerivedCellCount) {
    const cps::UniformGrid grid = grid_to(0.8f, 0.5f, 0.4f, 1.973e-2f);
    EXPECT_EQ(grid.nx, 41);
    EXPECT_EQ(grid.ny, 26);
    EXPECT_EQ(grid.nz, 21);
    EXPECT_EQ(grid.cells_per_layer, 41u * 26u);
    EXPECT_EQ(grid.num_cells, 22386u);
}

TEST(UniformGrid, EveryCellIsAtLeastTheSupportAndTheCellsCoverTheDomain) {
    const float domains[][3] = {{0.8f, 0.5f, 0.4f}, {1.0f, 1.0f, 1.0f}, {3.0f, 0.02f, 7.0f},
                                {0.04f, 0.04f, 0.04f}, {100.0f, 0.1f, 0.1f}};
    for (const float support : {0.01f, 0.01973f, 0.02f, 0.037f, 0.1f, 1.0f / 3.0f}) {
        for (const auto& domain : domains) {
            const cps::UniformGrid grid = grid_to(domain[0], domain[1], domain[2], support);
            EXPECT_GE(grid.cell_size, support);
            EXPECT_GE(static_cast<double>(grid.nx) * grid.cell_size, domain[0]);
            EXPECT_GE(static_cast<double>(grid.ny) * grid.cell_size, domain[1]);
            EXPECT_GE(static_cast<double>(grid.nz) * grid.cell_size, domain[2]);
            // Not a cell more than that needs.
            EXPECT_TRUE(grid.nx == 1 ||
                        static_cast<double>(grid.nx - 1) * grid.cell_size < domain[0]);
            EXPECT_TRUE(grid.ny == 1 ||
                        static_cast<double>(grid.ny - 1) * grid.cell_size < domain[1]);
            EXPECT_TRUE(grid.nz == 1 ||
                        static_cast<double>(grid.nz - 1) * grid.cell_size < domain[2]);
        }
    }
}

// A domain narrower than one support on an axis still has one cell on it.
TEST(UniformGrid, AnAxisNarrowerThanTheSupportHasOneCell) {
    const cps::UniformGrid grid = grid_to(0.05f, 1.0f, 0.0f, 0.1f);
    EXPECT_EQ(grid.nx, 1);
    EXPECT_GT(grid.ny, 1);
    EXPECT_EQ(grid.nz, 1);
    EXPECT_EQ(grid.num_cells, static_cast<uint32_t>(grid.ny));
}

TEST(UniformGrid, TheIndexHasXFastestThenYThenZ) {
    const cps::UniformGrid grid = grid_of(5, 4, 3);
    EXPECT_EQ(cps::cell_index_of(grid, 0, 0, 0), 0u);
    EXPECT_EQ(cps::cell_index_of(grid, 1, 0, 0), 1u);
    EXPECT_EQ(cps::cell_index_of(grid, 0, 1, 0), 5u);
    EXPECT_EQ(cps::cell_index_of(grid, 0, 0, 1), 20u);
    EXPECT_EQ(cps::cell_index_of(grid, 4, 3, 2), 59u);
    EXPECT_EQ(cps::row_start(grid, 2, 1), 30u);
}

// A unique index is the whole of the double counting argument, and walking the cells
// in row order has to count straight up, or a row would not be contiguous once sorted.
TEST(UniformGrid, TheIndexIsUniqueAndCountsUpInRowOrder) {
    const cps::UniformGrid grid = grid_of(7, 5, 6);
    uint32_t expected = 0u;
    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int ix = 0; ix < grid.nx; ++ix) {
                ASSERT_EQ(cps::cell_index_of(grid, ix, iy, iz), expected);
                ++expected;
            }
        }
    }
    EXPECT_EQ(expected, grid.num_cells);
}

// The furthest corners of the largest grid allowed, where a signed or a narrower
// product would have wrapped.
TEST(UniformGrid, TheIndexHoldsAtTheCornersOfTheLargestGrid) {
    const cps::UniformGrid grid = grid_of(1 << 11, 1 << 10, 1 << 10);
    EXPECT_EQ(static_cast<uint64_t>(grid.num_cells), cps::SPH_GRID_CELL_LIMIT);
    EXPECT_EQ(cps::cell_index_of(grid, grid.nx - 1, grid.ny - 1, grid.nz - 1), grid.num_cells - 1u);
    EXPECT_EQ(cps::cell_index_of(grid, 0, 0, grid.nz - 1), grid.num_cells - grid.cells_per_layer);
    EXPECT_EQ(cps::cell_index_of(grid, grid.nx - 1, 0, 0), static_cast<uint32_t>(grid.nx) - 1u);
    EXPECT_EQ(cps::reserved_slot(grid), 2147483648u);
}

TEST(UniformGrid, AFinitePositionIsAlwaysInsideTheTable) {
    const cps::UniformGrid grid =
        cps::make_uniform_grid(-1.0f, -2.0f, 0.5f, 1.0f, 3.0f, 2.0f, 0.07f);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> far(-50.0f, 50.0f);
    for (int trial = 0; trial < 20000; ++trial) {
        const float x = far(rng);
        const float y = far(rng);
        const float z = far(rng);
        EXPECT_LT(cps::cell_index(grid, x, y, z), grid.num_cells);
        EXPECT_EQ(cps::cell_index(grid, x, y, z), cps::cell_index(grid, x, y, z));
    }
}

// Truncation towards zero would fold the cells on either side of the origin onto each
// other, so every particle just below it would look like a neighbour of every particle
// just above it. Here the coordinate counts from the wall, so zero is nothing special,
// and neither side of it can land on the other.
TEST(UniformGrid, AxisCellIsNotFoldedAboutZero) {
    const float cell = 0.1f;
    const float inverse = 1.0f / cell;
    const float origin = -1.0f;
    const int cells = 20;
    EXPECT_EQ(cps::axis_cell(0.05f, origin, inverse, cells), 10);
    EXPECT_EQ(cps::axis_cell(0.15f, origin, inverse, cells), 11);
    EXPECT_EQ(cps::axis_cell(-0.05f, origin, inverse, cells), 9);
    EXPECT_EQ(cps::axis_cell(-0.15f, origin, inverse, cells), 8);
    EXPECT_NE(cps::axis_cell(-0.05f, origin, inverse, cells),
              cps::axis_cell(0.05f, origin, inverse, cells));
}

TEST(UniformGrid, AxisCellIsConsistentAcrossACellBoundary) {
    const float inverse = 4.0f;
    EXPECT_EQ(cps::axis_cell(0.2499f, 0.0f, inverse, 8), 0);
    EXPECT_EQ(cps::axis_cell(0.2501f, 0.0f, inverse, 8), 1);
}

TEST(UniformGrid, PositionsAtAndPastAWallClampToTheEdgeCell) {
    const float max = std::numeric_limits<float>::max();
    const float inverse = 10.0f;
    // On the far wall of a domain a whole number of cells wide, the coordinate is
    // one past the last cell.
    EXPECT_EQ(cps::axis_cell(1.0f, 0.0f, inverse, 10), 9);
    EXPECT_EQ(cps::axis_cell(0.0f, 0.0f, inverse, 10), 0);
    for (const float past : {1.5f, 1000.0f, 1.0e30f, max}) {
        EXPECT_EQ(cps::axis_cell(past, 0.0f, inverse, 10), 9) << past;
        EXPECT_EQ(cps::axis_cell(-past, 0.0f, inverse, 10), 0) << past;
    }
    EXPECT_EQ(cps::axis_cell(-std::numeric_limits<float>::min(), 0.0f, inverse, 10), 0);
    // One cell wide, where both walls are the same cell.
    EXPECT_EQ(cps::axis_cell(0.5f, 0.0f, inverse, 1), 0);
    EXPECT_EQ(cps::axis_cell(max, 0.0f, inverse, 1), 0);
    EXPECT_EQ(cps::axis_cell(-max, 0.0f, inverse, 1), 0);
}

struct Visited {
    std::vector<uint32_t> cells;
    uint32_t last_entry_read = 0u;
};

// What the search kernels do with a position, with the table left out: walks its
// rows, checks each is contiguous and stays on its own row, and lists the cells.
Visited walk_rows(const cps::UniformGrid& grid, float x, float y, float z) {
    Visited visited;
    const cps::NeighbourRows rows = cps::neighbour_rows(grid, x, y, z);
    for (int iz = rows.z_low; iz <= rows.z_high; ++iz) {
        for (int iy = rows.y_low; iy <= rows.y_high; ++iy) {
            const uint32_t row = cps::row_start(grid, iy, iz);
            EXPECT_GE(rows.x_low, 0);
            EXPECT_LT(rows.x_high, grid.nx);
            EXPECT_GE(iy, 0);
            EXPECT_LT(iy, grid.ny);
            EXPECT_GE(iz, 0);
            EXPECT_LT(iz, grid.nz);
            for (int ix = rows.x_low; ix <= rows.x_high; ++ix) {
                EXPECT_EQ(row + static_cast<uint32_t>(ix), cps::cell_index_of(grid, ix, iy, iz));
                visited.cells.push_back(row + static_cast<uint32_t>(ix));
            }
            visited.last_entry_read = std::max(visited.last_entry_read,
                                               row + static_cast<uint32_t>(rows.x_high) + 1u);
        }
    }
    return visited;
}

// The correctness of the search in one test: around every cell of a grid, the rows
// visit exactly the cells within one of it on every axis, with each edge cut off, each
// cell once. This runs at every shape of edge, a one cell wide grid on each axis among
// them, since an edge is where a range could run on into the next row.
TEST(UniformGrid, TheRowsVisitExactlyTheCellsWithinOneAndEachOnce) {
    const int shapes[][3] = {{5, 4, 3}, {1, 4, 3}, {5, 1, 3}, {5, 4, 1},
                             {1, 1, 3}, {1, 4, 1}, {5, 1, 1}, {1, 1, 1}, {2, 2, 2}};
    for (const auto& shape : shapes) {
        const cps::UniformGrid grid = grid_of(shape[0], shape[1], shape[2]);
        SCOPED_TRACE(std::to_string(grid.nx) + " x " + std::to_string(grid.ny) + " x " +
                     std::to_string(grid.nz));
        for (int iz = 0; iz < grid.nz; ++iz) {
            for (int iy = 0; iy < grid.ny; ++iy) {
                for (int ix = 0; ix < grid.nx; ++ix) {
                    std::set<uint32_t> expected;
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int nx = ix + dx;
                                const int ny = iy + dy;
                                const int nz = iz + dz;
                                if (nx >= 0 && nx < grid.nx && ny >= 0 && ny < grid.ny && nz >= 0 &&
                                    nz < grid.nz) {
                                    expected.insert(cps::cell_index_of(grid, nx, ny, nz));
                                }
                            }
                        }
                    }
                    const Visited visited = walk_rows(grid, centre(ix), centre(iy), centre(iz));
                    const std::set<uint32_t> unique(visited.cells.begin(), visited.cells.end());
                    EXPECT_EQ(unique.size(), visited.cells.size()) << "a cell was visited twice";
                    EXPECT_EQ(unique, expected);
                    // The table's last entry is read as an end, and never past that.
                    EXPECT_LE(visited.last_entry_read, grid.num_cells);
                }
            }
        }
    }
}

TEST(UniformGrid, TheCellsAroundAnInteriorPositionAreTwentySevenDistinctOnes) {
    const cps::UniformGrid grid = grid_of(9, 9, 9);
    const Visited visited = walk_rows(grid, centre(4), centre(4), centre(4));
    EXPECT_EQ(visited.cells.size(), 27u);
    EXPECT_EQ(std::set<uint32_t>(visited.cells.begin(), visited.cells.end()).size(), 27u);
}

// The furthest float from a along one axis that the search kernels' own distance test
// still accepts, on the side of a that target is on. Rounding is monotonic, so a
// bisection over the floats finds it, however closely they are spaced near zero.
float furthest_accepted(float a, float target, float support) {
    const float h_sq = support * support;
    const auto accepted = [&](float b) {
        const float r = a - b;
        return r * r < h_sq;
    };
    float inside = a;
    float outside = a + 2.0f * (target - a);
    while (true) {
        const float middle =
            static_cast<float>(0.5 * (static_cast<double>(inside) + static_cast<double>(outside)));
        if (middle == inside || middle == outside) {
            return inside;
        }
        (accepted(middle) ? inside : outside) = middle;
    }
}

struct PairCount {
    int pairs = 0;
    int too_far_apart = 0;
};

// Walks pairs just inside the support whose first particle sits within a few floats
// of a cell boundary, which is where rounding can carry a coordinate over one, and
// counts how many come out more than a cell apart.
PairCount scan_cell_boundaries(float origin, float inverse, int cells, float cell_size,
                               float support) {
    PairCount count;
    for (int k = 1; k < cells; ++k) {
        const float boundary = origin + static_cast<float>(k) * cell_size;
        float a = boundary;
        for (int step = 0; step < 16; ++step) {
            a = std::nextafter(a, origin);
        }
        for (int step = 0; step < 33; ++step, a = std::nextafter(a, boundary + cell_size)) {
            for (const float target : {a + support, a - support}) {
                const float b = furthest_accepted(a, target, support);
                const int ca = cps::axis_cell(a, origin, inverse, cells);
                const int cb = cps::axis_cell(b, origin, inverse, cells);
                ++count.pairs;
                if (std::abs(ca - cb) > 1) {
                    ++count.too_far_apart;
                }
            }
        }
    }
    return count;
}

// A cell of exactly the support does lose a neighbour, if only just: this finds a
// pair the distance test accepts and the twenty seven cells around one of them miss.
// The margin in make_uniform_grid is there for exactly these pairs, and with it, none
// of them comes out more than one cell apart.
TEST(UniformGrid, RoundingCannotPutTwoNeighboursMoreThanOneCellApart) {
    int lost_with_no_margin = 0;
    for (const float support : {0.01973f, 0.01f, 0.037f, 0.1f, 1.0f / 3.0f}) {
        for (const float origin : {0.0f, -1.0f, 0.3f}) {
            SCOPED_TRACE(support);
            SCOPED_TRACE(origin);
            const float length = 200.0f * support;
            const cps::UniformGrid grid =
                cps::make_uniform_grid(origin, origin, origin, origin + length, origin + length,
                                       origin + length, support);
            const PairCount with_margin = scan_cell_boundaries(
                origin, grid.inverse_cell_size, grid.nx, grid.cell_size, support);
            EXPECT_GT(with_margin.pairs, 10000);
            EXPECT_EQ(with_margin.too_far_apart, 0);

            const PairCount no_margin =
                scan_cell_boundaries(origin, 1.0f / support, grid.nx, support, support);
            lost_with_no_margin += no_margin.too_far_apart;
        }
    }
    EXPECT_GT(lost_with_no_margin, 0) << "the margin no longer has anything to protect against";
}

// The largest grid the axis limit allows, where rounding is worst: the margin is
// a large fraction of a cell there, and random pairs still stay within one cell.
TEST(UniformGrid, TheMarginStillCoversTheSupportAtTheAxisLimit) {
    const float support = 1.0f;
    const float length = std::nextafter(static_cast<float>(cps::SPH_GRID_AXIS_LIMIT), 0.0f);
    const cps::UniformGrid grid = grid_to(length, 1.0f, 1.0f, support);
    EXPECT_GE(grid.cell_size, support);
    EXPECT_LE(grid.cell_size, 1.6f * support);

    std::mt19937 rng(11u);
    std::uniform_real_distribution<float> along(0.0f, length);
    for (int trial = 0; trial < 200000; ++trial) {
        const float a = along(rng);
        const float b = furthest_accepted(a, a + support, support);
        const int ca = cps::axis_cell(a, 0.0f, grid.inverse_cell_size, grid.nx);
        const int cb = cps::axis_cell(b, 0.0f, grid.inverse_cell_size, grid.nx);
        ASSERT_LE(std::abs(ca - cb), 1) << "at " << a << " and " << b;
    }
}

TEST(UniformGrid, FiniteTestSeparatesNaNAndInfinityFromEveryFiniteValue) {
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

// The reserved slot is one past the last cell, so no finite position can get it and
// no search can scan it, wherever the search starts.
TEST(UniformGrid, TheReservedSlotIsOutOfReachOfEverySearch) {
    const cps::UniformGrid grid = grid_of(5, 4, 3);
    EXPECT_EQ(cps::reserved_slot(grid), grid.num_cells);

    // The last cell of the grid is the closest a search comes, and it reads the slot's
    // table entry only as where its range ends.
    const Visited corner =
        walk_rows(grid, centre(grid.nx - 1), centre(grid.ny - 1), centre(grid.nz - 1));
    EXPECT_EQ(*std::max_element(corner.cells.begin(), corner.cells.end()), grid.num_cells - 1u);
    EXPECT_EQ(corner.last_entry_read, grid.num_cells);

    const float max = std::numeric_limits<float>::max();
    EXPECT_EQ(cps::cell_index(grid, max, max, max), grid.num_cells - 1u);
    EXPECT_EQ(cps::cell_index(grid, -max, -max, -max), 0u);
}

// A non finite position searches nothing, on whichever axis it is bad.
TEST(UniformGrid, ANonFinitePositionHasNoRowsToSearch) {
    const cps::UniformGrid grid = grid_of(5, 4, 3);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const float bad : {nan, inf, -inf}) {
        SCOPED_TRACE(bad);
        EXPECT_TRUE(walk_rows(grid, bad, 1.0f, 1.0f).cells.empty());
        EXPECT_TRUE(walk_rows(grid, 1.0f, bad, 1.0f).cells.empty());
        EXPECT_TRUE(walk_rows(grid, 1.0f, 1.0f, bad).cells.empty());
    }
}

// Every key has to survive the sort's bit range, and the reserved slot is the largest
// key. At a power of two it is the only one with its top bit set, so a bit range that
// only counted the real cells would drop that bit and file the slot under cell zero.
TEST(UniformGrid, TheSortBitRangeHoldsEveryCellAndTheReservedSlot) {
    EXPECT_EQ(cps::sort_key_bits(1u), 1);
    EXPECT_EQ(cps::sort_key_bits(2u), 2);
    EXPECT_EQ(cps::sort_key_bits(3u), 2);
    EXPECT_EQ(cps::sort_key_bits(4u), 3);
    EXPECT_EQ(cps::sort_key_bits(4095u), 12);
    EXPECT_EQ(cps::sort_key_bits(4096u), 13);
    EXPECT_EQ(cps::sort_key_bits(22386u), 15);
    EXPECT_EQ(cps::sort_key_bits(2147483647u), 31);
    EXPECT_EQ(cps::sort_key_bits(2147483648u), 32);

    for (int shift = 0; shift < 31; ++shift) {
        for (const int offset : {-1, 0, 1}) {
            const int64_t cells = (int64_t{1} << shift) + offset;
            if (cells < 1) {
                continue;
            }
            const int bits = cps::sort_key_bits(static_cast<uint32_t>(cells));
            // The reserved slot fits in the bits sorted, and they are the fewest that do.
            EXPECT_EQ(static_cast<uint64_t>(cells) >> bits, 0u) << cells;
            EXPECT_NE(static_cast<uint64_t>(cells) >> (bits - 1), 0u) << cells;
        }
    }
}

TEST(UniformGrid, TheTableHasAnEntryPerCellAndOneForTheReservedSlot) {
    EXPECT_EQ(cps::cell_table_entries(22386u), 22387u);
    EXPECT_EQ(cps::cell_table_bytes(22386u), 22387u * 4u);
    EXPECT_EQ(cps::cell_table_bytes(cps::SPH_GRID_CELL_LIMIT), 8589934596u);
}

// The message of the error the check has to raise.
std::string size_error(int nx, int ny, int nz) {
    try {
        cps::check_sph_grid_size(nx, ny, nz);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "expected the size check to throw";
    return "";
}

TEST(UniformGrid, TheCellLimitIsInclusiveAndTheErrorNamesTheGridAndItsMemory) {
    EXPECT_NO_THROW(cps::check_sph_grid_size(1 << 11, 1 << 10, 1 << 10));
    EXPECT_NO_THROW(cps::check_sph_grid_size(1, 1, 1));
    EXPECT_NO_THROW(cps::check_sph_grid_size(1290, 1290, 1290));

    const std::string one_over = size_error((1 << 11) + 1, 1 << 10, 1 << 10);
    EXPECT_NE(one_over.find("too large"), std::string::npos) << one_over;
    EXPECT_NE(one_over.find("2049 x 1024 x 1024 cells"), std::string::npos) << one_over;
    EXPECT_NE(one_over.find("2148532224 in all"), std::string::npos) << one_over;
    EXPECT_NE(one_over.find("8196.000 MiB"), std::string::npos) << one_over;
    EXPECT_NE(one_over.find("2147483648"), std::string::npos) << one_over;
    EXPECT_NE(one_over.find("raise the smoothing radius or shrink the domain"), std::string::npos)
        << one_over;

    // Over the limit on each axis in turn, and where 32 bits alone would have wrapped
    // the count back to something small.
    EXPECT_FALSE(size_error(1291, 1290, 1290).empty());
    EXPECT_FALSE(size_error(1290, 1291, 1290).empty());
    EXPECT_FALSE(size_error(1290, 1290, 1291).empty());
    EXPECT_NE(size_error(1 << 16, 1 << 16, 1).find("65536 x 65536 x 1 cells"), std::string::npos);
    EXPECT_NE(size_error(1 << 20, 1 << 20, 1 << 20).find("1048576 x 1048576 x 1048576 cells"),
              std::string::npos);
}

TEST(UniformGrid, TheSolverSideOfTheCheckDerivesAndRefusesTheGrid) {
    const cps::UniformGrid accepted =
        cps::make_sph_grid({0.0f, 0.0f, 0.0f}, {0.8f, 0.5f, 0.4f}, 1.973e-2f);
    EXPECT_EQ(accepted.num_cells, 22386u);
    EXPECT_EQ(cps::sph_grid_description(accepted).find("41 x 26 x 21 cells, 22386 in all"), 0u);

    // Every axis is well inside its own limit, and the product is not.
    try {
        cps::make_sph_grid({0.0f, 0.0f, 0.0f}, {3000.0f, 3000.0f, 3000.0f}, 1.0f);
        ADD_FAILURE() << "expected the grid to be refused";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("too large"), std::string::npos) << message;
        EXPECT_NE(message.find(" x "), std::string::npos) << message;
        EXPECT_NE(message.find("MiB"), std::string::npos) << message;
    }
}

}  // namespace
