#pragma once

#include <cmath>
#include <cstdint>

#include "cps/physics/hd_macros.hpp"

namespace cps {

// Dense uniform grid for neighbour search.
//
// Every fluid domain here is bounded, so a cell can be named by where it sits in
// the grid, with x varying fastest:
//
//     index = ix + nx * (iy + ny * iz)
//
// Two different cells can never share an index, so the neighbour search needs no
// check of which cell a candidate belongs to, and no two cells it visits can count
// a particle twice. Sorting by this index also lays each x row of cells out
// contiguously, which is what lets the search read three cells as one range.
struct UniformGrid {
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float cell_size = 1.0f;
    float inverse_cell_size = 1.0f;
    int nx = 1;
    int ny = 1;
    int nz = 1;
    uint32_t cells_per_layer = 1u;  // nx * ny
    uint32_t num_cells = 1u;        // nx * ny * nz
};

// The cell one axis puts a position in. A position past either wall lands in the
// edge cell, and a value too large to convert to an int is clamped before it gets
// that far. Clamping cannot separate two neighbours, since it never moves two
// coordinates further apart than they already were.
//
// This is the hook for periodic boundaries: a periodic axis would wrap the
// coordinate here instead of clamping it, and neighbour_rows would have to split a
// row that crosses the seam. Neither is implemented, because every domain has walls.
CPS_HD CPS_INLINE int axis_cell(float position, float origin, float inverse_cell_size, int cells) {
    const float coordinate = (position - origin) * inverse_cell_size;
    const float top = static_cast<float>(cells - 1);
    const float above_zero = coordinate > 0.0f ? coordinate : 0.0f;
    // Not negative from here on, so truncating is the same as rounding down.
    return static_cast<int>(above_zero < top ? above_zero : top);
}

CPS_HD CPS_INLINE uint32_t cell_index_of(const UniformGrid& grid, int ix, int iy, int iz) {
    return static_cast<uint32_t>(ix) + static_cast<uint32_t>(iy) * static_cast<uint32_t>(grid.nx) +
           static_cast<uint32_t>(iz) * grid.cells_per_layer;
}

// The cell of a finite position. A non finite one gets reserved_slot instead.
CPS_HD CPS_INLINE uint32_t cell_index(const UniformGrid& grid, float x, float y, float z) {
    return cell_index_of(grid, axis_cell(x, grid.origin_x, grid.inverse_cell_size, grid.nx),
                         axis_cell(y, grid.origin_y, grid.inverse_cell_size, grid.ny),
                         axis_cell(z, grid.origin_z, grid.inverse_cell_size, grid.nz));
}

// Reads the exponent bits rather than calling isfinite, so the answer cannot
// depend on what a fast math build is allowed to assume about NaN.
CPS_HD CPS_INLINE bool is_finite_component(float value) {
    union {
        float as_float;
        uint32_t as_bits;
    } pun;
    pun.as_float = value;
    return (pun.as_bits & 0x7F800000u) != 0x7F800000u;
}

CPS_HD CPS_INLINE bool position_is_finite(float x, float y, float z) {
    return is_finite_component(x) && is_finite_component(y) && is_finite_component(z);
}

// The sort key of a NaN or Inf position, which has no real cell: one past the last
// cell, which no finite position can get. Every finite particle therefore sorts
// ahead of every non finite one. The cell table holds an entry for this slot, which
// closes the last real cell's range, and nothing ever scans from it, because
// neighbour_rows only names real cells. No search can read a non finite particle, so
// it cannot contaminate the particles around it.
CPS_HD CPS_INLINE uint32_t reserved_slot(const UniformGrid& grid) { return grid.num_cells; }

// The rows of cells around a position: every x row within one cell of it on y and
// z, each read as the range of cells x_low to x_high. Rows off the edge of the grid
// are left out and a row at a wall is cut short, so a range can never run on into
// the row next to it in the index. That is at most nine rows and twenty seven cells,
// and no cell turns up twice.
//
// A NaN or Inf position has no cell to search around, so its loops do not run. The
// test is on the bits, so that holds in a fast math build too.
struct NeighbourRows {
    int x_low;
    int x_high;
    int y_low;
    int y_high;
    int z_low;
    int z_high;
};

CPS_HD CPS_INLINE NeighbourRows neighbour_rows(const UniformGrid& grid, float x, float y, float z) {
    if (!position_is_finite(x, y, z)) {
        return NeighbourRows{0, 0, 0, -1, 0, -1};
    }
    const int cx = axis_cell(x, grid.origin_x, grid.inverse_cell_size, grid.nx);
    const int cy = axis_cell(y, grid.origin_y, grid.inverse_cell_size, grid.ny);
    const int cz = axis_cell(z, grid.origin_z, grid.inverse_cell_size, grid.nz);
    return NeighbourRows{cx > 0 ? cx - 1 : 0, cx + 1 < grid.nx ? cx + 1 : cx,
                         cy > 0 ? cy - 1 : 0, cy + 1 < grid.ny ? cy + 1 : cy,
                         cz > 0 ? cz - 1 : 0, cz + 1 < grid.nz ? cz + 1 : cz};
}

// The index of the first cell of a row. Adding x_low, and x_high + 1, gives the two
// entries of the cell table that bound the row's range.
CPS_HD CPS_INLINE uint32_t row_start(const UniformGrid& grid, int iy, int iz) {
    return cell_index_of(grid, 0, iy, iz);
}

// The number of low bits of a sort key that can differ, the reserved slot included.
// A grid of exactly 2^k cells needs k + 1 of them. Sorting only those is what keeps
// the radix sort's digit count down, and sorting one fewer would file the reserved
// slot in with cell zero.
inline int sort_key_bits(uint32_t num_cells) {
    int bits = 0;
    for (uint32_t value = num_cells; value > 0u; value >>= 1) {
        ++bits;
    }
    return bits > 0 ? bits : 1;
}

// Entries in the cell table of a grid, and what they take in memory. In 64 bits, so
// that a grid the size check is there to refuse can still be described.
inline uint64_t cell_table_entries(uint64_t num_cells) { return num_cells + 1u; }
inline uint64_t cell_table_bytes(uint64_t num_cells) {
    return cell_table_entries(num_cells) * sizeof(uint32_t);
}

// The most cells a grid may have. The sort keys are 32 bits, and the reserved slot
// and the table's last entry both have to fit one.
inline constexpr uint64_t SPH_GRID_CELL_LIMIT = uint64_t{1} << 31;

// The limit on an axis, in widths of the support. A cell coordinate below 2^20 keeps
// three bits below the cell in single precision, and the margin below stays near half
// a cell, so the search still covers the support. A position inside the same range
// still resolves an eighth of the support.
inline constexpr int SPH_GRID_AXIS_LIMIT = 1 << 20;

// Builds a grid over the given bounds with cells at least as wide as the support, so
// that the twenty seven cells around a position cover every neighbour. Takes valid
// bounds: make_sph_grid is what checks them. The cell counts are always right, while
// the two products are only meaningful once the size check has passed.
//
// A cell of exactly the support is not quite enough. In exact arithmetic, two points
// less than one cell apart cannot have coordinates that round down two cells apart.
// A computed coordinate is rounded by up to 2^-24 relative in the subtraction and
// again in the product, which is up to widest * 2^-23 cells for each of the two
// particles, and the distance test can accept a separation about 2^-22 relative over
// the support. So two particles just inside the support can come out two cells apart,
// and the search around one would miss the other. A cell wider than the support by
// (widest + 2) * 2^-21 of it is more than all of that added up, for any grid the axis
// limit allows, which is about 2e-5 of a cell in a grid 41 cells wide.
inline UniformGrid make_uniform_grid(float min_x, float min_y, float min_z, float max_x,
                                     float max_y, float max_z, float support) {
    const double h = static_cast<double>(support);
    const double extent_x = static_cast<double>(max_x) - static_cast<double>(min_x);
    const double extent_y = static_cast<double>(max_y) - static_cast<double>(min_y);
    const double extent_z = static_cast<double>(max_z) - static_cast<double>(min_z);
    const double longest = std::fmax(extent_x, std::fmax(extent_y, extent_z));
    const double widest = std::ceil(longest / h);
    const float cell_size = static_cast<float>(h * (1.0 + (widest + 2.0) * std::ldexp(1.0, -21)));
    const double cell = static_cast<double>(cell_size);

    const auto cells_along = [cell](double extent) {
        const double cells = std::ceil(extent / cell);
        return cells > 1.0 ? static_cast<int>(cells) : 1;
    };

    UniformGrid grid;
    grid.origin_x = min_x;
    grid.origin_y = min_y;
    grid.origin_z = min_z;
    grid.cell_size = cell_size;
    grid.inverse_cell_size = static_cast<float>(1.0 / cell);
    grid.nx = cells_along(extent_x);
    grid.ny = cells_along(extent_y);
    grid.nz = cells_along(extent_z);
    grid.cells_per_layer = static_cast<uint32_t>(grid.nx) * static_cast<uint32_t>(grid.ny);
    grid.num_cells = grid.cells_per_layer * static_cast<uint32_t>(grid.nz);
    return grid;
}

}  // namespace cps
