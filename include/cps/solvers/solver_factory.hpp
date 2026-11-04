#pragma once

#include <memory>

#include "cps/core/config.hpp"
#include "cps/solvers/solver.hpp"

namespace cps {

// The solver a configuration asks for. This is the one place that branches on
// the solver type to build something, so adding a solver means adding a class and
// a case here, with no change to the engine.
//
// Throws std::runtime_error if the configuration does not validate, if the solver
// type is not one this build knows, or if the solver itself refuses it.
std::unique_ptr<Solver> make_solver(const Config& config);

}  // namespace cps
