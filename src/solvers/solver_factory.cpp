#include "cps/solvers/solver_factory.hpp"

#include <stdexcept>
#include <string>

#include "cps/solvers/nbody_solver.hpp"
#include "cps/solvers/sph_solver.hpp"

namespace cps {

std::unique_ptr<Solver> make_solver(const Config& config) {
    config.validate();
    switch (config.solver) {
        case SolverType::NBody:
            return std::make_unique<NBodySolver>(config);
        case SolverType::Sph:
            return std::make_unique<SphSolver>(config, config.num_particles);
    }
    throw std::runtime_error("unknown solver type " +
                             std::to_string(static_cast<int>(config.solver)) +
                             ", expected nbody or sph");
}

}  // namespace cps
