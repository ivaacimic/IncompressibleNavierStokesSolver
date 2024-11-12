/**
 * @file main.cpp
 * @brief Main function for solving the 2D lid-driven cavity incompressible flow problem.
 */

// Standard library includes
#include <iostream>
#include <mpi.h>
#include <omp.h>

// Namespace used
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

// Application-specific includes the following .h files
#include "LidDrivenCavity.h"

/**
 * @brief Main function for the solver.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Returns 0 upon successful execution.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Define command-line options
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    // Parse command-line arguments
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    // Display help message if requested
    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    // Create solver instance, check MPI and configure parameters
    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->CheckMPI();
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());

    // Print configuration details
    solver->PrintConfiguration();

    // Initialize solver
    solver->Initialise();

    // Write initial condition to file
    solver->WriteSolution("ic.txt");

    // Integrate the solution
    solver->Integrate();

    // Write final solution to file
    solver->WriteSolution("final.txt");

    MPI_Finalize();
    return 0;
}