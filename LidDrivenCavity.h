#pragma once

#include <string>
#include <mpi.h>
#include "SolverCG.h"

/**
 * @brief Class for simulating a lid-driven cavity flow using MPI.
 */
class LidDrivenCavity
{
public:
    /**
     * @brief Constructor for the LidDrivenCavity class.
     */
    LidDrivenCavity();

    /**
     * @brief Destructor for the LidDrivenCavity class.
     */
    ~LidDrivenCavity();

    /**
     * @brief Sets the domain size of the cavity.
     * @param xlen Length of the domain in the x-direction.
     * @param ylen Length of the domain in the y-direction.
     */
    void SetDomainSize(double xlen, double ylen);

    /**
     * @brief Sets the grid size for discretizing the domain.
     * @param nx Number of grid points in the x-direction.
     * @param ny Number of grid points in the y-direction.
     */
    void SetGridSize(int nx, int ny);

    /**
     * @brief Sets the time step for the simulation.
     * @param deltat Time step size.
     */
    void SetTimeStep(double deltat);

    /**
     * @brief Sets the final time for the simulation.
     * @param finalt Final time for the simulation.
     */
    void SetFinalTime(double finalt);

    /**
     * @brief Sets the Reynolds number for the simulation.
     * @param Re Reynolds number.
     */
    void SetReynoldsNumber(double Re);

    /**
     * @brief Initializes the simulation.
     */
    void Initialise();

    /**
     * @brief Integrates the simulation over time.
     */
    void Integrate();

    /**
     * @brief Writes the solution to a file.
     * @param file Name of the file to write the solution to.
     */
    void WriteSolution(std::string file);

    /**
     * @brief Prints the configuration parameters of the simulation.
     */
    void PrintConfiguration();

    /**
     * @brief Checks the MPI configuration.
     */
    void CheckMPI();

    //void WriteSolution(std::string file);
    // int CalculateStartIndex(int size, int coord, int root);
    // void CalculateVelocities(double* u0, double* u1);
    // void BuildDataString(std::stringstream& ss, int Nx_write, int Ny_write, int i_start_global, int j_start_global);
    // void GatherAndWriteData(std::stringstream& ss, const std::string& file);
    //void WriteToFile(const std::string& file, const std::string& data, MPI_Offset offset, int size);
    //void WriteToFile(const std::string& file, const std::string& data, MPI_Offset offset, int size, int rank);


private:
    double* v    = nullptr; /**< Vorticity vector. */
    double* vnew = nullptr; /**< Updated vorticity vector. */
    double* s    = nullptr; /**< Temporary storage vector. */
    double* tmp  = nullptr; /**< Temporary vector for calculations. */
    
    double dt   = 0.01; /**< Time step size. */
    double T    = 1.0;  /**< Final time for the simulation. */
    double dx;          /**< Grid spacing in the x-direction. */
    double dy;          /**< Grid spacing in the y-direction. */
    int    Nx   = 9;    /**< Number of grid points in the x-direction. */
    int    Ny   = 9;    /**< Number of grid points in the y-direction. */
    int    Npts = 81;   /**< Total number of grid points. */
    double Lx   = 1.0;  /**< Length of the domain in the x-direction. */
    double Ly   = 1.0;  /**< Length of the domain in the y-direction. */
    double Re   = 10;   /**< Reynolds number. */
    double U    = 1.0;  /**< Vorticity magnitude. */
    double nu   = 0.1;  /**< Viscosity. */
    double* u0;
    double* u1;

    int    rank;        /**< Rank of the current MPI process. */
    int    size;        /**< Total number of MPI processes. */
    int    p_root;      /**< Square root of the total number of MPI processes. */ 
    int    mygrid_rank; /**< Rank of the current MPI process in the grid communicator. */
    int    Nx_loc;      /**< Number of grid points in the x-direction for the local process. */
    int    Ny_loc;      /**< Number of grid points in the y-direction for the local process. */
    int    Npts_local;      /**< Total number of grid points for the local process. */
    int    coords[2];       /**< Coordinates of the current MPI process in the MPI grid. */
    int    i_start_global;  /**< Starting index in the global x-direction for the local process. */
    int    j_start_global;  /**< Starting index in the global y-direction for the local process. */
    MPI_Comm mygrid;        /**< MPI communicator for the grid. */

    SolverCG* cg = nullptr;

    /**
     * @brief Exchanges data with neighboring MPI processes.
     * @param data The data array to exchange.
     */
    void ExchangeDataWithNeighbors(double* data);

    /**
     * @brief Cleans up allocated memory.
     */
    void CleanUp();

    /**
     * @brief Updates the grid spacing values based on the current grid size.
     */
    void UpdateDxDy();

    /**
     * @brief Advances the simulation by one time step.
     * @param t Current time step.
     */
    void Advance(int t);
};

