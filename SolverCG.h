#pragma once

#include <mpi.h>

/**
 * @brief The SolverCG class for solving linear systems using the Conjugate Gradient method.
 */
class SolverCG
{
public:
    /**
     * @brief Constructor for the SolverCG class.
     * @param Nx2 Number of grid points in the x-direction.
     * @param Ny2 Number of grid points in the y-direction.
     * @param dx2 Grid spacing in the x-direction.
     * @param dy2 Grid spacing in the y-direction.
     * @param coords2 Coordinates of the process in the MPI grid.
     * @param p_root2 Square root of the total number of processes.
     * @param rank2 Rank of the current MPI process.
     * @param grid The MPI communicator for the grid.
     */
    SolverCG(int Nx2, int Ny2, double dx2, double dy2, int* coords2, int p_root2, int rank2, MPI_Comm grid);
    
    /**
     * @brief Destructor for the SolverCG class.
     */
    ~SolverCG();

    /**
     * @brief Solve the linear system using the Conjugate Gradient method.
     * @param b The right-hand side vector.
     * @param x The solution vector.
     */
    void Solve(double* b, double* x);

private:
    double dx; /**< Grid spacing in the x-direction. */
    double dy; /**< Grid spacing in the y-direction. */
    int Nx; /**< Number of grid points in the x-direction. */
    int Ny; /**< Number of grid points in the y-direction. */
    unsigned int n; /**< Total number of grid points. */
    double* r; /**< Residual vector. */
    double* p; /**< Conjugate direction vector. */
    double* z; /**< Preconditioned vector. */
    double* t; /**< Temporary vector. */
    int coords[2]; /**< Coordinates of the process in the MPI grid. */
    int p_root; /**< Square root of the total number of processes. */
    int rank; /**< Rank of the current MPI process. */
    int world_size; /**< Total number of processes. */
    MPI_Comm mygrid; /**< MPI communicator for the grid. */
    double nu   = 0.1; /**< Viscosity. */

    /**
     * @brief Apply preconditioning to the input vector.
     * @param p Input vector.
     * @param t Temporary vector for storing the result.
     */
    void Precondition(double* p, double* t);

    /**
     * @brief Impose boundary conditions on the input vector.
     * @param p Vector on which to impose boundary conditions.
     */
    void ImposeBC(double* p);

    /**
     * @brief Apply the operator to the input vector.
     * @param p Input vector.
     * @param t Temporary vector for storing the result.
     */
    void ApplyOperator(double* p, double* t);

    /**
     * @brief Exchange data with neighboring processes to maintain consistency.
     * @param data The data array to be exchanged with neighbors.
     */
    void ExchangeDataWithNeighbors(double* data);
};
