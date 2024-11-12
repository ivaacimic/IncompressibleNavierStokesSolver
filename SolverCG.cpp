

// Standard library includes
#include <iostream>
#include <algorithm>
#include <cstring>

// Namespace used
using namespace std;

// External library includes for linear algebra operations, cmath, MPI, and OpenMP
#include <cblas.h>
#include <mpi.h>
#include <omp.h>
#include <cmath>

// Application-specific includes the following .h files
#include "SolverCG.h"

/**
 * @brief Computes the linear index of a 2D array given the row and column indices.
 * @param I Row index.
 * @param J Column index.
 * @return Linear index in a 1D array.
 */
#define IDX(I,J) ((J)*Nx + (I))

/**
 * @brief Constructor for the SolverCG class, initializes the conjugate gradient solver.
 * @param Nx2 Number of grid points in the x-direction.
 * @param Ny2 Number of grid points in the y-direction.
 * @param dx2 Grid spacing in the x-direction.
 * @param dy2 Grid spacing in the y-direction.
 * @param coords2 Coordinates of the process in the MPI grid.
 * @param p_root2 Square root of the total number of processes.
 * @param rank2 Rank of the current MPI process.
 * @param grid The MPI communicator for the grid.
 */
SolverCG::SolverCG(int Nx2, int Ny2, double dx2, double dy2, int* coords2, int p_root2, int rank2, MPI_Comm grid)
{
    dx = dx2;  // Grid spacing in the x-direction
    dy = dy2;  // Grid spacing in the y-direction
    Nx = Nx2;  // Number of grid points in the x-direction
    Ny = Ny2;  // Number of grid points in the y-direction
    n = Nx * Ny;  // Total number of grid points
    r = new double[n];  // Residual vector
    p = new double[n];  // Conjugate vector
    z = new double[n];  // Temporary vector
    t = new double[n];  // Temporary vector
    coords[0] = coords2[0];  // Coordinates of the process in the MPI grid (x-coordinate)
    coords[1] = coords2[1];  // Coordinates of the process in the MPI grid (y-coordinate)
    p_root = p_root2;  // Square root of the total number of processes
    world_size = p_root * p_root;  // Total number of processes
    rank = rank2;  // Rank of the current MPI process
    mygrid = grid;  // MPI communicator for the grid
}


/**
 * @brief Destructor for the SolverCG class, cleans up allocated memory.
 */
SolverCG::~SolverCG()
{
    delete[] r;  // Clean up memory allocated for the residual vector
    delete[] p;  // Clean up memory allocated for the conjugate vector
    delete[] z;  // Clean up memory allocated for the temporary vector
    delete[] t;  // Clean up memory allocated for the temporary vector
}


/**
 * @brief Impose boundary conditions on the input array.
 * @param inout Array on which to impose boundary conditions.
 */
void SolverCG::ImposeBC(double* inout) {
    bool check_left = (coords[0] == 0);  // Check if the process is on the left boundary
    bool check_right = (coords[0] == p_root - 1);  // Check if the process is on the right boundary
    bool check_top = (coords[1] == 0);  // Check if the process is on the top boundary
    bool check_bottom = (coords[1] == p_root - 1);  // Check if the process is on the bottom boundary

    // Using for loop first to utilize the OpenMP parallelization
    #pragma omp parallel for default(shared) schedule(static)
    for (int j = 0; j < Ny; ++j) {
        if (check_left) {
            inout[IDX(0, j)] = 0.0;  // Left boundary
        }
        if (check_right) {
            inout[IDX(Nx - 1, j)] = 0.0;  // Right boundary
        }
    }

    #pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < Nx; ++i) {
        if (check_top) {
            inout[IDX(i, 0)] = 0.0;  // Top boundary
        }
        if (check_bottom) {
            inout[IDX(i, Ny - 1)] = 0.0;  // Bottom boundary
        }
    }
}

/**
 * @brief Exchanges the boundary data across parallel processes to maintain data consistency.
 * @param data The data array to be updated across processes.
 */
void SolverCG::ExchangeDataWithNeighbors(double* data) {

    // Variables to store neighboring ranks
    int lrank;
    int rrank;
    int trank;
    int brank;

    // Arrays to store boundary data sent and received
    double *left_boundary_sent = new double[Ny];
    double *left_boundary_received = new double[Ny];
    double *right_boundary_sent = new double[Ny];
    double *right_boundary_received = new double[Ny];
    double *top_boundary_sent = new double[Nx];
    double *top_boundary_received = new double[Nx];
    double *bottom_boundary_sent = new double[Nx];
    double *bottom_boundary_received = new double[Nx];

    MPI_Request send_req[4]; // Array to store MPI requests
    int counter = 0; // Counter to keep track of the number of requests

    // Determine the neighboring ranks using MPI_Cart_shift
    MPI_Cart_shift(mygrid, 0, 1, &lrank, &rrank); // Shift along the x-axis
    MPI_Cart_shift(mygrid, 1, 1, &trank, &brank); // Shift along the y-axis

    // Determine if the process has neighboring processes in each directio
    bool hasNeighbor[4] = {coords[0] != 0, coords[0] != p_root -1, coords[1] != 0, coords[1] != p_root-1};

    // Send to the left, receive from the right
    if (hasNeighbor[0]) {
        
        // Sending left boundary to the neighboring process on the left
        #pragma omp parallel for
        for (int i = 0; i < Ny; ++i) {
            left_boundary_sent[i] = data[IDX(1, i)];
        }
        MPI_Isend(left_boundary_sent, Ny, MPI_DOUBLE, lrank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(left_boundary_received, Ny, MPI_DOUBLE, lrank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the left boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Ny; ++i) {
            data[IDX(0, i)] = left_boundary_received[i];
        }
    }

    // Send to the right, receive from the left
    if (hasNeighbor[1]) {
        // Sending right boundary to the neighboring process on the right
        #pragma omp parallel for
        for (int i = 0; i < Ny; ++i) {
            right_boundary_sent[i] = data[IDX(Nx-2, i)];
        }
        MPI_Isend(right_boundary_sent, Ny, MPI_DOUBLE, rrank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(right_boundary_received, Ny, MPI_DOUBLE, rrank, 0, mygrid, MPI_STATUS_IGNORE);  
    
        // Update the right boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Ny; ++i) {
            data[IDX(Nx-1, i)] = right_boundary_received[i];
        }      
    }

    // Send to the top, receive from the bottom
    if (hasNeighbor[2]) {
        // Sending top boundary to the neighboring process on the top
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            top_boundary_sent[i] = data[IDX(i, 1)];
        }
        MPI_Isend(top_boundary_sent, Nx, MPI_DOUBLE, trank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(top_boundary_received, Nx, MPI_DOUBLE, trank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the top boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            data[IDX(i, 0)] = top_boundary_received[i];
        }
    }

    // Send to the bottom, receive from the top
    if (hasNeighbor[3]) {
        
        // Sending bottom boundary to the neighboring process on the bottom
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            bottom_boundary_sent[i] = data[IDX(i, Ny-2)];
        }
        MPI_Isend(bottom_boundary_sent, Nx, MPI_DOUBLE, brank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(bottom_boundary_received, Nx, MPI_DOUBLE, brank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the bottom boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            data[IDX(i, Ny - 1)] = bottom_boundary_received[i];
        }
    }

    delete[] left_boundary_sent;
    delete[] left_boundary_received;
    delete[] right_boundary_sent;
    delete[] right_boundary_received;
    delete[] top_boundary_sent;
    delete[] top_boundary_received;
    delete[] bottom_boundary_sent;
    delete[] bottom_boundary_received;
}

/**
 * @brief Solves the system using the parallel conjugate gradient method.
 * @param b The right-hand side array.
 * @param x The solution array, which will be updated in-place.
 */
void SolverCG::Solve(double* b, double* x) {
    double tol = 0.001; // Tolerance for convergence

    int start = 1; // Start index of the array excluding boundary
    int end = Ny - 2; // End index of the array excluding boundary
    int vec_size = Nx - 2; // Size of the vector excluding boundary elements

    double eps_global = 0.0; // Global epsilon for convergence

    // Use OpenMP parallel for loop with reduction clause to compute global epsilon
    #pragma omp parallel for reduction(+:eps_global) schedule(static)
    for (int i = start; i <= end; i++) {
        // Compute dot product using cblas_ddot
        eps_global += cblas_ddot(vec_size, &b[i * Nx + 1], 1, &b[i * Nx + 1], 1);
    }

    MPI_Allreduce(&eps_global, &eps_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Check for convergence
    if (sqrt(eps_global) < tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << sqrt(eps_global) << endl;
        return;
    }

    ApplyOperator(x, t); // Apply the operator to x
    cblas_dcopy(n, b, 1, r, 1); // Initialize r_0 = b
    ImposeBC(r); // Impose boundary conditions on r

    cblas_daxpy(n, -1.0, t, 1, r, 1); // Update r: r = b - A*x
    Precondition(r, z); // Precondition r to get z
    cblas_dcopy(n, z, 1, p, 1); // Initialize p_0 = r_0

    int k = 0; // Iteration counter
    int k_max = 5000; // Maximum number of iterations
    do {
        k++; // Increment iteration count

        ApplyOperator(p, t); // Apply the operator to p

        // Compute alpha numerator and denominator
        double alpha_num_global = 0.0;
        double alpha_den_global = 0.0;

        #pragma omp parallel for reduction(+:alpha_num_global, alpha_den_global) schedule(static)
        for (int i = start; i <= end; i++) {
            // Use cblas_ddot to compute the dot products and add them directly to the global sums
            alpha_num_global += cblas_ddot(vec_size, p + i * Nx + 1, 1, t + i * Nx + 1, 1);
            // For alpha_den, since it's the same calculation, reuse the result from alpha_num
            alpha_den_global += cblas_ddot(vec_size, t + i * Nx + 1, 1, p + i * Nx + 1, 1);
        }

        // Sum alpha numerators and denominators across all processes
        MPI_Allreduce(MPI_IN_PLACE, &alpha_num_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &alpha_den_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha_global = alpha_num_global / alpha_den_global; // Compute global alpha

        cblas_daxpy(n,  alpha_global, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha_global, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps_global = 0.0; // Global epsilon for convergence

        // Use OpenMP parallel for loop with reduction clause to compute global epsilon
        #pragma omp parallel for reduction(+:eps_global) schedule(static)
        for (int i = start; i <= end; i++) {
            // Compute dot product using cblas_ddot and add it to eps_global
            eps_global += cblas_ddot(vec_size, &r[i * Nx + 1], 1, &r[i * Nx + 1], 1);
        }

        // Sum local epsilons across all processes using MPI reduction
        MPI_Allreduce(MPI_IN_PLACE, &eps_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Check for convergence
        if (sqrt(eps_global) < tol*tol) {
            if (rank == 0) {
                cout << "Converged in " << k << " iterations. eps = " << sqrt(eps_global) << endl;
            }
            break;
        }

        Precondition(r, z); // Precondition r to get z

        // Compute beta numerator and denominator
        double beta_num_global = 0.0;
        double beta_den_global = 0.0;

        #pragma omp parallel for reduction(+:beta_num_global, beta_den_global) schedule(static)
        for (int i = start; i <= end; i++) {
            // Use cblas_ddot to compute the dot products for beta_num and beta_den
            beta_num_global += cblas_ddot(vec_size, r + i * Nx + 1, 1, z + i * Nx + 1, 1);
            beta_den_global += cblas_ddot(vec_size, t + i * Nx + 1, 1, p + i * Nx + 1, 1);
        }

        // Sum beta numerators and denominators across all processes
        MPI_Allreduce(MPI_IN_PLACE, &beta_num_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &beta_den_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double beta_global = beta_num_global / beta_den_global; // Compute global beta

        cblas_dcopy(n, z, 1, t, 1); // Copy z to t
        cblas_daxpy(n, beta_global, p, 1, t, 1); // Update t: t = z + beta_k * p_k
        cblas_dcopy(n, t, 1, p, 1); // Update p: p = t

    } while (k < k_max); // Repeat until maximum iterations reached

    if (k == k_max) {
        if (rank == 0) {
        cout << "FAILED TO CONVERGE" << endl; // Output failure message
        }
        exit(-1); // Exit with failure status
    }
}

/**
 * @brief Applies the operator to the input array.
 * @param in The input array.
 * @param out The output array.
 */
void SolverCG::ApplyOperator(double* in, double* out) {
    // Calculate inverse of grid spacings
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;

    // Variables for neighboring indices
    int jm1 = 0, jp1 = 2;

    // Update p data among the submatrices
    ExchangeDataWithNeighbors(p);

    int i, j;
    // Apply operator in parallel
    #pragma omp parallel for default(shared) private(i,j,jm1,jp1) schedule(static)
    for (int j = 1; j < Ny - 1; ++j) {
        
        // Update neighboring indices
        jm1 = j-1;
        jp1 = j+1;

        // Apply finite difference operator
        for (int i = 1; i < Nx - 1; ++i) {
            out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i
                          + ( -     in[IDX(i, jm1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, jp1)])*dy2i;
        }
    }
}

/**
 * @brief Preconditioning function for the input array.
 * @param in The input array.
 * @param out The output array after preconditioning.
 */
void SolverCG::Precondition(double* in, double* out) {
    // Calculate inverse of grid spacings
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;

    // Compute the factor for preconditioning
    double factor = 2.0*(dx2i + dy2i);
    
    // Calculate the reciprocal of the factor for optimization
    double factor2 = 1/factor;

    // Apply preconditioning in parallel
    #pragma omp parallel for default(none) shared(in, out, factor2)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            out[IDX(i,j)] = in[IDX(i,j)]*factor2; 
        }
    }
}


