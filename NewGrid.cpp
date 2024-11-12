#include "NewGrid.h"

/**
 * @brief Calculates the local size of the grid based on the MPI rank coordinates.
 * 
 * This function calculates the local size of the grid in both the x-direction and y-direction
 * based on the MPI rank coordinates, ensuring that each process handles a portion of the grid
 * that excludes the boundary points.
 * 
 * @param Nx The total number of grid points in the x-direction.
 * @param Ny The total number of grid points in the y-direction.
 * @param p_root Square root of the total number of processes.
 * @param xcoord The x-coordinate of the MPI rank.
 * @param ycoord The y-coordinate of the MPI rank.
 * @param Nx_loc Reference to the variable to store the local size in the x-direction.
 * @param Ny_loc Reference to the variable to store the local size in the y-direction.
 */
void NewGrid(int Nx, int Ny, int p_root, int xcoord, int ycoord, int& Nx_loc, int& Ny_loc) {
    // Helper function to calculate local size based on rank
    auto calculateLocalSize = [](int size, int root, int coord, int& local_size) {
        int new_size = size - 2; // Grid size without the boundaries
        int r = new_size % root; // Remainder
        int k = (new_size - r) / root; // Minimum size of chunk for the inner grid
        if (coord < r) {
            local_size = k + 3; // For ranks < r, chunk size is k + 1
        } else {
            local_size = k + 2; // For ranks >= r, chunk size is k
        }
    };

    // In x-direction
    calculateLocalSize(Nx, p_root, xcoord, Nx_loc);

    // In y-direction
    calculateLocalSize(Ny, p_root, ycoord, Ny_loc);
}

