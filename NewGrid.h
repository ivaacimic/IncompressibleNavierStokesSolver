#pragma once

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
 * @param nx Reference to the variable to store the local size in the x-direction.
 * @param Ny_loc Reference to the variable to store the local size in the y-direction.
 */
void NewGrid(int Nx, int Ny, int p_root, int xcoord, int ycoord, int& nx, int& Ny_loc);
