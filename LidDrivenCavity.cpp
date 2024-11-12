// Standard library includes
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>

// Namespace used
using namespace std;

// External library includes for linear algebra operations, MPI, and OpenMP
#include <cblas.h>
#include <mpi.h>
#include <omp.h>

// Application-specific includes the following .h files
#include "LidDrivenCavity.h"
#include "SolverCG.h"
#include "NewGrid.h"

/**
 * @def IDX(I, J)
 * Macro to calculate the linear index for a 2D array stored in row-major order.
 * @param I The row index.
 * @param J The column index.
 * @return The linear index corresponding to the (I, J) position in the array.
 */
#define IDX(I,J) ((J)*Nx + (I))

/**
 * @def idx(I, J)
 * Macro to calculate the linear index for a local 2D array (in a distributed system) stored in row-major order.
 * @param I The local row index.
 * @param J The local column index.
 * @return The linear index corresponding to the local (I, J) position in the array.
 */
#define idx(I,J) ((J)*Nx_loc + (I))

/**
 * @brief Constructor for the LidDrivenCavity class.
 *
 * This constructor initializes a LidDrivenCavity object. It's the default constructor,
 * so it doesn't initialize member variables with specific values. The initialization
 * of the simulation parameters should be done using separate setter methods after
 * object creation.
 */
LidDrivenCavity::LidDrivenCavity(){}

/**
 * @brief Destructor for the LidDrivenCavity class.
 *
 * This destructor is called when a LidDrivenCavity object is destroyed. It ensures
 * that resources are properly released by calling the CleanUp method, which deletes
 * dynamically allocated memory to prevent memory leaks.
 */
LidDrivenCavity::~LidDrivenCavity(){
    // Call CleanUp to free dynamically allocated resources.
    CleanUp();
    }

/**
 * @brief Checks the validity of the MPI environment for the Lid Driven Cavity simulation.
 *
 * This function ensures that the number of MPI processes (size) is a perfect square,
 * which is a requirement for the 2D Cartesian grid decomposition used in the simulation.
 * If the number of processes is not a perfect square, the function will terminate the 
 * program with an error message.
 *
 * @note This function should be called after initializing MPI and assumes that 
 * MPI_Comm_rank and MPI_Comm_size are available.
 */
void LidDrivenCavity::CheckMPI()
{
    // Get the rank of the current MPI process and the total number of MPI processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate the square root of the total number of processes
    p_root = sqrt(size);

    // Check if the total number or processes is not a perfect square
    if (size != p_root*p_root)
    {   
        // If rank 0, print error message
        if (rank == 0) {
            cout << "The number of processes must be a perfect square" << endl;
        }
        // Finalize MPI and exit with error code
        MPI_Finalize();
        exit(-1);
    }
}

/**
 * @brief Sets the physical dimensions of the domain.
 *
 * This method sets the length (Lx) and width (Ly) of the rectangular domain used in the simulation.
 * After setting these values, it calls UpdateDxDy() to update the grid spacing based on the new domain
 * size and the number of grid points.
 *
 * @param xlen The length of the domain in the x-direction.
 * @param ylen The length of the domain in the y-direction.
 */
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    // Set the length of the domain in the x and y directions
    this->Lx = xlen;
    this->Ly = ylen;

    // Update dx and dy values based on the new domain size
    UpdateDxDy();
}

/**
 * @brief Sets the grid size and configures the 2D Cartesian topology for MPI processes.
 *
 * This method sets the number of grid points (Nx, Ny) in the x and y directions, respectively,
 * and updates the grid spacing. It also configures a 2D Cartesian topology for the MPI processes
 * to facilitate communication in a structured grid.
 *
 * @param nx The number of grid points in the x-direction.
 * @param ny The number of grid points in the y-direction.
 */
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    // Set the number of grid points in the x and y directions
    this->Nx = nx;
    this->Ny = ny;

    // Update dx and dy values based on the new grid size
    UpdateDxDy();

    // Define 2D Cartesian topology for the MPI processes
    int dims[2] = {p_root, p_root};
    int periods[2] = {0, 0};
    int reorder = 1;

    // Create Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &mygrid);
    
    // Get the rank of the current process in the Cartesian communicator
    MPI_Comm_rank(mygrid, &mygrid_rank);
    
    // Get the Cartesian coordinates of the current process
    MPI_Cart_coords(mygrid, mygrid_rank, 2, coords);
    
    // Create a new grid based on the specified grid size and MPI topology
    NewGrid(Nx, Ny, p_root, coords[0], coords[1], Nx_loc, Ny_loc);

}

/**
 * @brief Sets the time step for the simulation.
 *
 * This method assigns the value of the time step (dt) used in the numerical integration 
 * of the lid-driven cavity flow equations. The time step is crucial for the accuracy and 
 * stability of the simulation.
 *
 * @param deltat The time step to be used in the simulation.
 */
void LidDrivenCavity::SetTimeStep(double deltat)
{
    // Set the time step value
    this->dt = deltat;
}

/**
 * @brief Sets the final time for the simulation.
 *
 * This method establishes the final time (T) for the lid-driven cavity simulation, 
 * determining how long the simulation will run. The final time is essential to ensure 
 * that the simulation progresses to the desired point in time, providing meaningful 
 * and accurate results.
 *
 * @param finalt The final time up to which the simulation should run.
 */
void LidDrivenCavity::SetFinalTime(double finalt)
{
    // Set the final time value
    this->T = finalt;
}

/**
 * @brief Sets the Reynolds number for the simulation.
 *
 * This method sets the Reynolds number (Re) for the lid-driven cavity flow simulation. 
 * It also calculates and sets the kinematic viscosity (nu) based on the Reynolds number,
 * which is crucial for the simulation's fluid dynamics calculations.
 *
 * @param re The Reynolds number to be used in the simulation.
 */
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    // Set the Reynolds number value
    this->Re = re;

    // Calculate and set the kinematic viscosity
    this->nu = 1.0/re;
}

/**
 * @brief Initializes the simulation environment.
 *
 * This method prepares the simulation for execution by cleaning up any previous data,
 * allocating memory for the simulation variables, and setting up the conjugate gradient
 * solver. It is a crucial step to ensure that the simulation starts with the correct
 * setup and allocated resources.
 */
void LidDrivenCavity::Initialise()
{
    // Clean up any existing data
    CleanUp();

    // Calculate the total number of grid points on the local domain
    Npts_local = Nx_loc * Ny_loc;

    // Allocate memory for temporary arrays
    v   = new double[Npts_local]();
    vnew= new double[Npts_local]();
    s   = new double[Npts_local]();
    tmp = new double[Npts_local]();

    // Create a Conjugate Gradient solver object for solving pressure Poisson equation
    cg  = new SolverCG(Nx_loc, Ny_loc, dx, dy, coords, p_root, rank, mygrid);
}

/**
 * @brief Advances the simulation through time steps.
 *
 * This method iteratively advances the state of the simulation over a series of time steps
 * until the final simulation time is reached. At each step, it updates the simulation state
 * by calling the Advance method and logs the current step and simulation time.
 */
void LidDrivenCavity::Integrate()
{
    // Calculate the total number of time steps
    int NSteps = ceil(T/dt);

    // Perform time integration loop
    for (int t = 0; t < NSteps; ++t)
    {   
        // Print progress information for each time step from process 0
        if (rank == 0) {
            std::cout << "Step: " << setw(8) << t
                    << "  Time: " << setw(8) << t*dt
                    << std::endl;
        }
        // Advance the solution to the next time step
        Advance(t);
    }
}

/**
 * @brief Writes the simulation results to a file in parallel.
 *
 * This method calculates the starting indices and sizes for each process to write its portion of the
 * data to the output file. It uses MPI I/O functions to write data in parallel, ensuring that each process
 * writes its data segment to the correct location in the file.
 *
 * @param file The name of the file to which the data will be written.
 */
void LidDrivenCavity::WriteSolution(std::string file)
{
    // Check if the submatrix has neighbouring submatrices
    bool hasNeighbor[4] = {coords[0] != 0, coords[0] != p_root -1, coords[1] != 0, coords[1] != p_root-1};
    // Allocate memory for velocity components
    double* u0 = new double[Npts_local]();
    double* u1 = new double[Npts_local]();
    // Calculate the new grid size without boundaries
    int Nx_new = Nx - 2;
    int Ny_new = Ny - 2;

    // Define variables for the starting indices
    int start_index_x, start_index_y;

    // Calculate the remainder when dividing Nx_new by p_root
    int reminder_x = Nx_new % p_root;
    // Determine the starting index in the x-direction
    if (coords[0] < reminder_x) {
        start_index_x = (Nx_new / p_root) * coords[0] + coords[0];
    } else {
        start_index_x = (Nx_new / p_root) * coords[0] + reminder_x;
    }

    // Calculate the remainder when dividing Ny_new by p_root
    int reminder_y = Ny_new % p_root;
    // Determine the starting index in the y-direction
    if (coords[1] < reminder_y) {
        start_index_y = (Ny_new / p_root) * coords[1] + coords[1];
    } else {
        start_index_y = (Ny_new / p_root) * coords[1] + reminder_y;
    }
    
    // Calculate the new size of the local grid in the x-direction based on boundary conditions
    // If there are no neighbors in the left or right direction, decrease the grid size by 1, otherwise by 2
    Nx_new = (!hasNeighbor[0] || !hasNeighbor[1]) ? Nx_loc - 1 : Nx_loc - 2;
    // Calculate the new size of the local grid in the y-direction based on boundary conditions
    // If there are no neighbors in the bottom or top direction, decrease the grid size by 1, otherwise by 2
    Ny_new = (!hasNeighbor[2] || coords[1] == !hasNeighbor[3]) ? Ny_loc - 1 : Ny_loc - 2;
    // Determine the offset in the x-direction based on boundary conditions
    // If there are no neighbors in the left direction, set the offset to 0, otherwise to 1
    int offset_x = (!hasNeighbor[0]) ? 0 : 1;
    // Determine the offset in the y-direction based on boundary conditions
    // If there are no neighbors in the bottom direction, set the offset to 0, otherwise to 1
    int offset_y = (!hasNeighbor[2]) ? 0 : 1;

    // Compute the velocity components u0 and u1
    for (int i = offset_x; i < Nx_loc - 1 + offset_x; ++i) {
        for (int j = offset_y; j < Ny_loc - 1 + offset_y; ++j) {
            u0[idx(i, j)] = (s[idx(i, j + 1)] - s[idx(i, j)]) / dy;
            u1[idx(i, j)] = (s[idx(i, j)] - s[idx(i + 1, j)]) / dx;
        }
    }

    // Apply boundary conditions for the top edge
    if (!hasNeighbor[3]) {
        for (int i = 0; i < Nx_loc; ++i) {
            u0[idx(i, Ny_loc - 1)] = U;
        }
    }

    // Display a message indicating the beginning of file writing
    if (rank==0) {
        std::cout << "Writing the file " << file << std::endl;
    }

    // stringstream to store data
    stringstream ss;

    // Loop over the local grid in the x-direction
    for (int i = 0; i < Nx_new; ++i) {
        // Loop over the local grid in the y-direction
        for (int j = 0; j < Ny_new; ++j) {
            // Compute the index in the global array
            int idx = Nx_loc * (j + offset_y) + i + offset_x;

            // Compute the global coordinates of the current point
            int wherex = start_index_x + i;
            int wherey = start_index_y + j;

            // Append the data to the stringstream
            ss << wherex * dx << " "    // x-coordinate
            << wherey * dy << " "    // y-coordinate
            << v[idx] << " "         // velocity component
            << s[idx] << " "         // some value
            << u0[idx] << " "        // another value
            << u1[idx] << "\n";      // yet another value
        }
    }

    // Gather and write the data using MPI

    // Convert the stringstream to a string to gather the data
    string gathered_data = ss.str();

    // Array to hold sizes of gathered data from all processes
    int all_data_sizes[p_root * p_root];

    // Calculate the size of the gathered data
    int data_size = gathered_data.size();

    // Gather the size of data from each process
    MPI_Allgather(&data_size, 1, MPI_INT, all_data_sizes, 1, MPI_INT, MPI_COMM_WORLD);

    // Offset for writing data in the file
    MPI_Offset write_offset = 0;

    // Calculate the offset for this process based on gathered data sizes
    for (int i = 0; i < rank; ++i) {
        write_offset += all_data_sizes[i];
    }

    // Declare MPI file handle 'fh' and set access mode to create a new file or open an existing one for reading and writing
    MPI_File fh;
    int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;

    // Open the file for reading and writing
    MPI_File_open(MPI_COMM_WORLD, file.c_str(), access_mode, MPI_INFO_NULL, &fh);

    int mpi_result;

    // Check if file opening was successful
    if ((mpi_result = MPI_File_open(MPI_COMM_WORLD, file.c_str(), access_mode, MPI_INFO_NULL, &fh)) != MPI_SUCCESS) {
        // Display error message and abort if opening failed
        cerr << "[MPI process " << rank << "] Failure in opening the file. MPI Error code: " << mpi_result << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Write the gathered_data to the file
    MPI_File_write_at_all(fh, write_offset, gathered_data.c_str(), size, MPI_CHAR, MPI_STATUS_IGNORE);
    if ((mpi_result = MPI_File_write_at_all(fh, write_offset, gathered_data.c_str(), size, MPI_CHAR, MPI_STATUS_IGNORE)) != MPI_SUCCESS) {
        // Display error message and abort if writing failed
        cerr << "[MPI process " << rank << "] Failure in writing to the file. MPI Error code: " << mpi_result << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Display a message indicating the completion of file writing
    if (rank==0) {
        std::cout << "Finished writing the file " << file << std::endl;
    }

    // Clean up the dynamically allocated memory
    delete[] u0;
    delete[] u1;
}

/**
 * @brief Prints the simulation configuration.
 *
 * This method outputs the simulation's grid configuration, time stepping details, and physical parameters.
 * It is designed to be called only by the process with rank 0 to avoid cluttering the output with redundant
 * information from other processes. It also checks for the time-step restriction based on the viscosity
 * and grid spacing to ensure numerical stability.
 */
void LidDrivenCavity::PrintConfiguration()
{   
    if (rank == 0) {        
        // Print grid size, spacing, domain length, grid points, time step, Reynolds number
        cout << "Grid size: " << Nx << " x " << Ny << endl;
        cout << "Spacing:   " << dx << " x " << dy << endl;
        cout << "Length:    " << Lx << " x " << Ly << endl;
        cout << "Grid pts:  " << Npts << endl;
        cout << "Timestep:  " << dt << endl;
        cout << "Steps:     " << ceil(T/dt) << endl;
        cout << "Reynolds number: " << Re << endl;
        cout << "Linear solver: preconditioned conjugate gradient" << endl;
        cout << endl;

        // Check time-step restriction
        if (nu * dt / dx / dy > 0.25) {
            cout << "ERROR: Time-step restriction not satisfied!" << endl;
            cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
            exit(-1);
        }
    }
}

/**
 * @brief Cleans up dynamically allocated memory.
 *
 * This method is responsible for freeing the dynamically allocated memory used by the simulation.
 * It is typically called during the destruction of a LidDrivenCavity object or before re-initializing
 * the simulation to prevent memory leaks.
 */
void LidDrivenCavity::CleanUp()
{   
    // Check if resources are allocated before cleaning up
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete cg;
    }
}

/**
 * @brief Updates the grid spacing and total number of grid points.
 *
 * This method calculates the grid spacings (dx and dy) based on the domain size (Lx and Ly) and
 * the number of grid points (Nx and Ny). It also updates the total number of grid points (Npts) 
 * and the number of local grid points (Npts_local) for use in the simulation.
 */
void LidDrivenCavity::UpdateDxDy()
{
    // Calculate grid spacing in x and y directions
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);

    // Update total number of grid points and local number of grid points
    Npts = Nx * Ny;
    Npts_local = Nx_loc * Ny_loc;
}

/**
 * @brief Updates the boundary data with parallel processes.
 * 
 * This method is responsible for updating the boundary data of the current process's grid
 * by communicating with adjacent processes in the 2D Cartesian grid. It ensures the continuity
 * of data across process boundaries which is crucial for the accuracy of the simulation.
 * 
 * @param data Pointer to the data array that will be updated with boundary values from neighboring processes.
 */
void LidDrivenCavity::ExchangeDataWithNeighbors(double* data) {

    // Variables to store neighboring ranks
    int lrank;
    int rrank;
    int trank;
    int brank;

    // Arrays to store boundary data sent and received
    double *left_boundary_sent = new double[Ny_loc];
    double *left_boundary_received = new double[Ny_loc];
    double *right_boundary_sent = new double[Ny_loc];
    double *right_boundary_received = new double[Ny_loc];
    double *top_boundary_sent = new double[Nx_loc];
    double *top_boundary_received = new double[Nx_loc];
    double *bottom_boundary_sent = new double[Nx_loc];
    double *bottom_boundary_received = new double[Nx_loc];

    // Array to store MPI requests
    MPI_Request send_req[4]; 

    // Counter to keep track of the number of requests
    int counter = 0; 

    // Determine the neighboring ranks using MPI_Cart_shift
    MPI_Cart_shift(mygrid, 0, 1, &lrank, &rrank); // Shift along the x-axis
    MPI_Cart_shift(mygrid, 1, 1, &trank, &brank); // Shift along the y-axis

    // Determine if the process has neighboring processes in each direction
    bool hasNeighbor[4] = {coords[0] != 0, coords[0] != p_root -1, coords[1] != 0, coords[1] != p_root-1};

    // Send to the left, receive from the right
    if (hasNeighbor[0]) {
        
        // Sending left boundary to the neighboring process on the left
        #pragma omp parallel for
        for (int i = 0; i < Ny_loc; ++i) {
            left_boundary_sent[i] = data[idx(1, i)];
        }
        MPI_Isend(left_boundary_sent, Ny_loc, MPI_DOUBLE, lrank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(left_boundary_received, Ny_loc, MPI_DOUBLE, lrank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the left boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Ny_loc; ++i) {
            data[idx(0, i)] = left_boundary_received[i];
        }
    }

    // Send to the right, receive from the left
    if (hasNeighbor[1]) {
        
        // Sending right boundary to the neighboring process on the right
        #pragma omp parallel for
        for (int i = 0; i < Ny_loc; ++i) {
            right_boundary_sent[i] = data[idx(Nx_loc-2, i)];
        }
        MPI_Isend(right_boundary_sent, Ny_loc, MPI_DOUBLE, rrank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(right_boundary_received, Ny_loc, MPI_DOUBLE, rrank, 0, mygrid, MPI_STATUS_IGNORE);  
        
        // Update the right boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Ny_loc; ++i) {
            data[idx(Nx_loc-1, i)] = right_boundary_received[i];
        }      
    }

    // Send to the top, receive from the bottom
    if (hasNeighbor[2]) {

        // Sending top boundary to the neighboring process on the top
        #pragma omp parallel for
        for (int i = 0; i < Nx_loc; ++i) {
            top_boundary_sent[i] = data[idx(i, 1)];
        }
        MPI_Isend(top_boundary_sent, Nx_loc, MPI_DOUBLE, trank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(top_boundary_received, Nx_loc, MPI_DOUBLE, trank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the top boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Nx_loc; ++i) {
            data[idx(i, 0)] = top_boundary_received[i];
        }
    }

    // Send to the bottom, receive from the top
    if (hasNeighbor[3]) {

        // Sending bottom boundary to the neighboring process on the bottom
        #pragma omp parallel for
        for (int i = 0; i < Nx_loc; ++i) {
            bottom_boundary_sent[i] = data[idx(i, Ny_loc-2)];
        }
        MPI_Isend(bottom_boundary_sent, Nx_loc, MPI_DOUBLE, brank, 0, mygrid, &send_req[counter++]);
        MPI_Recv(bottom_boundary_received, Nx_loc, MPI_DOUBLE, brank, 0, mygrid, MPI_STATUS_IGNORE);
        
        // Update the bottom boundary of the local domain with received data
        #pragma omp parallel for
        for (int i = 0; i < Nx_loc; ++i) {
            data[idx(i, Ny_loc - 1)] = bottom_boundary_received[i];
        }
    }

    // Clean up dynamically allocated memory
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
 * @brief Advances the simulation by one time step.
 * 
 * This method advances the simulation by updating the vorticity and stream function
 * values across the grid for one time step. It involves computing the interior vorticity,
 * applying boundary conditions, updating the vorticity with time, and solving the Poisson
 * equation to obtain the stream function values.
 * 
 * @param idxT The current time step index.
 */
void LidDrivenCavity::Advance(int idxT)
{
    // Define reciprocal values of grid spacings
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    int i, j;

    // Array indicating if the process has neighbors on its four sides
    bool hasNeighbor[4] = {coords[0] != 0, coords[0] != p_root -1, coords[1] != 0, coords[1] != p_root-1};
    
    // Compute vorticity boundary conditions
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Set vorticity boundary conditions for processes without neighbors on their left side
            if (!hasNeighbor[0]) {
                #pragma omp task
                for (j = 1; j < Ny_loc-1; ++j) {
                    v[idx(0,j)]    = 2.0 * dx2i * (s[idx(0,j)]    - s[idx(1,j)]);
                }
            }

            // Set vorticity boundary conditions for processes without neighbors on their right side
            if (!hasNeighbor[1]) {
                #pragma omp task
                for (j = 1; j < Ny_loc-1; ++j) {
                    // right
                    v[idx(Nx_loc-1,j)] = 2.0 * dx2i * (s[idx(Nx_loc-1,j)] - s[idx(Nx_loc-2,j)]);
                }
            }

            // Set vorticity boundary conditions for processes without neighbors on their top side
            if (!hasNeighbor[2]) {
                #pragma omp task
                for (i = 1; i < Nx_loc-1; ++i) {
                    // top
                    v[idx(i,0)]    = 2.0 * dy2i * (s[idx(i,0)]    - s[idx(i,1)]);
                }
            }
            
            // Set vorticity boundary conditions for processes without neighbors on their bottom side
            if (!hasNeighbor[3]) {
                #pragma omp task
                for (i = 1; i < Nx_loc-1; ++i) {
                    // bottom
                    v[idx(i,Ny_loc-1)] = 2.0 * dy2i * (s[idx(i,Ny_loc-1)] - s[idx(i,Ny_loc-2)])
                                - 2.0 * dyi*U;
                }
            }
        }
        #pragma omp taskwait // Wait for all tasks to complete
    }

    // Update vorticity data among the submatrices
    ExchangeDataWithNeighbors(v);

    // Compute interior vorticity
    #pragma omp parallel for collapse(2) default(shared) private(i,j) schedule (static)
    for (i = 1; i < Nx_loc - 1; ++i) {
        for (j = 1; j < Ny_loc - 1; ++j) {
            v[idx(i,j)] = dx2i*(
                    2.0 * s[idx(i,j)] - s[idx(i+1,j)] - s[idx(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[idx(i,j)] - s[idx(i,j+1)] - s[idx(i,j-1)]);
        }
    }

    // Update vorticity data among the submatrices
    ExchangeDataWithNeighbors(v);

    // Time advance vorticity
    #pragma omp parallel for collapse(2) default(shared) private(i,j) schedule (static)
    for (i = 1; i < Nx_loc - 1; ++i) {
        for (j = 1; j < Ny_loc - 1; ++j) {
            vnew[idx(i,j)] = v[idx(i,j)] + dt*(
                ( (s[idx(i+1,j)] - s[idx(i-1,j)]) * 0.5 * dxi
                 *(v[idx(i,j+1)] - v[idx(i,j-1)]) * 0.5 * dyi)
              - ( (s[idx(i,j+1)] - s[idx(i,j-1)]) * 0.5 * dyi
                 *(v[idx(i+1,j)] - v[idx(i-1,j)]) * 0.5 * dxi)
              + nu * (v[idx(i+1,j)] - 2.0 * v[idx(i,j)] + v[idx(i-1,j)])*dx2i
              + nu * (v[idx(i,j+1)] - 2.0 * v[idx(i,j)] + v[idx(i,j-1)])*dy2i);
        }
    }

    // Solve Poisson problem
    cg->Solve(vnew, s);

    // Update stream function data among the submatrices
    ExchangeDataWithNeighbors(s);
}
