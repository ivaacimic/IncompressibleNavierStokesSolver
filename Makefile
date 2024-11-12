# Compiler and flags
CXX_COMPILER = mpicxx
CXX_FLAGS = -std=c++11 -Wall -O3 
LDFLAGS = -fopenmp
LIBRARIES = -lblas -llapack -lboost_program_options -lmpi_cxx -lmpi 

# Targets
TEST_EXECUTABLE = unit_tests
OMPIMP = OMP
PROFILER_EXECUTABLE = profiler_solver
PARAMETERS = --Lx 1 --Ly 1 --Nx 201 --Ny 201 --Re 1000 --dt 0.005 --T 1

# Phony targets
.PHONY: clean run $(OMPIMP) profiler doc

# Compilation rules
# Object files
LidDrivenCavitySolver.o: LidDrivenCavitySolver.cpp LidDrivenCavity.h SolverCG.h
	$(CXX_COMPILER) -c LidDrivenCavitySolver.cpp -o LidDrivenCavitySolver.o $(LDFLAGS)

LidDrivenCavity.o: LidDrivenCavity.cpp LidDrivenCavity.h
	$(CXX_COMPILER) -c LidDrivenCavity.cpp -o LidDrivenCavity.o $(LDFLAGS)

SolverCG.o: SolverCG.cpp SolverCG.h
	$(CXX_COMPILER) -c SolverCG.cpp -o SolverCG.o $(LDFLAGS)

NewGrid.o: NewGrid.cpp NewGrid.h
	$(CXX_COMPILER) -c NewGrid.cpp -o NewGrid.o $(LDFLAGS)

# Linking the solver executable
solver: LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o
	$(CXX_COMPILER) $(CXX_FLAGS) -o solver LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o $(LIBRARIES) $(LDFLAGS)

# OpenMP executable
$(OMPIMP): LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o
	$(CXX_COMPILER) $(CXX_FLAGS) $(LDFLAGS) -o $(OMPIMP) LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o $(LIBRARIES)

# Unit test object files
UnitTest_LidDrivenCavity.o: LidDrivenCavityTest.cpp LidDrivenCavity.h SolverCG.h 
	$(CXX_COMPILER) $(CXX_FLAGS) -c LidDrivenCavityTest.cpp -o LidDrivenCavityTest.o $(LDFLAGS)

UnitTest_SolverCG.o: SolverCGTest.cpp SolverCG.h
	$(CXX_COMPILER) $(CXX_FLAGS) -c SolverCGTest.cpp -o SolverCGTest.o $(LDFLAGS)

# Unit test executables
UnitTest_LidDrivenCavity: UnitTest_LidDrivenCavity.o LidDrivenCavity.o SolverCG.o NewGrid.o
	$(CXX_COMPILER) $(CXX_FLAGS) -o UnitTest_LidDrivenCavity LidDrivenCavityTest.o LidDrivenCavity.o SolverCG.o NewGrid.o $(LIBRARIES)

UnitTest_SolverCG: UnitTest_SolverCG.o SolverCG.o NewGrid.o
	$(CXX_COMPILER) $(CXX_FLAGS) -o UnitTest_SolverCG SolverCGTest.o SolverCG.o NewGrid.o $(LIBRARIES)

# Unit tests executable
unit_tests: tests.cpp LidDrivenCavity.o SolverCG.o NewGrid.o
	$(CXX_COMPILER) $(CXX_FLAGS) -o unit_tests tests.cpp LidDrivenCavity.o SolverCG.o NewGrid.o $(LIBRARIES)

# Profiler executable
$(PROFILER_EXECUTABLE): LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o
	export OMPI_CXX=g++-10
	$(CXX_COMPILER) $(LDFLAGS) -g -o $(PROFILER_EXECUTABLE) LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o $(LIBRARIES)

# Running the solver
run: solver
	mpiexec -np 9 ./solver $(PARAMETERS)

# Running the OpenMP version
run_OMP: $(OMPIMP)
	export OMP_NUM_THREADS=9 && mpiexec --bind-to none -np 1 ./$(OMPIMP) $(PARAMETERS)

# Running unit tests
run_UnitTest_LidDrivenCavity: UnitTest_LidDrivenCavity
	./UnitTest_LidDrivenCavity

run_UnitTest_SolverCG: UnitTest_SolverCG
	./UnitTest_SolverCG

# Profiling the solver
profiler: $(PROFILER_EXECUTABLE)
	module load dev-studio
	export OMP_NUM_THREADS=1
	collect -o profiler_test.er ./$(PROFILER_EXECUTABLE) $(PARAMETERS)
	analyzer profiler_test.er

# Target to generate Doxygen configuration file if it doesn't exist
Doxyfile:
	doxygen -g

# Target to generate documentation using Doxygen
doc: Doxyfile
	doxygen $(Doxyfile)

# Clean all generated files
clean:
	rm -f LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o NewGrid.o solver $(OMPIMP) unit_tests $(PROFILER_EXECUTABLE) profiler_test.er
