# Multi-Threading
This tutorial teaches basics of OpenMP parallelization in CE 4SP4 using a simple example.


### Logging in to the server
Use``ssh <username>@srv-cad.ece.mcmaster.ca`` to login to the server.
* `username` and password is provided to you.


### Cloning and Buidling the repository
* Use `git clone https://github.com/cheshmi/vectorization.git` to clone the repository in your scratch directory

### Running the code
* Use `sbatch build_run.sh` to run the code on a compute node of the teach cluster.
* The script runs both examples and generates performance benchmarks.

# Multi-Threading Examples

### 1. Vector Multiplication (vec_mul)

Basic OpenMP parallelization example demonstrating:
- Base unthreaded implementation
- OpenMP parallelized version with and without unrolling
- Performance benchmarking

**File:** `src/vec_mul.cpp`

### 2. Pi Computation (pi)
Demonstrates the use of OpenMP to compute the value of Pi using numerical integration.
- Base unthreaded implementation
- OpenMP parallelized version without #pragma parallel for

**File:** `src/pi.cpp`

TODO: 
* add a new implementation of pi computation using #pragma parallel for and compare the performance with the existing implementation.
* add a benchmarking script to compare the performance of the two implementations of pi computation.





