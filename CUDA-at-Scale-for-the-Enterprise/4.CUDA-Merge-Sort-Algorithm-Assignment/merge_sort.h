#include <iostream>
#include <string> // stoi()
#include <tuple>
//#include <helper_cuda.h>
#include <sys/time.h>

using namespace std;

__host__ long *mergesort(long *data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid);
__host__ long *generateRandomLongArray(int numElements);
__host__ std::tuple<long *, long *> allocateMemory(int numElements);
__host__ void printHostMemory(long *host_mem, int num_elments);
__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char **argv);

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices) ;
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end);
