
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define MAX_VAL 127

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
 int tid = threadIdx.x + blockIdx.x * blockDim.x;

__shared__ unsigned int cache[NUM_BINS];

// Initialize 
if (blockDim.x < num_bins){ 
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    if (i < num_bins) 
      cache[i] = 0;
  }
}
else {
  if(threadIdx.x < num_bins)  
    cache[threadIdx.x] = 0;
}

// Wait for synchronization
__syncthreads();

 // Update cache
 while (tid < num_elements) {
  atomicAdd(&(cache[input[tid]]), 1);
  tid += blockDim.x * gridDim.x;
 }

// Wait for synchronization
__syncthreads();

// Add back using atomic operation
if (blockDim.x < num_bins){ 
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    if (i < num_bins) 
      atomicAdd(&(bins[i]), cache[i]);
  }
}
else {
  if(threadIdx.x < num_bins)  
    atomicAdd(&(bins[threadIdx.x]), cache[threadIdx.x]);
  }
}
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
int i = blockIdx.x * blockDim.x + threadIdx.x;    
  if (i < num_bins){
    if (bins[i] > MAX_VAL) 
        bins[i] = MAX_VAL;  
  }      
}

int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; i++){
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU
  memset(resultRef, 0, NUM_BINS * sizeof(*resultRef));

  for (int i = 0; i < inputLength; i++) {
    unsigned int num = hostInput[i];
    if (resultRef[num] < MAX_VAL) 
        resultRef[num] += 1;
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  int Dbh = 512;
  int Dgh = (inputLength + Dbh - 1) / Dbh;

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<Dgh, Dbh>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  int Dbc = 512;
  int Dgc = (NUM_BINS + Dbc - 1) / Dbc;

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<Dgc, Dbc>>>(deviceBins, NUM_BINS);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  int is_equal = 1;
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i])
        is_equal = 0;
        break;
  }       
  if(is_equal == 1)
    printf("Result is equal to reference");
  else
    printf("Result is not equal to reference");

  // Plot histogram of the code
  FILE *fp;
  
  fp = fopen("./histogram.txt","w+");
  
  if (fp == NULL) {
        printf("Error");   
        exit(1);             
  }

  for (int i = 0; i < NUM_BINS; ++i) {
        fprintf(fp, "%d\n", hostBins[i]);
  }
  fclose(fp);

  
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  
  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

