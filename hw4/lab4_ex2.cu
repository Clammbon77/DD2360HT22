
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < len)
    out[id] = in1[id] + in2[id];
}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
  
  int inputLength;
  int S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength and segement size from args
  inputLength = atoi(argv[1]);
  S_seg = atoi(argv[2]);
  const int nStreams = inputLength / S_seg;
  printf("The input length is %d\n", inputLength);
  printf("The stream size is %d\n", S_seg);
  printf("The number of streams is %d\n", nStreams);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength*sizeof(DataType)); 
  hostInput2 = (DataType*)malloc(inputLength*sizeof(DataType)); 
  hostOutput = (DataType*)malloc(inputLength*sizeof(DataType));
  resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
   
  // @@ Insert code below to create streams
  cudaStream_t streams[nStreams]; 
  for(int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
      
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i< inputLength; ++i) {
        DataType rand1 = rand() / (DataType) (RAND_MAX + 1.0); // Random number in interval [0, 1.0)
        DataType rand2 = rand() / (DataType) (RAND_MAX + 1.0); // Random number in interval [0, 1.0)
        hostInput1[i] = rand1;
        hostInput2[i] = rand2;
        resultRef[i] = rand1 + rand2;
    }
  
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));
  
  //@@ Initialize the 1D grid and block dimensions here
  int Db = 256;
  int Dg = S_seg / Db; 

  //@@ Insert code to below to Copy mem to the GPU here
  for(int i = 0; i < nStreams; i++) {
        int offset = i * S_seg;
        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[i]); 
        vecAdd<<<Dg, Db, 0, streams[i]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, S_seg);
        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, S_seg * sizeof(DataType), 
                        cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamSynchronize(streams[i]);
  }
  
  for(int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  cudaDeviceSynchronize();
  
  //@@ Insert code below to compare the output with the reference 
  int is_equal = 1;
  for(int i = 0; i < inputLength; i++){
    if (hostOutput[i] - resultRef[i]> 1e-10)
      is_equal = 0;
    break;           
  }
  if(is_equal == 1)
    printf("Result is equal to reference");
  else
    printf("Result is not equal to reference");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
