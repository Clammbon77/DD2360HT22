
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){

  //@@ Insert code to implement matrix multiplication here
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    DataType sum = 0.0;

      if( col < numBColumns && row < numARows )
      {
        for(int m = 0; m < numAColumns; m++) 
        {
            sum += A[row * numAColumns + m] * B[m * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
      }

}

int main(int argc, char **argv) {
  
  DataType *uniA; // The A matrix
  DataType *uniB; // The B matrix
  DataType *uniC; // The output C matrix
  DataType *resultRef; // The reference result

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocManaged((void**)&uniA, sizeof(DataType) * numARows * numAColumns);
  cudaMallocManaged((void**)&uniB, sizeof(DataType) * numBRows * numBColumns);
  cudaMallocManaged((void**)&uniC, sizeof(DataType) * numCRows * numCColumns);
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
           DataType randn1 = rand() / (DataType) RAND_MAX;
           uniA[i*numAColumns + j] = randn1;
        }
  }
  for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
           DataType randn2 = rand() / (DataType) RAND_MAX;
           uniB[i*numBColumns + j] = randn2;
        }
  }
  
  for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
          resultRef[i*numBColumns + j] = 0.0;
          for (int k = 0; k < numAColumns; k++) {
            resultRef[i*numBColumns + j] +=  uniA[i*numAColumns + k] * uniB[k*numBColumns + j];
          }        
        }
  }
   
  //@@ Initialize the grid and block dimensions here
  int Dbx = 32;
  int Dby = 32;
  int Dgx = (numCColumns + Dbx - 1) / Dbx;
  int Dgy = (numCRows + Dby - 1) / Dby;

  //@@ Launch the GPU Kernel here
  gemm<<<dim3(Dgx, Dgy, 1), dim3(Dbx, Dby, 1)>>>(uniA, uniB, uniC, numARows, numAColumns, numBRows, numBColumns);

  cudaDeviceSynchronize();
  
  //@@ Insert code below to compare the output with the reference
  int is_equal = 1;
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      if (fabs(uniC[i*numCColumns + j] - resultRef[i*numCColumns + j]) > 1e-4)
        is_equal = 0;
        break;
    }       
  }
  if(is_equal == 0)
    printf("Result is not equal to reference");
  else
    printf("Result is equal to reference");

  cudaFree(uniA);
  cudaFree(uniB);
  cudaFree(uniC);

  
  free(resultRef);

  return 0;
}
