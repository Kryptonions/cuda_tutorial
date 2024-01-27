#include <iostream>
#include <cassert>
#include <cstdlib>
#include "../tools/common.cuh"

#define MASK_DIM 5
#define RADIUS MASK_DIM / 2
// 每个tile需要写的数量
#define O_TILE_WIDTH 256
// 每个tile需要读的数量，需要额外读取更多数据，所以block内的线程数多于每个block需要写的量，一部分线程只负责读数据
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_DIM - 1) 

// mask数据在常量内存里
__constant__ int M[MASK_DIM * MASK_DIM];


__global__ void tiled2DConvKernel(int *d_array, int *d_result, int width) {
    // 共享内存
    //__shared__ int Ns[BLOCK_WIDTH];  
    extern __shared__ int shmem[][];
    // 当前线程负责写的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
    int col_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    // 当前线程负责读的位置
    int row_i = row_o - RADIUS;
    int col_i = col_o - RADIUS;

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
        shmem[ty][tx] = d_array[row_i * width + col_i];
    } else {
        shmem[ty][tx] = 0;
    }
    __syncthreads();
    int output = 0;
    if ((ty < O_TILE_WIDTH) && (tx < O_TILE_WIDTH)) {
        for (int i = 0; i < MASK_DIM; i++) {
            for (int j = 0; j < MASK_DIM; j++) {
                output += M[i][j] * shmem[i + ty][j + tx];
            }
        }
    }

    if (row_o < height && col_o < width) {
        d_result[row_o * width + col_o] = output
    }
}

void verify_result(int *m, int *mask, int *result, int N) {
  int temp;

  int offset_r;
  int offset_c;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      temp = 0;

      for (int k = 0; k < MASK_DIM; k++) {
        offset_r = i - RADIUS + k;
        for (int l = 0; l < MASK_DIM; l++) {
          offset_c = j - RADIUS + l;
          if (offset_r >= 0 && offset_r < N) {
            if (offset_c >= 0 && offset_c < N) {
              temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      assert(result[i * N + j] == temp);
    }
  }
}

void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}


int main() {
    int N = 1 << 10;

    size_t bytes_n = N * N * sizeof(int);

    // 初始化
    int *matrix = new int[N * N];
    int *result = new int[N * N];
    init_matrix(matrix, N);

    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

    int *h_mask = new int[MASK_DIM * MASK_DIM];
    init_matrix(h_mask, MASK_DIM);

    // device分配内存
    int *d_matrix;
    int *d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // host2device
    cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
    // mask到常量内存
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);


    dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
    // 每个block需要写O_TILE_WIDTH * O_TILE_WIDTH个数据
    dim3 grid_dim((N + O_TILE_WIDTH - 1) / O_TILE_WIDTH, N + O_TILE_WIDTH - 1) / O_TILE_WIDTH);

    size_t SHMEM = BLOCK_WIDTH * BLOCK_WIDTH * sizeof(int);
    tiled2DConvKernel<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

    cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(matrix, h_mask, result, N);

    std::cout << "COMPLETED SUCCESSFULLY!";

    delete[] matrix;
    delete[] result;
    delete[] h_mask;

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
