#include <iostream>
#include "../tools/common.cuh"

using namespace std;

const int TILE_WIDTH = 16; // 必须加const
//extern __shared__ float sharedA[][];
//extern __shared__ float sharedB[][];

struct Matrix
{
    int width;
    int height;
    float *elements;
};

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void tiledMatMulKernel(Matrix *A, Matrix *B, Matrix *C)
{   
    // block内共享内存，矩阵维度必须是常量，否则编译报错
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];  
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
	  float Cvalue = 0.0;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
	int row = ty + by * blockDim.y;  //线程计算的元素所在的行，(row, col)表示计算元素的矩阵索引
	int col = tx + bx * blockDim.x;  //线程计算的元素所在的列，(row, col)表示计算元素的矩阵索引
    int m = A->height;                //A的行数
    int n = A->width;                 //A的列数
    int k = B->width;                 //B的列数
    for (int i = 0; i < (n - 1) / TILE_WIDTH + 1; i++) {
        //保证A的元素valid
        if(row < m && i * TILE_WIDTH + tx < n) {
            sharedA[ty][tx] = getElement(A, row, i * TILE_WIDTH + tx);
        } else {
            sharedA[ty][tx] = 0.0;
        }
        //保证B的元素valid
        if (col < k && i * TILE_WIDTH + ty < n) {
            sharedB[ty][tx] = getElement(B, i * TILE_WIDTH + ty, col);
        } else {
            sharedA[ty][tx] = 0.0;
        } 
        // 等待block内所有线程读取数据到共享内存
        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; j++) {
            Cvalue += sharedA[ty][j] * sharedB[j][tx];
        }
        // 等待block内所有线程计算得到
        __syncthreads();
    }
	  if (row < m && col < k) {
        setElement(C, row, col, Cvalue);
    }
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int width = 1 << 10;
    int height = 1 << 10;


    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);
    cudaMallocManaged((void**)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }


    // 定义kernel的执行配置
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    tiledMatMulKernel <<< gridSize, blockSize >>>(A, B, C);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    cout << "最大误差: " << maxError << endl;

    return 0;
}