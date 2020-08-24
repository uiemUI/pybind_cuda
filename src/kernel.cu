#include "kernel.h"
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cublas_v2.h>
__global__ void add_kernel(float *inputleft, float *inputright, float *output, int count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= count)
        return;
    output[idx] = inputleft[idx] + inputright[idx];
}

template <int TILE_WIDTH>
__global__ void matrix_MulG(float *_A, float *_B, float *_C, int M, int K, int N)
{
    __shared__ float subM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subN[TILE_WIDTH][TILE_WIDTH];
    int x = threadIdx.x + blockIdx.x * blockDim.x; // 17
    int y = threadIdx.y + blockIdx.y * blockDim.y; //y为行，x为列 //0
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //int Tp = TILE_WIDTH;//调试用
    // if (y >= M || x >= N)
    //     {
    //         subM[ty][tx] = 0.;
    //         subN[ty][tx] = 0.;
            
    //         return;
    //     }
    float tmp = 0.;
    for (int i = 0; i < ((K + TILE_WIDTH - 1) / TILE_WIDTH); ++i)
    {
        if ((tx + i * TILE_WIDTH) < K && (y<M))
            subM[ty][tx] = _A[y * K + tx + i * TILE_WIDTH];
        else
        {
            subM[ty][tx] = 0.;
        }
        if ((ty + i * TILE_WIDTH) < K && (x<N))
            subN[ty][tx] = _B[(ty + i * TILE_WIDTH) * N + x];
        else
        {
            subN[ty][tx] = 0.;
        }
        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; ++j)
        {
            tmp += subM[ty][j] * subN[j][tx];
        }
        __syncthreads();
    }
    if(y<M && x<N)  //天啦停止条件在最后
        _C[y * N + x] = tmp; 

}


void addOnCuda(float *inputleft, float *inputright, float *output, int count)
{
    //const int out_n = count;
    //float *output = new float[count];
    //float *output = (float *)malloc(count * sizeof(float));
    constexpr const int NSTREAM = 2;
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, count * sizeof(float));
    cudaMalloc((void **)&d_b, count * sizeof(float));
    cudaMalloc((void **)&d_c, count * sizeof(float));
    // cudaMemcpy(d_a, inputleft, count * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, inputright, count * sizeof(float), cudaMemcpyHostToDevice);
    CHECK(cudaMemcpyAsync(d_a, inputleft, count * sizeof(float), cudaMemcpyHostToDevice, stream[0]));
    CHECK(cudaMemcpyAsync(d_b, inputright, count * sizeof(float), cudaMemcpyHostToDevice, stream[1]));
    dim3 threadPerBlock(512);
    dim3 blocksPer((count + threadPerBlock.x - 1) / threadPerBlock.x);
    add_kernel<<<blocksPer, threadPerBlock>>>(d_a, d_b, d_c, count);
    CHECK(cudaMemcpyAsync(output, d_c, count * sizeof(float), cudaMemcpyDeviceToHost, stream[1]));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamDestroy(stream[i]));
    }
    //return output;
}

void useCublas(float *_A, float *_B, float *_C, int M, int K, int N)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1, beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, _B, N, _A, K, &beta, _C, N);
}
void Gpu_mul(float *ptrLeft, float *ptrRight, float *ptrResult, int M, int K, int N,int flage)
{
    

    constexpr const int NSTREAM = 2;
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, K * N * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));
    // cudaMemcpy(d_a, inputleft, count * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, inputright, count * sizeof(float), cudaMemcpyHostToDevice);
    CHECK(cudaMemcpyAsync(d_a, ptrLeft, M * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]));
    CHECK(cudaMemcpyAsync(d_b, ptrRight, K * N * sizeof(float), cudaMemcpyHostToDevice, stream[1]));
    // CHECK(cudaMemcpy(d_a, ptrLeft, M * K * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_b, ptrRight, K * N * sizeof(float), cudaMemcpyHostToDevice));
    constexpr const int TP = 16;
    dim3 threadsPer(TP, TP);
    dim3 blocksPer((M + TP - 1) / TP, (N + TP - 1) / TP);
    if(flage)
    {matrix_MulG<TP><<<blocksPer, threadsPer>>>(d_a, d_b, d_c, M, K, N);}
    else
    {
        useCublas(d_a, d_b, d_c, M, K, N);
    }
    
    //useCublas(d_a, d_b, d_c,M,K,N);
    //matr_MulG<TP><<<blocksPer, threadsPer>>>(d_a, d_b, d_c, M, K, N);
    //CHECK(cudaMemcpy(ptrResult, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(ptrResult, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream[1]));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamDestroy(stream[i]));
    }
}

void test1()
{
    
    float left[4] = {1., 2., 3., 4.};
    float right[4] = {1.,2.,3.,4.};
    float result[4] = {0.,0.,0.,0.};
    Gpu_mul(left, right, result, 2, 2, 2,1);
    for (size_t i = 0; i < 2; i++)
    {
        std::cout << "[  ";
        for (int j = 0; j < 2; j++)
            std::cout << result[i*2+j] << "  ";
        std::cout << " ] \n";
    }
}
int main(int argc, char const *argv[])
{
    float left[17 * 17] = {0.};
    float right[17*17]={0.};

    for(int i = 0; i <17*17; ++i)
    {
        left[i] = 1.;
        right[i] = 1.;
    }
    //auto result = addOnCuda(left, right, 2);
    float result[17*17] = {0.};
    Gpu_mul(left, right, result, 17,17,17,1);

    for (size_t i = 0; i < 17; i++)
    {
        std::cout << "[  ";
        for (int j = 0; j < 17;j++)
            std::cout << result[i] << "  ";
        std::cout << " ] \n";

    }
    //free(result);
    test1();
    return 0;
}

// int main(int argc, char const *argv[])
// {
//     /* code */
//     float left[2] = {1.0,
//                      2.0};
//     float right[2] = {1.0, 2.0};
//     auto result = addOnCuda(left, right, 2);
//     auto a = py::array_t<float>(2);
//     auto abuf = a.request();
//     abuf.ptr = left;
//     auto b= py::array_t<float>(2);
//     auto bbuf = a.request();
//     bbuf.ptr = right;
//     auto result = addOnCuda(left, right, 2);
//     auto c = py::array_t<float>(2);
//     auto cbuf = a.request();
//     cbuf.ptr = left;
//         for (size_t i = 0; i < 2; i++)
//         {
//             std::cout << c.at<float>(i) << std::endl;

//         }
//         delete[] result;

//     return 0;
// }
