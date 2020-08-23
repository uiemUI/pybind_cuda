#include "kernel.h"
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


__global__ void add_kernel(float *inputleft, float *inputright, float *output, int count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= count)
        return;
    output[idx] = inputleft[idx] + inputright[idx];
}

void addOnCuda(float *inputleft, float *inputright, float* output,int count)
{
    //const int out_n = count;
    //float *output = new float[count];
    //float *output = (float *)malloc(count * sizeof(float));
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, count * sizeof(float));
    cudaMalloc((void **)&d_b, count * sizeof(float));
    cudaMalloc((void **)&d_c, count * sizeof(float));
    cudaMemcpy(d_a, inputleft, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, inputright, count * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadPerBlock(512);
    dim3 blocksPer((count + threadPerBlock.x - 1) / threadPerBlock.x);
    add_kernel<<<blocksPer, threadPerBlock>>>(d_a, d_b, d_c, count);
    cudaMemcpy(output, d_c, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //return output;
}

// int main(int argc, char const *argv[])
// {
//     float left[2] = {1.0,
//                   2.0};
//     float right[2] = {1.0, 2.0};
//     auto result = addOnCuda(left, right, 2);
//     for (size_t i = 0; i < 2; i++)
//     {
//         std::cout << result[i] << std::endl;

//     }
//     free(result);
//     return 0;
// }

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
