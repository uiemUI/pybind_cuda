#pragma once
#ifndef KERNEL_H
#define KERNEL_H
#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

#include <iostream>


DLL_EXPORT void addOnCuda(float *inputleft, float *inputright, float *output, int count);
DLL_EXPORT void Gpu_mul(float *ptrLeft, float *ptrRight, float *ptrResult, int M, int K, int N);
#endif