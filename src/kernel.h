#pragma once
#ifndef KERNEL_H
#define KERNEL_H
#ifdef _MSC_VER
    #define DLL_EXPORT __declspec (dllexport)
#else
    #define DLL_EXPORT
#endif
#include<iostream>
DLL_EXPORT void addOnCuda(float *inputleft, float *inputright, float *output, int count);

#endif