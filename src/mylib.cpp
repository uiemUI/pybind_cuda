#include<iostream>
#include<string>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include"kernel.h"
namespace py = pybind11;

// float *addLoop(float *inputleft, float *inputright, int count)
// {
// }
py::array_t<float> np_sum(py::array_t<float>& input1, py::array_t<float>& input2)
{
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();
    assert(buf1.ndim == buf2.ndim);
    for (int i = 0; i < buf1.ndim; ++i)
        assert(buf1.shape[i] == buf2.shape[i]);
    auto result = py::array_t<float>(buf1.size);
    result.resize(buf1.shape);
    py::buffer_info buf3 = result.request();
    float *ptr1 = (float *)buf1.ptr,
          *ptr2 = (float *)buf2.ptr,
          *ptr3 = (float *)buf3.ptr;
    

    //ptr3 = addOnCuda(ptr1, ptr2, buf1.size);
    addOnCuda(ptr1, ptr2, ptr3, buf1.size); //主要是内存分配问题，特别注意不要随意换ptr
    // for (int i = 0; i < buf1.size;++i)
    // {
    //     ptr3[i] = ptr1[i] + ptr2[i];
    // }
    
    return result;
}
py::array_t<float> np_multiply_Cublas(py::array_t<float> &inLeft, py::array_t<float> &inRight)
{
    py::buffer_info bufLeft = inLeft.request(), bufRight = inRight.request();
    assert(bufLeft.ndim == bufRight.ndim);
    assert(bufLeft.ndim == 2);
    assert(bufLeft.shape[1] == bufRight.shape[0]);
    const int M = bufLeft.shape[0], K = bufLeft.shape[1], N = bufRight.shape[1];
    auto result = py::array_t<float>(M * N);
    result.resize({M, N});

    py::buffer_info bufResult = result.request();
    float *ptrLeft = (float *)bufLeft.ptr,
          *ptrRight = (float *)bufRight.ptr,
          *ptrResult = (float *)bufResult.ptr;
    Gpu_mul(ptrLeft, ptrRight, ptrResult, M, K, N);
    return result;
}

py::array_t<float> np_multiply(py::array_t<float> &inLeft, py::array_t<float> &inRight)
{
    py::buffer_info bufLeft = inLeft.request(), bufRight = inRight.request();
    assert(bufLeft.ndim == bufRight.ndim);
    assert(bufLeft.ndim == 2);
    assert(bufLeft.shape[1] == bufRight.shape[0]);
    const int M = bufLeft.shape[0], K = bufLeft.shape[1], N = bufRight.shape[1];
    auto result = py::array_t<float>(M * N);
    result.resize({M, N});

    py::buffer_info bufResult = result.request();
    float *ptrLeft = (float *)bufLeft.ptr,
          *ptrRight = (float *)bufRight.ptr,
          *ptrResult = (float *)bufResult.ptr;
    Gpu_mul(ptrLeft, ptrRight, ptrResult, M, K, N,1);
    return result;
}
PYBIND11_MODULE(mylib, m)
{
   // m.doc("use cuda and demo");

    m.def("np_sum", &np_sum, "Add two Numpy arrays use cuda");
    m.def("Gpu_mul", &np_multiply, "Multuply tow arrays use cuda");
    m.def("Gpu_Cublas", &np_multiply_Cublas, "Multuply tow arrays use cublas");
}

