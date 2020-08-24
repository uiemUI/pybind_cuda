import os
os.sys.path.append("../build/bin")
os.sys.path.append("build/bin")

import mylib as my
import time
import numpy as np
import numba


@numba.jit(nopython=True)
def addSum_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape == b.shape
    result = np.zeros(a.shape).astype(np.float32)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i][j] = a[i][j] + b[i][j]
            return result


# @cuda.jit
# def cudaSum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     x,y=cuda.grid(2)

a = np.random.rand(100 * 16, 100 * 16).astype(np.float32)
b = np.random.rand(100 * 16, 100 * 16).astype(np.float32)
hstart = time.time()

h_c = a + b

hcost = time.time() - hstart
print('numpy cost time %f' % (hcost))

dstart = time.time()
d_c = my.np_sum(a, b)
dcost = time.time() - dstart
print("gpu cuda add cost time %f" % (dcost))

print("whether is equal ", (h_c == d_c).any())

n_temp = addSum_numba(a, b)
nstart = time.time()

n_c = addSum_numba(a, b)
ncost = time.time() - nstart
print("numba jit cost time %f" % (ncost))

print("whether is equal ", (h_c == n_c).any())

ta = np.array([[1, 1], [1, 1]]).astype(np.float32)
tb = ta.copy()
tc = my.Gpu_mul(ta, tb)
print(tc)
print(ta @ tb)

txx = np.ones((16, 16)).astype(np.float32)
taa = 2 * txx.copy()
# txx = np.random.rand(5, 5).astype(np.float32)
# taa=

# print(my.Gpu_mul(txx, taa))
tacc = my.Gpu_mul(a, b)

taxx = a @ b
print((taxx == tacc).any())

Ga = np.random.rand(17 * 100, 18 * 100).astype(np.float32)
Gb = np.random.rand(18 * 100, 13 * 100).astype(np.float32)

Gstart = time.time()
Ghc = Ga @ Gb
Gpaush = time.time()
Gc = my.Gpu_Cublas(Ga, Gb)
Gend = time.time()
Gcu = my.Gpu_mul(Ga, Gb)
Gcuend = time.time()

print("the numpy cost", (Gpaush - Gstart))
print("the cublas cost time ", (Gend - Gpaush))
print("the two result is equal? ", (Ghc == Gc).any())
print("the Gpu cost time ", (Gcuend - Gend))
print("the two result is equal? ", (Ghc == Gcu).any())
# print(Gc)
# print(Ghc)

aM = np.random.rand(17, 17).astype(np.float32)
Mb = np.random.rand(17, 17).astype(np.float32)
Gstart = time.time()
GpuResult = my.Gpu_mul(aM, Mb)
Gpaush = time.time()
NResult = aM @ Mb

print((NResult == GpuResult).any())

af = np.array([[1, 2], [3, 4]]).astype(np.float32)
bf = np.array([[1, 2], [3, 4]]).astype(np.float32)
# print(my.Gpu_mul(af,bf))
# print(af@bf)