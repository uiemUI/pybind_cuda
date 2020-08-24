from numba import cuda,float32
import numba
import numpy as np
import numpy 
import math
import time

TPB = 16

@numba.jit(nopython=True)
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            temp = 0.
            for k in range(A.shape[1]):
                temp = temp + A[x][k] * B[k][y]
            C[x][y] = temp


@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        temp = 0.
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col]
        C[row, col] = temp

@cuda.jit
def matmul_shared_mem(A, B, C: numpy.ndarray):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32) # shared_memory
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    x, y = cuda.grid(2) #x,y在grid中的

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tmp = 0.
    for i in range(int(math.ceil(A.shape[1] / TPB))):
        if ((ty+i*TPB)<A.shape[1] ):
            sA[tx, ty] = A[x, ty + i * TPB]
        else:
            sA[tx, ty] = 0.
        if((tx + i * TPB)<B.shape[0]):
            sB[tx, ty] = B[tx + i * TPB, y]
        else:
            sB[tx,ty]= 0.

        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j,ty]
        cuda.syncthreads()
    if x <C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp


# def matmul_shared_mem(A, B, C):
#     A = numpy.full((TPB * 50, TPB * 50), 3, numpy.float)
#     B=numpy
# A = numpy.full((3, 4),2)
# B = numpy.full((4, 5), 2)
# print(A @ B)
# C = numpy.full((3, 5), 1)
# matmul_cpu(A, B, C)
# print(C)
# A = numpy.full((15*100, 15*100), 3, dtype=numpy.float)
# B = numpy.full((15*100, 15*100 ), 4, numpy.float)
A= np.array([[1, 2], [3, 4]]).astype(np.float32)
B = np.array([[1, 2], [3, 4]]).astype(np.float32)
C_cpu = numpy.full((A.shape[0], B.shape[1]), 0, numpy.float)
print("start processing in CPU")
start_cpu = time.time()
matmul_cpu(A, B, C_cpu)
time_cpu = time.time() - start_cpu
print("CPU time: " + str(time_cpu))
start_numpy = time.time()
C_numpy = A @ B
time_numpy = time.time() - start_numpy
print("NUMPY time  " + str(time_numpy))

print("Start processint in GPU")
A_global_mem = cuda.to_device(A)  # 复制到GPU的global memory
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))
C_shared_mem = cuda.device_array((A.shape[0], B.shape[1]))
# 先在GPU开辟C的内存空间
threadPerBlock = (TPB, TPB)  # BLock 维度
blockPerGrid = (int(math.ceil(A.shape[0] / threadPerBlock[0])), int(math.ceil(B.shape[1] / threadPerBlock[1])))
start_gpu = time.time()
matmul_gpu[blockPerGrid, threadPerBlock](A_global_mem, B_global_mem, C_global_mem)
cuda.synchronize() # 同步
gpu_time = time.time() - start_gpu
print("GPU time(global memory: " + str(gpu_time))
C_cpu_global = C_global_mem.copy_to_host()  # 复制到主机
print("whether the two array is equals: " + str((C_numpy == C_cpu_global).all()))

start_shared_time = time.time()
matmul_shared_mem[blockPerGrid, threadPerBlock](A_global_mem, B_global_mem, C_shared_mem)
cuda.synchronize()
shared_time = time.time()-start_shared_time
print("GPU time(Shared memory): " + str(shared_time))
C_cpu_shared = C_shared_mem.copy_to_host()
print("whether the two array is equals: " + str((C_numpy == C_cpu_shared).all()))
print(C_cpu_shared)