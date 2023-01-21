import cupy
from cupyx import jit
from cupyx.profiler import benchmark

@jit.rawkernel()
def reduction(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x * jit.gridDim.x

    value = cupy.float32(0)
    for i in range(tid, size, ntid):
        value += x[i]

    smem = jit.shared_memory(cupy.float32, 1024)
    smem[jit.threadIdx.x] = value

    jit.syncthreads()

    if jit.threadIdx.x == cupy.uint32(0):
        value = cupy.float32(0)
        for i in range(jit.blockDim.x):
            value += smem[i]
        jit.atomic_add(y, 0, value)


size = cupy.uint32(2 ** 20)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.empty((1,), dtype=cupy.float32)

times = 1000
print(benchmark(reduction, ((64,), (1024,),(x, y, size)), n_repeat=times))
print(y[0])
print(benchmark(x.sum, n_repeat=times))
print(x.sum())
