import cupy as cp
from cupyx import jit

@jit.rawkernel()
def elementwise_copy(input, output):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x * jit.gridDim.x
    max = jit.blockDim.x * jit.threadDim.x * jit.gridDim.x
    for i in range(tid, max, ntid):
        output[i] = input[i]


@jit.rawkernel()
def transform(coord, input, output, xsize, total_size_input, total_size_output):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x

    for i in range(tid * 3, total_size_input -3, ntid * 3):
        pixel_start = ((coord[i+1] * xsize) + coord[i]) * 3
        if pixel_start + 3 < total_size_output and pixel_start >= 0:
            output[pixel_start] = input[i]
            output[pixel_start + 1] = input[i + 1]
            output[pixel_start + 2] = input[i + 2]


@jit.rawkernel()
def transform_with_1doffset(offset, input, output, total_size_input, total_size_output):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x * jit.gridDim.x
    max = total_size_output if total_size_output < total_size_input else total_size_input
    for i in range(tid, max, ntid):
        output[offset[i]] = input[i]


@jit.rawkernel()
def offset_cuda(coord, output, xsize, ysize, total_size_input, total_size_output):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x * jit.gridDim.x
    for i in range(tid * 3, total_size_input, ntid * 3):
        if coord[i]*3 < xsize and coord[i+1]*3 < ysize and coord[i] > 0 and coord[i+1] > 0:
            pixel_start = ((coord[i] * xsize) + coord[i + 1]) * 3
            max = (coord[i] * xsize) + xsize
            min = (coord[i] * xsize)
            if i + 2 < total_size_output and pixel_start >= 0:
                if pixel_start > min and pixel_start < max:
                    output[i] = pixel_start
                    output[i + 1] = pixel_start + 1
                    output[i + 2] = pixel_start + 2

