from __future__ import annotations
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
from cupyx import jit
import timeit
import cv2
import inspect

from cv2 import (
    COLOR_BGR2GRAY,
    INTER_AREA,
    THRESH_BINARY,
    bitwise_not,
    bitwise_or,
    cvtColor,
    imread,
    resize,
    threshold,
    warpPerspective,
    INTER_NEAREST,
    getPerspectiveTransform
)

from matrix import coordinates_matrix, translation_matrix_dot, translation_matrix
from kernels import offset_cuda, transform_with_1doffset, transform
from util import scale_points


def sand(image, matrix, dimensions):
    return warpPerspective(
        image,
        matrix,
        dimensions,
        flags=INTER_NEAREST,
    )


image_scale = 0.3
map_scale = 0.1
width = int(3840 * image_scale)
height = int(2160 * image_scale)
output_width = int(7000 * map_scale)
output_height = int(8000 * map_scale)

input_points = scale_points([[0, 0], [0, 2160], [3840, 2160], [3840, 0]], image_scale)
output_points = scale_points([[76, -913], [1786, 2880], [5785, 2829], [7295, -848]], map_scale)
coord_matrix = coordinates_matrix(width, height)
transformation_matrix = getPerspectiveTransform(input_points, output_points, )

image_cv = cv2.resize(cv2.imread('f1_l3_v1.jpg'), dsize=(width, height), interpolation=cv2.INTER_CUBIC)

translation_matrix = translation_matrix(transformation_matrix, coord_matrix)

coord = cp.array(translation_matrix)
coord_shape = coord.shape
coord = coord.reshape(coord.size)

cv2.imwrite("input.jpg", image_cv)
input_image = cp.array(image_cv)
input_shape = input_image.shape
input_image = input_image.reshape(input_image.size)

output_shape = (output_height, output_width, 3)
output = cp.zeros(output_height * output_width * 3)
output = output.astype(int)

blocksize = 1000
cores = 256
times = 10

offsets = cp.zeros(input_image.size)
offset_cuda((blocksize,), (cores,), (coord, offsets, output_shape[1], output_shape[0], input_image.size, output.size))
offsets = offsets.astype(int)

print(benchmark(
    sand,
    (image_cv, transformation_matrix, [output_shape[1], output_shape[0]]),
    n_repeat=times
))
output_cv = sand(image_cv, transformation_matrix, [output_shape[1], output_shape[0]])
cv2.imwrite("output_cv.jpg", output_cv)


print(benchmark(
    transform,
    ((blocksize,), (cores,),(coord, input_image, output, output_shape[1], input_image.size, output.size)),
    n_repeat=times
))
cv2.imwrite("output_transform.jpg", output.reshape(output_shape).get())

for blocksize in range(500,2000,500):
    for cores in range(256,1024,256):
        print(blocksize, cores, benchmark(
            transform_with_1doffset,
            ((blocksize,), (cores,), (offsets, input_image, output, input_image.size, output.size)),
            n_repeat=times
        ))
        cv2.imwrite("output_transform_with_1doffset.jpg", output.reshape(output_shape).get())
