import numpy as np


def coordinates_matrix(width: int, height: int):
    arr = []
    for y in range(height):
        inner = []
        for x in range(width):
            inner.append([x, y, 1])
        arr.append(inner)
    return np.array(arr)


def translation_matrix(matrix, input, corrected=True):
    old_shape = input.shape
    input = input.reshape((int(input.size / 3), 3))
    translation = np.array([matrix @ x for x in input])
    if corrected:
        return np.array([[x[0]/x[2],x[1]/x[2],1] for x in translation]).reshape(old_shape).astype(int)
    return translation.reshape(old_shape).astype(int)


def translation_matrix_dot(matrix):
    old_shape = matrix.shape
    matrix_2d = matrix.reshape((int(matrix.size / 3), 3))
    translation = np.array([np.dot(matrix, x) for x in matrix_2d])
    return translation.reshape(old_shape).astype(int)


def translation_matrix_einsum(matrix):
    return np.einsum("ijk,lk->ijl", matrix, matrix).astype(int)

#
