import numpy as np
from matrix import coordinates_matrix,translation_matrix

matrix = np.array([
    [6,0,1],
    [0,1,0],
    [2,0,1]
])


coord_matrix = coordinates_matrix(3, 3)
translation = translation_matrix(matrix, coord_matrix)
translation2 = translation_matrix(matrix, coord_matrix, False)

print(coord_matrix)
print(translation)
print(translation2)
