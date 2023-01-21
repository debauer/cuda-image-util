
import numpy as np


def scale_points(points, scale):
    output = []
    for point in points:
        output.append(
            [int(point[0] * scale), int(point[1] * scale)]
        )
    return np.float32(output)
