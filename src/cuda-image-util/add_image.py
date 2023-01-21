from __future__ import annotations
import numpy as np
try:
    import cupy as np
    from cupyx.profiler import benchmark
except ImportError:
    print("CUPY Library not found, use Numpy instead")

import timeit
import cv2

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
)

from numpy.typing import NDArray

from sync.main import compare_images

Image = NDArray[np.Shape["Height, Width"], np.Int8]

# Time:  0.001929166988702491
def add1(a: Image, b: Image, mask: Image):
    a[mask] = b[mask]
    return a

# Time:  0.0008525410084985197
def add2(bg: Image, image: Image, mask: Image):
    return bg + bitwise_or(image, image, mask=bitwise_not(mask))

def add3(a: Image, b: Image, mask: Image):
    return np.where(mask, a, b)


def add4(a: Image, b: Image, mask: Image):
    return cp.where(mask, a, b)


image = cv2.imread('camera_mask/katze_f1.jpg')
image = cv2.resize(image, dsize=(800, 700), interpolation=cv2.INTER_CUBIC)
bg = cv2.imread('camera_mask/katze_f2.jpg')
bg = cv2.resize(bg, dsize=(800, 700), interpolation=cv2.INTER_CUBIC)

times = 1000

mask = (bg == 0)
_th, maskcv = threshold(cvtColor(image, COLOR_BGR2GRAY), 3, 255, THRESH_BINARY)
maskcp = cp.array((bg == 0))
bgcp = cp.array(bg)
imagecp = cp.array(image)

print(benchmark(add1, (bg, image, mask), n_repeat=times))
print(benchmark(add2, (bg, image, maskcv), n_repeat=times))
print(benchmark(add3, (bg, image, mask), n_repeat=times))
print(benchmark(add4, (bgcp, imagecp, maskcp), n_repeat=times))

images = []
images.append(add1(bg, image, mask))
images.append(add2(bg, image, maskcv))
images.append(add3(bg, image, mask))
images.append(add4(bgcp, imagecp, maskcp).get())
print("add1 - add1 ", compare_images(images[0], images[0]), "%")
print("add1 - add2 ", compare_images(images[0], images[1]), "%")
print("add1 - add3 ", compare_images(images[0], images[2]), "%")
print("add1 - add4 ", compare_images(images[0], images[3]), "%")
print("add2 - add3 ", compare_images(images[1], images[2]), "%")
print("add2 - add4 ", compare_images(images[1], images[3]), "%")
print("add3 - add4 ", compare_images(images[2], images[3]), "%")

#cv2.imwrite("add1.jpg", images[0])
#cv2.imwrite("add2.jpg", images[1])
#cv2.imwrite("add3.jpg", images[2])
#cv2.imwrite("add4.jpg", images[3])

# cv2.namedWindow("asd", cv2.WND_PROP_FULLSCREEN)
# cv2.imshow("asd", ret3)
# cv2.waitKey(0)
