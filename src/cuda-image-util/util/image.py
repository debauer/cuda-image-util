
import numpy as np


EPSILON = 20
def compare_images(
    image_benchmark,
    image_to_test
) -> float:
    assert image_to_test.shape == image_benchmark.shape

    diff_black_white = (
        np.linalg.norm(image_to_test - image_benchmark, axis=2) > EPSILON
    ).astype(np.uint8)
    n_different = np.sum(diff_black_white)

    diff_fraction = n_different / image_benchmark.size
    return f"{diff_fraction:.2f}"

