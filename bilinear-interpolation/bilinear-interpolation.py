from math import floor

import numpy as np

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    image = np.array(image)
    h, w = image.shape

    new_image = np.zeros(shape=(new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            src_x = j * (w - 1) / (new_w - 1) if new_w > 1 else 0
            src_y = i * (h - 1) / (new_h - 1) if new_h > 1 else 0

            x0, y0 = floor(src_x), floor(src_y)
            dx, dy = src_x - x0, src_y - y0
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

            new_image[i][j] = image[y0][x0] * (1 - dy) * (1 - dx) + image[y1][x0] * dy * (1 - dx) + image[y0][x1] * (1 - dy) * dx + image[y1][x1] * dy * dx

    return list(new_image)

