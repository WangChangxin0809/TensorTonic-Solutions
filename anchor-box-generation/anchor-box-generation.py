import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    cx = (np.arange(feature_size) + 0.5) * stride
    cy = (np.arange(feature_size) + 0.5) * stride

    scales_exp = np.expand_dims(scales, axis=1)  # (len(scales), 1)
    ratios_exp = np.expand_dims(aspect_ratios, axis=0)  # (1, len(ratios))

    w = scales_exp * np.sqrt(ratios_exp)  # (len(scales), len(ratios))
    h = scales_exp / np.sqrt(ratios_exp)  # (len(scales), len(ratios))

    anchors = []
    for y in cy:
        for x in cx:
            for i in range(len(scales)):
                for j in range(len(aspect_ratios)):
                    x1 = x - w[i, j] / 2
                    y1 = y - h[i, j] / 2
                    x2 = x + w[i, j] / 2
                    y2 = y + h[i, j] / 2
                    anchors.append([x1, y1, x2, y2])

    #anchors = np.array(anchors)
    return anchors

