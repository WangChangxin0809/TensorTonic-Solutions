import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T)  # [4,4]
    points = np.array(points)  # [x,3]
    points = points.reshape(-1,3)
    points = np.hstack([points, np.ones(shape=(len(points), 1))])
    points = points.swapaxes(0,1)

    trans_points = np.dot(T, points)
    trans_points = trans_points[:-1]
    trans_points = trans_points.swapaxes(0,1)
    return trans_points if trans_points.shape[0] > 1 else trans_points[0]