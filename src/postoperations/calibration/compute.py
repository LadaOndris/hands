import cv2 as cv
import numpy as np


def deproject_pixel(pixel, intrin, depth):
    ppx, ppy = intrin[0, 2], intrin[1, 2]
    fx, fy = intrin[0, 0], intrin[1, 1]

    x = depth * (pixel[0] - ppx) / fx
    y = depth * (pixel[1] - ppy) / fy
    z = depth
    return [x, y, z]


def deproject_pixels(pixels, depth_intrinsics):
    points = []
    for i in range(pixels.shape[0]):
        depth_pixel_coords = (pixels[i, 0], pixels[i, 1])
        depth_pixel_value = pixels[i, 2]
        point = deproject_pixel(depth_pixel_coords, depth_intrinsics, depth_pixel_value)
        points.append(point)
    return np.array(points)


def solve():
    depth_intrinsics = np.load("../depth_intrinsics.npy")
    color_intrinsics = np.load("../color_intrinsics.npy")
    depth_uvz = np.load("../depth_uvz.npy")
    color_uv = np.load("../color_uv.npy")

    objectPoints = deproject_pixels(depth_uvz, depth_intrinsics).astype(np.float32)  # 3D points (N, 3)
    imagePoints = color_uv.astype(np.float32)  # 2D points in color image (N, 2)
    cameraMatrix = color_intrinsics  # color intrinsics
    distCoeffs = 0  # No distortion

    retval, rvec, tvec = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

    print(retval, rvec, tvec)


def extrinsics():
    """
    Returns
    -------
    Translation is in meters.

    [[ 9.9985880e-01  6.9866995e-03  1.5281244e-02  1.6818050e+01]
     [-6.9994810e-03  9.9997520e-01  7.8308542e-04 -1.3267192e+00]
     [-1.5275395e-02 -8.8993565e-04  9.9988294e-01 -1.1069615e+01]
     [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

    """
    rvec = np.array([[-0.00083655],
                     [0.01527904],
                     [-0.00699342]])
    tvec = np.array([[16.81804967],
                     [-1.32671918],
                     [-11.06961509]]) / 1000 # to meters
    rmat, jacobian = cv.Rodrigues(rvec)
    extr = np.zeros([4, 4], dtype=np.float32)
    extr[0:3, 0:3] = rmat
    extr[0:3, 3:4] = tvec
    extr[3, 3] = 1
    return extr


if __name__ == "__main__":
    extrinsics()
