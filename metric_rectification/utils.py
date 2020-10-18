import numpy as np

def getAffineMatrix(l1, l2, m1, m2):
    v1 = np.cross(l1, l2)
    v2 = np.cross(m1, m2)
    vLine = np.cross(v1, v2)
    vLine[0] = vLine[0] / vLine[2]
    vLine[1] = vLine[1] / vLine[2]
    vLine[2] = 1.0
    h = [[1.0, 0, 0], [0, 1.0, 0], [vLine[0], vLine[1], vLine[2]]]
    return h

def getMetricMatrix(l1, m1, l2, m2):
    m = [[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]], [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]]
    b = [-1.0 * l1[1] * m1[1], -1.0 * l2[1] * m2[1]]
    x = np.linalg.solve(m, b)
    s = [[x[0], x[1]], [x[1], 1.0]]
    a = np.linalg.cholesky(s)
    h = [[a[0][0], a[0][1], 0], [a[1][0], a[1][1], 0], [0, 0, 1.0]]
    h = np.linalg.inv(h)
    return h

def check(minValue, maxValue, value):
    return min(maxValue, max(minValue, value))

def transformPoint(p, h):
    newP = np.matmul(h, p)
    x = float(int(newP[0] / newP[2]))
    y = float(int(newP[1] / newP[2]))
    newP = [x, y, 1.0]
    return newP

def transformImage(image, h):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    newImage = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            p = [float(j), float(i), 1.0]
            newP = np.matmul(h, p)
            x = check(0, width - 1, int(newP[0] / newP[2]))
            y = check(0, height - 1, int(newP[1] / newP[2]))
            newImage[y, x] = image[i, j]
    return newImage