import numpy as np
import cv2
import subspace as sub


def _inverseEval(ba, y):
  # ax + b + y = 0
  # x = (-b - y) / a
  if ba[1] == 0:
    return np.nan
  return (-y - ba[0]) / ba[1]


def _eval(ba, x):
  # ax + b + y = 0
  # y = -ax - b
  if np.isinf(ba[0]) or np.isinf(ba[1]):
    return np.nan
  return -ba[1] * x - ba[0]


def g(image, ba):
  # print("y = {}x + {}".format(ab[0], ab[1]))
  start = (-1.0, _eval(ba, -1.0))
  end = (1.0, _eval(ba, 1.0))

  if abs(start[1]) > 1:
    start = (_inverseEval(ba, np.sign(start[1])), np.sign(start[1]))

  if abs(end[1]) > 1:
    end = (_inverseEval(ba, np.sign(end[1])), np.sign(end[1]))

  if abs(start[1]) > 1 or abs(start[0]) > 1:
    return

  if abs(end[1]) > 1 or abs(end[0]) > 1:
    return

  if np.isnan(start[0]) or np.isnan(start[1]) or \
     np.isnan(end[0]) or np.isnan(end[1]):
    return

  resolution = sub.subspace_resolution(image)

  # ab = sub.subspace_pos(image.shape[0], ab)

  # print(start, end, end=' -> ')

  start = sub.subspace_pos(resolution, start)
  end = sub.subspace_pos(resolution, end)

  # print(start, end)

  cv2.line(image, start, end, (0, 0, 255), 1)
