#!/bin/env python3

import cv2 as cv
import numpy as np
import time
import sys

SELECT_POINTS_WINDOW = 'select points';

def normalize(g):
  norm = np.linalg.norm(np.array([g[0], g[1]]))
  if norm == 0:
    return g
  g = g / norm
  return g

def eval(g, x):
  # y = mx + b
  m = -g[0]/g[1]
  b = -g[2]/g[1]
  return np.array([x, m * x - b])

def inverseEval(g, y):
  # (y - b) / m = x
  m = -g[0]/g[1]
  b = -g[2]/g[1]
  return np.array([(y - b) / m, y])

def toMxb(g):
  m = -g[0]/g[1]
  b = -g[2]/g[1]
  return "y = {}x + ({})".format(m, b)

def T(x):
  return np.transpose(x)

def drawG(image, g):
  start = inverseEval(g, 0).astype(int)
  end = inverseEval(g, image.shape[1]).astype(int)

  cv.line(image, (start[0], start[1]), (end[0], end[1]), (1, 1, 0))
  cv.imshow(SELECT_POINTS_WINDOW, image)

lines = []
dragStart = None

def select_line(event, x, y, flags, param):
  global dragStart, image

  copy = image.copy()

  if event == cv.EVENT_LBUTTONDOWN:
    dragStart = np.array([x, y, 1])

  if event == cv.EVENT_LBUTTONUP:
    dragEnd = np.array([x, y, 1])

    print(dragStart, dragEnd)
    Ix = normalize(np.cross(dragStart, dragEnd))
    lines.append(Ix)

    dragStart = None

    drawG(image, Ix)

    if len(lines) == 3:
      calculate_line()

  if dragStart is not None:
    cv.line(copy, (dragStart[0], dragStart[1]), (x, y), (1, 0, 0))
    cv.imshow(SELECT_POINTS_WINDOW, copy)

  cv.displayOverlay(SELECT_POINTS_WINDOW, "{} {} color: {}".format(x, y, image[y][x]), 0)

def calculate_line():
  global lines, image

  I0 = lines[0]
  I1 = lines[1]
  I2 = lines[2]

  P02 = np.cross(I0, I2)
  P02 = P02 / P02[2]

  P12 = np.cross(I1, I2)
  P12 = P12 / P12[2]

  P01 = np.cross(I0, I1)
  P01 = P01 / P01[2]

  P21 = np.cross(I2, I1)
  P21 = P21 / P21[2]

  I = (T(P02) * P12) * I1 + 2 * (T(P01) * P21) * I2
  I = normalize(I)
  V2 = np.cross(I0, I)
  V3 = np.cross(I1, I)
  V4 = np.cross(I2, I)

  drawG(image, I)

  print("I:", I)
  print("I normalized:", normalize(I))
  print("I in mxb:", toMxb(I))
  print("V2:", V2)
  print("V2 - V3:", np.linalg.norm(V2 - V3))
  print("V2 - V4:", np.linalg.norm(V2 - V4))

image = cv.imread(sys.argv[1])
image = cv.resize(image, None, fx = 0.5, fy = 0.5)
cv.imshow(SELECT_POINTS_WINDOW, image)
cv.setMouseCallback(SELECT_POINTS_WINDOW, select_line)

try:
  cv.waitKey(0)
  cv.destroyAllWindows()
except KeyboardInterrupt:
  cv.destroyAllWindows()
