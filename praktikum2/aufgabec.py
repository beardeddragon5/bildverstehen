#!/bin/env python3
import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
from skimage.feature import peak_local_max

image_path = "../images/Garagen.jpg"

# +
image_path = os.path.join(os.getcwd(), image_path)
if not os.path.exists(image_path):
  raise SystemError("File {} doesn't exist".format(image_path))

image = cv.imread(image_path)
if image is None:
  raise SystemError("File {} couldn't be loaded".format(image_path))

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()


# +
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
  m = -g[0, 0]/g[0, 1]
  b = -g[0, 2]/g[0, 1]
  return "y = {}x + ({})".format(m, b)

def T(x):
  return np.transpose(x)

def drawG(image, g):
  start = inverseEval(g, 0).astype(int)
  end = inverseEval(g, image.shape[1]).astype(int)

  cv.line(image, (start[0], start[1]), (end[0], end[1]), (1, 1, 0))
  cv.imshow(SELECT_POINTS_WINDOW, image)

# +
def plot_line(axs, I):
  if I[0,1] != 0:
    m = -I[0,0]/I[0,1]
    b = -I[0,2]/I[0,1]
    axs.plot(x, m * x + b)

h, w, channels = image.shape
x = np.arange(3000)

@interact(
  x1 = FloatSlider(value=0, min=0, max=w),
  y1 = FloatSlider(value=0, min=0, max=h),
  x2 = FloatSlider(value=0, min=0, max=w),
  y2 = FloatSlider(value=0, min=0, max=h))
def select_line(x1, y1, x2, y2):
  copy = image.copy()

  start = np.array([x1, y1, 1])
  end = np.array([x2, y2, 1])
  
  Ix = normalize(np.cross(start, end))

  fig, axs = plt.subplots(1, 1, sharey=True)
  fig.suptitle("image", fontsize=32)
  fig.set_size_inches(15, 15)
  axs.set_ylim((h, 0))
  
  axs.imshow(image)
  
  if Ix[1] != 0:
    m = -Ix[0]/Ix[1]
    b = -Ix[2]/Ix[1]
    
    print(Ix) # 50, 88.5, 80.3, 241.9
    axs.plot(x, m * x + b)
    axs.plot(x1, y1, 'ro')
    axs.plot(x2, y2, 'ro')

# +
points = [
  np.array((55, 101, 1)), np.array((86, 271, 1)),
  np.array((514, 266, 1)), np.array((544, 466, 1)),
  np.array((1503, 638, 1)), np.array((1528, 918, 1))
]

lines = [
  np.cross(points[0], points[1]),
  np.cross(points[2], points[3]),
  np.cross(points[4], points[5])
]

def calculate_line():
  global lines, image

  I0 = normalize(lines[0])
  I1 = normalize(lines[1])
  I2 = normalize(lines[2])
  
  P02 = np.cross(I0, I2)
  P02 = (P02 / P02[2]).reshape(1, 3)

  P12 = np.cross(I1, I2)
  P12 = (P12 / P12[2]).reshape(1, 3)

  P01 = np.cross(I0, I1)
  P01 = (P01 / P01[2]).reshape(1, 3)

  P21 = np.cross(I2, I1)
  P21 = (P21 / P21[2]).reshape(1, 3)

  I = ((P02.T @ P12) @ I1 + 2 * (P01.T @ P21) @ I2)
  I = normalize(I)
  V2 = np.cross(I0, I)
  V2 = V2 / V2[2]
  V3 = np.cross(I1, I)
  V3 = V3 / V3[2]
  V4 = np.cross(I2, I)
  V4 = V4 / V4[2]
  I = I.reshape(1, 3)
  
  fig, axs = plt.subplots(1, 1, sharey=True)
  fig.suptitle("image", fontsize=32)
  fig.set_size_inches(15, 15)
  axs.set_ylim((18000, -500))
  axs.set_xlim((-1000, 3000))
  
  axs.imshow(image)
  
  plot_line(axs, I)
  for line in lines:
    plot_line(axs, line.reshape(1, 3))
    axs.plot(P12[0,0], P12[0,1], 'ro')
    axs.plot(P02[0,0], P02[0,1], 'ro')
    axs.plot(P01[0,0], P01[0,1], 'ro')
    axs.plot(P21[0,0], P21[0,1], 'ro')
    
    axs.plot(V2[0], V2[1], 'bo')
    axs.plot(V3[0], V3[1], 'bo')
    axs.plot(V4[0], V4[1], 'bo')
    
  print("I:", I)
  # print("I normalized:", normalize(I))
  print("I in mxb:", toMxb(I))
  print("V2:", V2)
  print("V3:", V3)
  print("V4:", V4)
  print("V2 - V3:", np.linalg.norm(V2 - V3))
  print("V2 - V4:", np.linalg.norm(V2 - V4))
  
if len(lines) == 3:
  calculate_line()
# -


