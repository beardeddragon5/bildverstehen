#!/usr/bin/env python3

import cv2 as cv
import numpy as np

def show_methods(obj):
  return [method_name for method_name in dir(obj) if callable(getattr(obj, method_name))]

# [ print(method) if 'Mat' in method else "" for method in show_methods(cv) ]

MP = np.matrix([
  [-6.23410,  13.75944, 42.50663, 2143.00000],
  [-10.18040, 19.18011, 50.57156, 3128.00000],
  [-0.20487,  0.31880,  0.92542,  12.00000  ],
])

MSP = np.matrix([
  [33.41169,  51.58735, 21.22862, 3998.50000],
  [8.25988,   12.83230, 3.36994,  4809.00000],
  [0.49034,   0.80382,  0.33682,  22.00000],
])

h = - np.linalg.det(np.concatenate((MP[:,0], MP[:,1], MP[:,2]), axis=1))
x = np.linalg.det(np.concatenate((MP[:,1], MP[:,2], MP[:,3]), axis=1))
y = - np.linalg.det(np.concatenate((MP[:,0], MP[:,2], MP[:,3]), axis=1))
z = np.linalg.det(np.concatenate((MP[:,0], MP[:,1], MP[:,3]), axis=1))

C = np.array([x, y, z, h])

e = MSP @ C
print(e)
E = np.matrix([
  [    0, -e[0, 2],  e[0, 1]],
  [ e[0, 2],     0, -e[0, 0]],
  [-e[0, 1],  e[0, 0],     0]
])

# i) Direkte Invertierung
MPPi = MP.transpose() * np.linalg.inv(MP * MP.transpose())

# ii) Mittels SVD
MPPii = np.zeros((4, 3))
cv.invert(MP, MPPii, flags=cv.DECOMP_SVD)

def normMat(a):
  return np.array(a / np.array(a).sum(axis=0, keepdims=1))

def sumMat(a):
  return np.array(a).sum(0)

Fi = E * MSP * MPPi
Fii = E * MSP * MPPii

Fi = normMat(Fi)
Fii = normMat(Fii)

print('M^+_Pi =', MPPi)
print('M^+_Pii =', MPPii)
print('Fi =', Fi)
print('Fii =', Fii)
print('sum(Fi) =', sumMat(Fi))
print('sum(Fii) =', sumMat(Fii))
print('det(Fi) =', np.linalg.det(Fi))
print('det(Fii) =', np.linalg.det(Fii))

