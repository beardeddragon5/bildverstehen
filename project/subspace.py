# -*- coding: utf-8 -*-
from typing import NewType, List
import numpy as np
import cv2


# Um ein kartesisches Produkt zweier Mengen zu erhalten wird hier diese sehr einfache Funktion erstellt.

def cartesian_product(a: np.array, b: np.array):
  return np.array(np.meshgrid(a, b)).T.reshape(-1, 2)


# ### 3.1 Subspaces
# In [1,2] wurden subspaces benutzt um den unendlichen Raum zu diskretisieren. Der Raum wird in 3 subspaces aufgeteilt deren Wertebereich jeweils von [-1, 1] definiert ist. Statt der für die Hough Transformation übliche
# Darstellung von Geraden wird die Steigungs-Achsenabschnitts Form verwendet.
#
# $$
#     ax + b + y = 0
# $$
#
# Wobei $a$ und $b$ jeweils auf einer der 3 Subspaces abgebildet werden.
#
# $$
# subspaces(a, b) = \left\{
# \begin{array}{}
#     subspace_0(a, b), & |a| \leq 1 \land |b| \leq 1 \\
#     subspace_1(\frac 1 a, \frac b a), & |a| \geq 1 \land |b| \leq |a| \\
#     subspace_2(\frac 1 b, \frac a b), & |b| \geq 1 \land |a| \lt |b|
# \end{array}
# \right.
# $$

# +
subspace_t = NewType('subspace_t', np.array)
subspaces_t = NewType('subspaces_t', List[subspace_t])

SUBSPACE_AB = 0
SUBSPACE_1A_BA = 1
SUBSPACE_1B_AB = 2


# -

# Ein Subspace ist dabei ein quadratisches zwei dimensionales Bild aus integer werten. Für die akumulation im während der Hough Transformation wird ein großer Datentyp benötigt um einen reibungslose Interaktion zwischen opencv, python und numpy zu ermöglichen wird hier ein 32-bit signed integer verwendet.

# +
def subspace_create(resolution: int, dtype = np.int32) -> subspace_t:
  subspace_shape = (resolution, resolution)
  return np.zeros(subspace_shape, dtype = dtype)

def subspaces_create(resolution: int, dtype = np.int32) -> subspaces_t:
  spaces = [SUBSPACE_AB, SUBSPACE_1A_BA, SUBSPACE_1B_AB]
  return [subspace_create(resolution, dtype) for i in spaces]

def subspace_resolution(subspace: subspace_t) -> int:
  return subspace.shape[0]

def subspaces_resolution(subspaces: subspaces_t) -> int:
  return subspace_resolution(subspaces[SUBSPACE_AB])


# -

# #### Positionierung
# Anhand der oben genannten Regeln zur Verteilung von $a$ und $b$ werden zusammen mit einer Transformationsmatrix $M$ die richtigen Koordinaten und der Subspace ausgewählt.
#
# $$
# subspace_i(a, b) = \left(
# \begin{matrix}
#     \frac {(resolution - 1)} 2 & 0 & \frac {(resolution - 1)} 2 \\
#     0 & \frac {(resolution - 1)} 2 & \frac {(resolution - 1)} 2 \\
#     0 & 0 & 1
# \end{matrix} \right) \cdot \left(
# \begin{matrix}
#     a \\
#     b \\
#     1
# \end{matrix}
# \right)
# $$

def subspace_pos(resolution: int, ab: tuple) -> tuple:
  a, b = ab
  off = (resolution - 1) / 2
  M = np.array([
      [off, 0, off],
      [0, off, off],
      [0, 0, 1]
  ])
  if np.isnan(b) or np.isnan(a):
    return (np.nan, np.nan, -1)
  
  if abs(a) <= 1 and abs(b) <= 1:
    output = (M @ np.array([a, b, 1]))
    subspace = SUBSPACE_AB

  elif abs(a) > 1 and abs(b) <= abs(a):
    output = (M @ np.array([1.0 / a, b / a, 1]))
    subspace = SUBSPACE_1A_BA

  elif abs(b) > 1 and abs(a) < abs(b):
    output = (M @ np.array([1.0 / b, a / b, 1]))
    subspace = SUBSPACE_1B_AB
  
  output = np.round(output[:2]).astype('int')
  return np.r_[output, subspace]


# Die inverse Berechnung von $a$ und $b$ aus einer Koordinate im Subspace ist auch möglich.
#
# $$
# subspaces(i, x, y) = subspace^{-1}(x, y) \rightarrow (k, j) \rightarrow \left\{
# \begin{array}{}
#     (k, j), & \text{für } i = 0 \\
#     \left(\frac 1 k, \frac j k\right), & \text{für } i = 1 \\
#     \left(\frac k j, \frac 1 j\right), & \text{für } i = 2 \\
# \end{array}
# \right.
# $$
#     
# $$
# subspace^{-1}(x, y) = \left(
# \begin{matrix}
#     \frac 2 {(resolution - 1)} & 0 & -1 \\
#     0 & \frac 2 {(resolution - 1)} & -1 \\
#     0 & 0 & 1
# \end{matrix} \right) \cdot \left(
# \begin{matrix}
#     x \\
#     y \\
#     1
# \end{matrix}
# \right)
# $$

def subspace_inverse_pos(resolution: int, subspace: int, xy: tuple) -> tuple:
  x, y = xy
  off = 2 / (resolution - 1)
  M_inv = np.array([
      [off, 0, -1],
      [0, off, -1],
      [0, 0, 1]
  ])
  i, j, _ = M_inv @ np.array([x, y, 1])
  
  if subspace == SUBSPACE_AB:
    return (i, j)
  
  if subspace == SUBSPACE_1A_BA:
    frac_1_a, frac_b_a = i, j
    return (1 / frac_1_a, frac_b_a / frac_1_a)
  
  if subspace == SUBSPACE_1B_AB:
    frac_1_b, frac_a_b = i, j
    return (frac_a_b / frac_1_b, 1 / frac_1_b)
  
  raise ValueError("invalid subspace: " + subspace)


# Um die Subspaces zu durchlaufen wird für jede Subspace Zelle ein entsprechender Wert und die dazugehörige größe berechnet. Dabei wird zunächst die innere Achse (Subspace 0) berechnet.
#
# $$
#     \text{inner} = \left[ \frac {2i} r \right]_{\frac {\text{resolution}} 2 \leq i \lt \frac {\text{resolution}} 2} \\
#     \text{offset_inner} = \left[ \frac 1 {(\text{resolution} - 1) / 2} \right]_{i \lt \text{resolution}}
# $$
#
# Dabei muss `inner` noch zusätzlich um `offset_inner` verschoben werden da in `inner` nur die entsprechenden Grenzen zwischen Zellübergängen enthalten sind.
#
# Für Subspace 1 und 2 muss die Achse anders berechnet werden zunächst wird die halbe Achse berechnet
#
# $$
#     outer = \left[ \frac {resolution - 1} {2i} \right]_{1\leq i \lt \frac {resolution} 2} \\
#     \text{offset_outer} = \left[ \frac {resolution - 1} {4i^2 + 4i} \right]_{1\leq i \lt \frac {resolution} 2}
# $$
#
# Bei `outer` handelt es sich wieder um die Grenzen zwischen den Zellen die mit `offset_outer` entsprechend zentriert werden müssen. Letztendlich muss die Achse nur noch zusammengesetzt werden.

def subspace_axis(resolution: int) -> np.array:
  r = resolution - 1
  borders = resolution // 2
  
  inner = np.array([ i * (2 / r) for i in range(-borders, borders)])
  offset_inner = np.full(resolution, 1.0 / r)
  inner = inner + offset_inner
  
  outer = np.array([ r / (i * 2) for i in range(1, borders)])
  offset_outer = np.array([r / (4 * (i**2) + 4 * i) for i in range(1, borders)])  
  outer = np.concatenate((np.array([r / 2 + r / 8 ]), outer - offset_outer))
  
  return (
    np.concatenate((-outer, inner, np.flip(outer)), axis=0),
    np.concatenate(([r / 8], offset_outer, offset_inner, np.flip(offset_outer), [r / 8]), axis=0)
  )


# Berechnen der Fehlertoleranz die für einen bestimmten Punkt in den Subspaces gilt.

def subspace_offset(resolution: int, a: float, b: float) -> tuple:
  values, offsets = subspace_axis(resolution)
  
  index_a = np.where((a >= (values - offsets)) & (a <= (values + offsets)))
  index_b = np.where((b >= (values - offsets)) & (b <= (values + offsets)))
  if index_a[0].size == 0 and abs(a) > (values + offsets).max():
    offset_a = offsets[0]
  else:
    offset_a = np.take(offsets, index_a).max()

  if index_b[0].size == 0 and abs(b) > (values + offsets).max():
    offset_b = offsets[0]
  else:
    offset_b = np.take(offsets, index_b).max()
  
  return (offset_a, offset_b)


# #### Subspace Bild
# Ein passendes Bild kann in den ersten Subspace eingefügt werden um dann weiter verarbeitet zu werden.

def subspaces_from_image(resolution: int, image: np.array):
  w, h = image.shape
  if w != resolution or h != resolution:
    raise ValueError("image must be of size {0}x{0}".format(resolution))
  src = subspaces_create(resolution)
  np.add(src[0][:resolution,:resolution], image, out=src[0][:resolution,:resolution])
  return src


# Um einfach Werte aus dem Subspaces zu holen und zu setzen werden entsprechend 2 Hilfsfunktionen erstellt die
# mit $(a, b)$ Tupeln arbeiten können.

# +
def subspaces_item(subspaces: subspaces_t, ab: tuple) -> np.uint8:
  x, y, sub_id = subspace_pos(subspaces_resolution(subspaces), ab)
  return subspaces[sub_id].item((y, x))

def subspaces_itemset(subspaces: subspaces_t, ab: tuple, value: np.uint8):
  x, y, sub_id = subspace_pos(subspaces_resolution(subspaces), ab)
  subspaces[sub_id].itemset((y, x), value)


# -

# Um weiter mit Subspaces einfach arbeiten zu können sind weitere Hilfsfunktionen erstellt worden die Entsprechend zum addieren bzw. zur analyse von Subspaces verwendet werden können.

# +
def subspaces_add_to(subspaces: subspaces_t, value: subspaces_t):
  for i, space in enumerate(subspaces):
    np.add(space, value[i], out=space)
    
def subspaces_max(subspaces: subspaces_t) -> np.int32:
  return np.max([space.max() for space in subspaces])

def subspaces_peaks(subspaces: subspaces_t, threshold: np.int32) -> List[tuple]:
  resolution = subspaces_resolution(subspaces)
  out = []
  for i, space in enumerate(subspaces):
    for y, x in np.array(np.where(space >= threshold)).T.reshape(-1, 2):
      out.append(subspace_inverse_pos(resolution, i, (x, y)))
  return out


# -

if __name__ == '__main__':
  ss = subspaces_create(256)

  #print(subspaces_pos(ss, (-1, -1)))
  #print(subspaces_pos(ss, (0, 0)))
  #print(subspaces_pos(ss, (1, 1)))

  #print(subspace_inverse_pos(256, (255, 255)))
  #print(subspace_inverse_pos(256, (128, 128)))
  #print(subspace_inverse_pos(256, (0, 0)))
  
  #print(subspaces_pos(ss, (-0.17, -0.35))[:2], 256//2)
  #print(subspaces_pos(ss, (-0.17, -0.35))[:2], 256//2)
  #print(subspace_inverse_pos(256, subspaces_pos(ss, (-0.17, -0.35))[:2]))
  
  #for v in subspace_axis(256):
  #  print(subspaces_pos(ss, (v, 0)))
  
  ss = subspaces_create(7)
  ss[0].itemset((2, 1), 1)
  #print(ss[0])
  #display(subspace_peaks(ss[0], 1))
  #display(subspaces_item(ss, (-0.666, -0.3333)))
  
  values, offsets = subspace_axis(10)
  print(*[(x, offsets[i]) for i, x in enumerate(values)])
  #print([subspaces_pos(ss, (0, x)) for x in values])
  #print(values, offsets)





