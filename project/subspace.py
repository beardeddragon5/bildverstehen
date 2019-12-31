from typing import NewType, List
import numpy as np
import cv2
from skimage.feature import peak_local_max

subspace_t = NewType('subspace_t', np.array)
subspaces_t = NewType('subspaces_t', List[subspace_t])

SUBSPACE_AB = 0
SUBSPACE_1A_BA = 1
SUBSPACE_1B_AB = 2


def subspace_create(resolution: int) -> subspace_t:
  subspace_shape = (resolution + 1, resolution + 1)
  return np.zeros(subspace_shape, dtype=np.uint8)


def subspace_resolution(subspace: subspace_t) -> int:
  return subspace.shape[0] - 1


def subspace_peaks(subspace: subspace_t, threshold: int) -> List[tuple]:
  out = list()
  w, h = subspace.shape
  
  for y in range(h):
    for x in range(w):
      if subspace.item((x, y)) >= threshold:
        out.append(subspace_inverse_pos(w, (x, y)))

  return out


def subspace_pos(resolution: int, ba: tuple) -> tuple:
  b, a = ba
  off = resolution // 2
  M = np.array([
      [off, 0, off],
      [0, off, off],
      [0, 0, 1]
  ])
  output = (M @ np.array([b, a, 1]))
  return (output[1].astype('int'), output[0].astype('int'))


def subspace_itemset(subspace: subspace_t, ba: tuple, value: np.uint8):
  xy = subspace_pos(subspace_resolution(subspace), ba)
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  return subspace.itemset(xy, value)


def subspace_inverse_pos(resolution: int, xy: tuple) -> tuple:
  x, y = xy
  off = resolution // 2
  M_inv = np.linalg.inv(np.array([
      [off, 0, off],
      [0, off, off],
      [0, 0, 1]
  ]))
  out = M_inv @ np.array([x, y, 1])
  return (out[0], out[1])  # or out[1], out[0] ???


def subspaces_create(resolution: int) -> subspaces_t:
  spaces = [SUBSPACE_AB, SUBSPACE_1A_BA, SUBSPACE_1B_AB]
  return [subspace_create(resolution) for i in spaces]


def subspaces_resolution(subspaces: subspaces_t) -> int:
  return subspace_resolution(subspaces[SUBSPACE_AB])


def subspaces_item(subspaces: subspaces_t, ba: tuple) -> np.uint8:
  x, y, sub_id = subspaces_pos(subspaces, ba)
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  return subspaces[sub_id].item((x, y))


def subspaces_itemset(subspaces: subspaces_t, ba: tuple, value: np.uint8):
  x, y, sub_id = subspaces_pos(subspaces, ba)
  if sub_id == -1:
    return
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  subspaces[sub_id].itemset((x, y), value)


def subspaces_max(subspaces: subspaces_t) -> int:
  out = 0
  for subspace in subspaces:
    t, max_value, t1, t2 = cv2.minMaxLoc(subspace)
    if max_value > out:
      out = max_value
  return out


def subspaces_add_to(subspaces: subspaces_t, value: subspaces_t):
  for i in range(len(subspaces)):
    np.add(subspaces[i], value[i], out=subspaces[i])


def subspaces_to_line(subspaces: subspaces_t, threshold: float) -> tuple:
  if type(threshold) == float:
    max_value = subspaces_max(subspaces)
    print("use threshold=({} * {})".format(max_value, threshold), end="=")
    threshold = int(max_value * threshold)
    print(threshold)

  out = subspace_peaks(subspaces[SUBSPACE_AB], threshold)
  ba_locs = subspace_peaks(subspaces[SUBSPACE_1A_BA], threshold)
  for ba in ba_locs:
    if ba[1] != 0:
      # 1/a = ab[0]
      # 1/ab[0] = a
      # b/a = ab[1]
      # b = ab[1] * a
      out.append((ba[0] / ba[1], 1 / ba[1]))

  ba_locs = subspace_peaks(subspaces[SUBSPACE_1B_AB], threshold)
  for ba in ba_locs:
    if ba[0] != 0:
      out.append((1 / ba[0], ba[1] / ba[0]))

  return out


def subspaces_pos(subspaces: subspaces_t, ba: tuple) -> tuple:
  b, a = ba
  off = subspaces_resolution(subspaces) // 2
  M = np.array([
      [off, 0, off],
      [0, off, off],
      [0, 0, 1]
  ])
  if np.isnan(b) or np.isnan(a):
    return (np.nan, np.nan, -1)
  
  if abs(a) <= 1 and abs(b) <= 1:
    output = (M @ np.array([b, a, 1]))
    subspace = SUBSPACE_AB

  elif abs(a) > 1 and abs(b) <= abs(a):
    output = (M @ np.array([b / a, 1.0 / a, 1]))
    subspace = SUBSPACE_1A_BA

  elif abs(b) > 1 and abs(a) < abs(b):
    output = (M @ np.array([a / b, 1.0 / b, 1]))
    subspace = SUBSPACE_1B_AB

  x = output[0].astype('int')  # round(output[0]).astype('int')
  y = output[1].astype('int')  # round(output[1]).astype('int')

  return (x, y, subspace)


if __name__ == '__main__':
  ss = subspaces_create(256)

  print(subspaces_pos(ss, (-1, -1)))
  print(subspaces_pos(ss, (0, 0)))
  print(subspaces_pos(ss, (1, 1)))

  print(subspace_inverse_pos(256, (256, 256)))
  print(subspace_inverse_pos(256, (128, 128)))
  print(subspace_inverse_pos(256, (0, 0)))
