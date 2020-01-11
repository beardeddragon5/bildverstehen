```python
from typing import NewType, List
import numpy as np
import cv2
# from skimage.feature import peak_local_max
```

```python
subspace_t = NewType('subspace_t', np.array)
subspaces_t = NewType('subspaces_t', List[subspace_t])
```

```python
SUBSPACE_AB = 0
SUBSPACE_1A_BA = 1
SUBSPACE_1B_AB = 2
```

```python
def subspace_create(resolution: int, dtype = np.int32) -> subspace_t:
  subspace_shape = (resolution + 1, resolution + 1)
  return np.zeros(subspace_shape, dtype = dtype)
```

```python
def subspace_resolution(subspace: subspace_t) -> int:
  return subspace.shape[0] - 1
```

```python
def subspace_peaks(subspace: subspace_t, threshold: int) -> List[tuple]:
  out = list()
  resolution = subspace_resolution(subspace)
  for x in range(resolution):
    for y in range(resolution):
      if subspace.item((y, x)) >= threshold:
        ba = subspace_inverse_pos(resolution, (x, y))
        # print("wtf", (x, y), "->", ba)
        out.append(ba)

  return out
```

```python
# def subspace_pos(resolution: int, ba: tuple) -> tuple:
#   b, a = ba
#   off = resolution // 2
#   M = np.array([
#       [off, 0, off],
#       [0, off, off],
#       [0, 0, 1]
#   ])
#   output = (M @ np.array([b, a, 1])).astype('int')
#   return output[:2]
```

```python
def subspace_itemset(subspace: subspace_t, ba: tuple, value: np.uint64):
  x, y = subspace_pos(subspace_resolution(subspace), ba)
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  return subspace.itemset((y, x), value)
```

```python
def subspace_inverse_pos(resolution: int, xy: tuple) -> tuple:
  x, y = xy
  off = resolution // 2
  M_inv = np.array([
      [1/off, 0, -1],
      [0, 1/off, -1],
      [0, 0, 1]
  ])
  out = M_inv @ np.array([x, y, 1])
  return out[:2]  # or out[1], out[0] ???
```

```python
def subspace_axis(resolution: int) -> np.array:
  arr = np.arange(-1, 1, 2.0 / resolution)
  outer = np.array([ resolution / (i * 2) for i in range(1, resolution // 2)])
  return np.concatenate((-outer, arr, np.flip(outer)), axis=0)
```

```python
def subspaces_create(resolution: int, dtype = np.int32) -> subspaces_t:
  spaces = [SUBSPACE_AB, SUBSPACE_1A_BA, SUBSPACE_1B_AB]
  return [subspace_create(resolution, dtype) for i in spaces]
```

```python
def subspaces_from_image(resolution: int, image):
  w, h = image.shape
  if w != resolution or h != resolution:
    raise ValueError("image must be of size {0}x{0}".format(resolution))
  src = subspaces_create(resolution)
  
  np.add(src[0][:resolution,:resolution], image, out=src[0][:resolution,:resolution])
  return src
```

```python
def subspaces_resolution(subspaces: subspaces_t) -> int:
  return subspace_resolution(subspaces[SUBSPACE_AB])
```

```python
def subspaces_item(subspaces: subspaces_t, ba: tuple) -> np.uint8:
  x, y, sub_id = subspaces_pos(subspaces, ba)
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  return subspaces[sub_id].item((y, x))
```

```python
def subspaces_itemset(subspaces: subspaces_t, ba: tuple, value: np.uint8):
  x, y, sub_id = subspaces_pos(subspaces, ba)
  if sub_id == -1:
    return
  # print(ab, '->', (a, b, sub_id), '->', subspaces[sub_id].item(a, b))
  subspaces[sub_id].itemset((y, x), value)
```

```python
def subspaces_max(subspaces: subspaces_t) -> int:
  out = 0
  for subspace in subspaces:
    t, max_value, t1, t2 = cv2.minMaxLoc(subspace)
    if max_value > out:
      out = max_value
  return out
```

```python
def subspaces_add_to(subspaces: subspaces_t, value: subspaces_t):
  for i, space in enumerate(subspaces):
    np.add(space, value[i], out=space)
```

```python
def subspaces_to_line(subspaces: subspaces_t, threshold: float) -> tuple:
  if type(threshold) == float:
    max_value = subspaces_max(subspaces)
    print("use threshold=({} * {})".format(max_value, threshold), end="=")
    threshold = int(max_value * threshold)
    print(threshold)
  
  # print("(b, a):")
  out = subspace_peaks(subspaces[SUBSPACE_AB], threshold)
  # print("(b/a, 1/a):")
  ba_locs = subspace_peaks(subspaces[SUBSPACE_1A_BA], threshold)
  for ba in ba_locs:
    if ba[1] != 0:
      frac_b_a = ba[0]
      frac_1_a = ba[1]
      out.append((frac_b_a / frac_1_a, 1 / frac_1_a))
  # print("(a/b, 1/b):")
  ba_locs = subspace_peaks(subspaces[SUBSPACE_1B_AB], threshold)
  for ba in ba_locs:
    if ba[1] != 0:
      frac_a_b = ba[0]
      frac_1_b = ba[1]
      out.append((1 / frac_1_b, frac_a_b / frac_1_b))

  return out
```

```python
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

  output = output[:2].astype('int')

  return np.r_[output, subspace]
```

```python
if __name__ == '__main__':
  ss = subspaces_create(256)

  print(subspaces_pos(ss, (-1, -1)))
  print(subspaces_pos(ss, (0, 0)))
  print(subspaces_pos(ss, (1, 1)))

  print(subspace_inverse_pos(256, (256, 256)))
  print(subspace_inverse_pos(256, (128, 128)))
  print(subspace_inverse_pos(256, (0, 0)))
  
  print(subspaces_pos(ss, (-0.17, -0.35))[:2], 256//2)
  print(subspace_inverse_pos(256, subspaces_pos(ss, (-0.17, -0.35))[:2]))
  
  ss = subspaces_create(7)
  ss[0].itemset((2, 1), 1)
  display(subspace_peaks(ss[0], 1))
  display(subspaces_item(ss, (-0.666, -0.3333)))
```

```python

```

