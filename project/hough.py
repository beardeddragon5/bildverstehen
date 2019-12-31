# +
# %load_ext autoreload
# %autoreload 2

import subspace as sub
import numpy as np
import cv2
import sys
import multiprocessing as multi
import draw
import time


# -

def _hough_process(args):
  x, y, src = args
  if sub.subspaces_item(src, (x, y)) == 0:
    return None
  
  resolution = sub.subspaces_resolution(src)
  arr = np.arange(-1, 1, 2.0 / resolution)

  nonzero = arr[arr != 0]
  ones = np.ones(len(nonzero), dtype=np.float64)
  arr_reciprocal = np.true_divide(ones, nonzero)

  values = np.concatenate((arr, arr_reciprocal), axis=0)
  
  pixel_subspace = sub.subspaces_create(resolution)
  
  values = np.c_[ values, values, np.ones(len(values)) ]
  
  Ma_iterate = np.array([
      [-x, 0, -y],
      [0, 1, 0],
      [0, 0, 1]
  ])

  ba_array = np.einsum('ij,...j', Ma_iterate, values)[:, :2]
  np.apply_along_axis(lambda ba: sub.subspaces_itemset(pixel_subspace, ba, 1), 1, ba_array)
  
  if x != 0:
    Mb_iterate = np.array([
        [1, 0, 0],
        [0, -1/x, -y/x],
        [0, 0, 1]
    ])
    
    ba_array = np.einsum('ij,...j', Mb_iterate, values)[:, :2]
    np.apply_along_axis(lambda ba: sub.subspaces_itemset(pixel_subspace, ba, 1), 1, ba_array)
  return pixel_subspace


# +
# x = 1
# y = 0
# resolution = 200

# values = np.arange(-1, 1, 2.0 / resolution)
# values = np.c_[ values, values, np.ones(len(values)) ]

# Ma_iterate = np.array([
#     [-x, 0, -y],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# display(values[:5])
# np.einsum('ij,...j', Ma_iterate, values)[:5]

# +
def _subspace_axis(resolution: int) -> np.array:
  arr = np.arange(-1, 1, 2.0 / resolution)
  
  nonzero = arr[arr != 0]
  ones = np.ones(len(nonzero), dtype=np.float64)
  arr_reciprocal = np.true_divide(ones, nonzero)

  return np.concatenate((arr, arr_reciprocal), axis=0)

def _hough_vec_process(input_values):
  resolution = int(input_values[6, 0])
  Ma = input_values[0:3]
  Mb = input_values[3:6]
  
  values = _subspace_axis(resolution)
  values = np.c_[ values, values, np.ones(len(values)) ]
  
  space = sub.subspaces_create(resolution)
  for M in [Ma, Mb]:
    ba_array = np.einsum('ij,...j', M, values)[:, :2]
    ba_array = ba_array[~np.isnan(ba_array[:, 0])]
    if ba_array.size > 0:
      np.apply_along_axis(lambda ba: sub.subspaces_itemset(space, ba, 1), 1, ba_array)
  
  return space
  

def hough_vec(src: sub.subspaces_t) -> sub.subspaces_t:
  resolution = sub.subspaces_resolution(src)
  ss = sub.subspaces_create(resolution)
  values = _subspace_axis(resolution)
  
  prod = np.array(np.meshgrid(values, values)).T.reshape(-1, 2).tolist()  
  
  nan_m = np.array([
        [np.nan, 0, 0],
        [0, np.nan, 0],
        [0, 0, np.nan]
      ])
  
  def value_filter_a_iterate(xy):
    x, y = xy
    if x == 0 or sub.subspaces_item(src, (x, y)) == 0:
      return nan_m
    return np.array([
      [1, 0, 0],
      [0, -1/x, -y/x],
      [0, 0, 1]
    ])
  
  def value_filter_b_iterate(xy):
    x, y = xy
    if sub.subspaces_item(src, (x, y)) == 0:
      return nan_m
    return np.array([
      [-x, 0, -y],
      [0, 1, 0],
      [0, 0, 1]
    ])
  # needs the most amount of time
  Ma_iterate = np.apply_along_axis(value_filter_a_iterate, 1, prod)
  Mb_iterate = np.apply_along_axis(value_filter_b_iterate, 1, prod)

  # Ms = np.concatenate((Ma_iterate, Mb_iterate)).reshape(-1, 3, 3)
  M_iterate_filter = np.all([~np.isnan(Ma_iterate[:,0,0]), ~np.isnan(Mb_iterate[:,0,0])], axis=0)
  
  Ma_iterate = Ma_iterate[M_iterate_filter].reshape(-1, 3, 3)
  Mb_iterate = Mb_iterate[M_iterate_filter].reshape(-1, 3, 3)
  M_iterate = np.column_stack((Ma_iterate, Mb_iterate, np.full((len(Ma_iterate), 3, 3), resolution)))

  with multi.Pool(multi.cpu_count()) as p:
    chunksize = len(M_iterate) // multi.cpu_count()
    iterator = p.imap_unordered(_hough_vec_process, M_iterate, chunksize = chunksize)
    
    for i, space in enumerate(iterator):
      sub.subspaces_add_to(ss, space)

  return ss


# -

def hough(src: sub.subspaces_t) -> sub.subspaces_t:
  resolution = sub.subspaces_resolution(src)
  ss = sub.subspaces_create(resolution)
  arr = np.arange(-1, 1, 2.0 / resolution)
  
  # cartesian of x an y
  prod = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2).tolist()
  for p in prod:
    p.append(src)
    
  print("done with setup")

  with multi.Pool(multi.cpu_count()) as p:
    chunksize = len(prod) // multi.cpu_count()
    iterator = p.imap_unordered(_hough_process, prod, chunksize = chunksize)
    for i, space in enumerate(iterator):
      if space is not None:
        sub.subspaces_add_to(ss, space)

  return ss


def from_image(resolution: int, image):
  w, h = image.shape
  if w != resolution or h != resolution:
    raise ValueError("image must be of size {0}x{0}".format(resolution))
  src = sub.subspaces_create(resolution)
  
  np.add(src[0][:resolution,:resolution], image, out=src[0][:resolution,:resolution])
  return src


if __name__ == '__main__':
  resolution = 256
  image = cv2.imread(sys.argv[1])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  src = from_image(resolution, image)
  print("convert image")

  # cv2.imshow('gray', src[0])
  ss = hough(src)
  print("hough done")

  # sub.subspaces_to_line_fast(ss)

  out = cv2.cvtColor(src[0], cv2.COLOR_GRAY2BGR)

  [draw.g(out, ab) for ab in sub.subspaces_to_line(ss, 0.9)]

  cv2.imshow('subspace 1', cv2.normalize(
      ss[0], None, alpha=0, beta=255,
      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

  cv2.imshow('subspace 2', cv2.normalize(
      ss[1], None, alpha=0, beta=255,
      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

  cv2.imshow('subspace 3', cv2.normalize(
      ss[2], None, alpha=0, beta=255,
      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

  cv2.imshow('gray', out)

  print("done")

  cv2.waitKey(0)


