import subspace as sub
import numpy as np
import cv2
import sys
import multiprocessing as multi
import draw
import time


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
# -

def hough_vec(src: sub.subspaces_t) -> sub.subspaces_t:
  resolution = sub.subspaces_resolution(src)
  ss = sub.subspaces_create(resolution)
  
  arr = np.arange(-1, 1, 2.0 / resolution)

  nonzero = arr[arr != 0]
  ones = np.ones(len(nonzero), dtype=np.float64)
  arr_reciprocal = np.true_divide(ones, nonzero)

  values = np.concatenate((arr, arr_reciprocal), axis=0)
  values = np.c_[ values, values, np.ones(len(values)) ]
  
  prod = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2).tolist()  
  
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
  
  print("apply on axis")
  # needs the most amount of time
  Ma_iterate = np.apply_along_axis(value_filter_a_iterate, 1, prod)
  Mb_iterate = np.apply_along_axis(value_filter_b_iterate, 1, prod)

  # Ms = np.concatenate((Ma_iterate, Mb_iterate)).reshape(-1, 3, 3)
  Ma_iterate = Ma_iterate[~np.isnan(Ma_iterate[:,0,0])].reshape(-1, 3, 3)
  Mb_iterate = Mb_iterate[~np.isnan(Mb_iterate[:,0,0])].reshape(-1, 3, 3)
  
  print("setup done")
  for i, M in enumerate(Ma_iterate):
    ba_array = np.einsum('ij,...j', M, values)[:, :2]
    ba_array = ba_array[~np.isnan(ba_array[:, 0])]
    if ba_array.size > 0:
      space = sub.subspaces_create(resolution)
      np.apply_along_axis(lambda ba: sub.subspaces_itemset(space, ba, 1), 1, ba_array)
      sub.subspaces_add_to(ss, space)

  for i, M in enumerate(Mb_iterate):
    ba_array = np.einsum('ij,...j', M, values)[:, :2]
    ba_array = ba_array[~np.isnan(ba_array[:, 0])]
    if ba_array.size > 0:
      space = sub.subspaces_create(resolution)
      np.apply_along_axis(lambda ba: sub.subspaces_itemset(space, ba, 1), 1, ba_array)
      sub.subspaces_add_to(ss, space)
  return ss


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
    
    print("chunksize = ", chunksize)
    start = time.time()
    space = next(iterator)
    while space is None:
      start = time.time()
      space = next(iterator)
    sub.subspaces_add_to(ss, space)
    end = time.time()
    duration = end - start
    
    print("estimated time in s: {}, single in s: {}".format(duration * len(prod), duration))
    
    for space in iterator:
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


