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
from matplotlib import pyplot as plt


# +
def _hough_process(args):
  x, y, src = args
  if sub.subspaces_item(src, (x, y)) == 0:
    return None
  
  resolution = sub.subspaces_resolution(src)
  values = sub.subspace_axis(resolution)
  
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

def hough(src: sub.subspaces_t) -> sub.subspaces_t:
  resolution = sub.subspaces_resolution(src)
  ss = sub.subspaces_create(resolution)
  values = sub.subspace_axis(resolution)
  
  # cartesian of x an y
  prod = np.array(np.meshgrid(values, values)).T.reshape(-1, 2).tolist()
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


# +
def _hough_vec_process(input_values):
  resolution = int(input_values[6, 0])
  Ma = input_values[0:3]
  Mb = input_values[3:6]
  
  values = sub.subspace_axis(resolution)
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
  values = sub.subspace_axis(resolution)
  
  prod = np.array(np.meshgrid(values, values)).T.reshape(-1, 2).tolist()  
  
  nan_m = np.array([
        [np.nan, 0, 0],
        [0, np.nan, 0],
        [0, 0, np.nan]
      ])
  
  def value_filter_a_iterate(ba):
    b, a = ba
    if b == 0 or sub.subspaces_item(src, ba) == 0:
      return nan_m
    return np.array([
      [1, 0, 0],
      [0, -1/b, -a/b],
      [0, 0, 1]
    ])
  
  def value_filter_b_iterate(ba):
    b, a = ba
    if sub.subspaces_item(src, ba) == 0:
      return nan_m
    return np.array([
      [-b, 0, -a],
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
    
  cpus = multi.cpu_count() - 2
  with multi.Pool(multi.cpu_count()) as p:
    chunksize = max(1, (len(M_iterate) // multi.cpu_count()) // 20)
    iterator = p.imap_unordered(_hough_vec_process, M_iterate, chunksize = chunksize)
    
    for i, space in enumerate(iterator):
      sub.subspaces_add_to(ss, space)

  return ss


# -

if __name__ == '__main__':
  image_path = '../images/hough.png'
  resolution = 200
  
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (resolution, resolution))

  src = sub.subspaces_from_image(resolution, image)
  
  ss = hough_vec(src)

  fig, axs = plt.subplots(1, 3, sharey=True)
  for i, space in enumerate(ss):
    axs[i].imshow(space, 'gray', extent=[-1, 1, 1, -1])


