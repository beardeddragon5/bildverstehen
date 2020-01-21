# +
import cv2
import numpy as np
import subspace as sub
import sys
from matplotlib import pyplot as plt

def plot_lines(axis, ss: sub.subspaces_t, style='-'):
  resolution = sub.subspaces_resolution(ss)
  x = np.arange(*axis.get_xlim(), 2.0 / resolution)
  for a, b in sub.subspaces_peaks(ss, 1):
    axis.plot(x, -a * x - b, style)
    
def plot_intersect(axis, ss: sub.subspaces_t, style: str):
  for x, y in sub.subspaces_peaks(ss, 1):
    axis.plot(x, y, style)
    
def plot_layer(name: str, ss: sub.subspaces_t, fss: sub.subspaces_t = None, line_ss: sub.subspaces_t = None):
  # left, right, bottom, top
  extent = [-1, 1, 1, -1]
  subplot_opt = {"nrows": 1 if fss is None else 2, "ncols": 3, "sharey": True}
  first = (0) if fss is None else (0, 0)
  
  fig, axs = plt.subplots(**subplot_opt)
  fig.suptitle(name, fontsize = 16)
  fig.set_size_inches(10, 3 * (1 if fss is None else 2))
  
  if line_ss is not None:
    axs[first].set_ylim((1, -1))
    axs[first].set_xlim((-1, 1))
    plot_lines(axs[first], line_ss)
  
  if fss is None:
    for i, space in enumerate(ss):
      axs[i].imshow(space, 'gray', extent = extent) # , vmin = 0, vmax = space.max())
  else:
    for i, space in enumerate(ss):
      axs[0, i].imshow(space, 'gray', extent = extent) # , vmin = 0, vmax = space.max())
    for i, space in enumerate(fss):
      axs[1, i].imshow(space, 'gray', extent = extent) # , vmin = 0, vmax = space.max())
  return (fig, axs)

def full_space(ss: sub.subspaces_t):
  resolution = sub.subspaces_resolution(ss)
  values, _ = sub.subspace_axis(resolution)
  prod = sub.cartesian_product(values, values)
  image = np.apply_along_axis(lambda xy: sub.subspaces_item(ss, xy), 1, prod).reshape((values.size, values.size))
  return image.T

def ss_thres(ss: sub.subspaces_t, threshold: float, threshold_max: float = 1.0):
  ss_max = sub.subspaces_max(ss)
  return [np.where((s > ss_max * threshold) & (s <= ss_max * threshold_max), s, 0) for s in ss]

def progress(count, total, status='', bar_len=60):
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stderr.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stderr.flush()
# -


