# -*- coding: utf-8 -*-
# # Hough-basierte Fluchtpunktberechnung
# ---
# <div style="float: right; text-align: right"> 
#     Author: Matthias Ramsauer <br />
#     Datum: 05.01.2020
# </div>
# In dieser Studienarbeit wird der Algorithmus aus:
#
# > Tuytelaars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998
#
# implementiert und getestet auf die Berechnung von Fluchtpunkten. Für die Implementeriung wird Python 3.7 und
# OpenCV 4.2 verwendet. Zur Darstellung und Bearbeitung des Quellcodes wurde jupyter notebook verwendet.

# +
# %load_ext autoreload
# %autoreload 2

import cv2
import hough
import subspace as sub
import os
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
from skimage.feature import peak_local_max
# -

# ## 1. Konfiguration
# Die Konfiguration aller Parameter die für den Algorithmus wichtig sind. Dabei kann die zu verwendete
# Resolution auch auf `None` gesetzt werden um eine Resolution wie im Abschnitt 3 beschrieben ist
# zu erhalten.
#
# Image Sources:
#
# - Google Maps
# - https://earth.jsc.nasa.gov/DatabaseImages/ESC/large/ISS013/ISS013-E-18319.JPG

image_path = '../images/munich_airport.jpg'
resolution = None
max_memory = 25 * 1024**3

# Für den Canny Algorithmus zur Reduzierung der verwendeten Daten aus dem Bild werden `canny_min` und 
# `canny_max` verwendet.

canny_min = 60 # 230
canny_max = 255

# # TODO

non_maximum_suppression_size = 10
layer1_threshold = 0.5
on_line_threshold = 0.5 # None
lines_intersecting_threshold = None
layer3_threshold = 1

# ## 2. Laden des Bildes
# In diesem Abschnitt wird das oben in Abschnitt 1 ausgewählte Bild geladen. Wird dabei ein Fehler gefunden wird
# der Prozess abgebrochen. Zur späteren Darstellung wird das Bild als Farbbild eingelesen und von den in
# OpenCV standardmäßig verwendeteten `BGR` Farbraum in den `RGB` Farbraum konvertiert.

# +
image_path = os.path.join(os.getcwd(), image_path)
if not os.path.exists(image_path):
  raise SystemError("File {} doesn't exist".format(image_path))

image = cv2.imread(image_path)
if image is None:
  raise SystemError("File {} couldn't be loaded".format(image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
# -

# ## 3. Vorbearbeitung des Bildes
# Im zu implementierenden Algorithmus wird davon gesprochen das Bild in den ersten Subspace zu setzen. Um dieser
# Anforderung nachzukommen wird zunächst falls nicht vorhanden eine `resolution` für die Subspaces ausgewählt.
# Im Artikel wird eine 600x600 Auflösung der einzelnen Subspaces für 1000x1000 pixel Bilder verwendet. Wenn
# keine `resolution` angegeben wird geht die Implementierung deshalb davon aus 60% der Auflösung zu verwenden.
#
# Desweiteren muss das Bild in den ersten Subspace passen. Dabei wird das Bild in diesem Fall zunächst
# quadratisch beschnitten und dann verkleinert auf die benötigte Auflösung. Von einer Beschneidung wird im
# Artikel nicht gesprochen jedoch von einer Verkleinerung.

# +
min_size = min(image.shape[:2])
image = image[0:min_size, 0:min_size]

if resolution is None:
  resolution = int(0.6 * min_size)
  print("use resolution =", resolution)
image = cv2.resize(image, (resolution, resolution))

plt.imshow(image)
plt.show()
# -

# Nach der Größenanpassung wird das Rauschen durch einen Gausischen Weichzeichner verringert.

# +
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# gray_image = np.zeros((500, 500), np.uint8)
# cv2.line(gray_image, (0, 0), (500, 50), 255, 2)
# resolution = 500

blured_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
plt.imshow(blured_image, 'gray')
plt.show()
# -

# Im Artikel wird davon gesprochen, dass nur Punkte verwendet werden die bereits so aussehen als ob sie auf
# einer Linie sind. Für diesen Schritt wird die OpenCV Implementierung von Canny verwendet.

canny = cv2.Canny(blured_image, canny_min, canny_max)
plt.imshow(canny, cmap='gray')
plt.show()

# ## 4. Hough Durchgänge
#
# Konvertierung des Bildes in die Subspaces Implementierung

layer0 = sub.subspaces_from_image(resolution, canny)


# Erster Hough durchlauf auf dem in subspaces konvertierten Bild. In Layer 1 werden somit mögliche Geraden
# gefunden die durch Kanten des Bildes verlaufen.

# %time layer1 = hough.hough_vec(layer0, max_memory = max_memory)

# Vor jedem weiteren Hough durchgang werden zunächst die Daten gefiltered. Dazu steht im Artikel:
# > ... As to the data read out at each layer, local maxima are selected. These discrete points are also the acutal data passed on to the next layer. In order to avoid closely spaced clusters of peaks, a non-maximum suppression is applied. The logarithm of peak height ist used as weightining factor for the votes it has at the next level. ... [1]
#
# In einem anderem Artikel von Tuytelaars steht dies zur Filterung:
# > ... Only truly 'non-accidental' structures are read out at the different layers. What that means is layer dependent: for the subsequent layers these are straight lines of a  minimum length, points where at least  three straight lines intersect, and lines that contain at least three line intersections. Note that in the latter case, these can be three intersections of each time two lines. ...  [2]
#

# +
# import scipy.ndimage as ndimage
# import scipy.ndimage.filters as filters

def non_maximum_suppression(image, distance):
  out = np.zeros(image.shape, dtype=image.dtype)
  resolution = sub.subspace_resolution(image)
  pad_image = np.pad(image, distance, 'constant', constant_values=0)
  for y in range(distance, distance + resolution, 1):
    for x in range(distance, distance + resolution, 1):
      neighbor_max = pad_image[y - distance:y + distance, x - distance:x + distance].max()
      if neighbor_max != 0 and pad_image.item((y, x)) == neighbor_max:
        out.itemset((y - distance, x - distance), neighbor_max)
  return out

def filter_subspace(spaces: sub.subspaces_t, nms: int, threshold: np.int32 = 3):
    out = []
    for space in spaces:
      local_max_mask = peak_local_max(
        space, 
        min_distance=1,
        indices=False,
        exclude_border=False,
        threshold_abs=threshold,
      )
      local_max_space = np.where(local_max_mask, space, 0)
      local_max_space = non_maximum_suppression(local_max_space, nms)
      local_max_space = np.log2(1 + local_max_space).astype(np.int32)
      out.append(local_max_space)
    return out
  
def filter_intersect(
  lines: sub.subspaces_t, 
  intersects: sub.subspaces_t, 
  on_line_threshold: float = 0.01,
  lines_intersecting_threshold: int = 3
):
  on_line_threshold = 0.01 if on_line_threshold is None else on_line_threshold
  lines_intersecting_threshold = 3 if lines_intersecting_threshold is None else lines_intersecting_threshold
  
  mask = sub.subspaces_create(sub.subspaces_resolution(lines), dtype = np.bool_)
  linear_equations = [np.array([-a, -1, -b]) for a, b in sub.subspaces_to_line(lines, 1)]
  homogene_points = [np.array([x, y, 1]) for x, y in sub.subspaces_to_line(intersects, 1)]

  for p in homogene_points:
    count = 0
    for eq in linear_equations:
      if abs(eq @ p) < on_line_threshold: 
        count += 1
      if count >= lines_intersecting_threshold:
        sub.subspaces_itemset(mask, p[:2], True)
        break
  return [np.where(mask_space, intersects[i], 0) for i, mask_space in enumerate(mask)]
# -

max_value = sub.subspaces_max(layer1)
filtered_layer1 = filter_subspace(layer1, non_maximum_suppression_size, threshold=max_value*layer1_threshold)
print([space[space > 0].size for space in filtered_layer1])
print([space.max() for space in filtered_layer1])

# %time layer2 = hough.hough_vec(filtered_layer1, max_memory = max_memory)

filtered_layer2 = filter_subspace(layer2, non_maximum_suppression_size)
filtered_layer2 = filter_intersect(filtered_layer1, filtered_layer2, 
                                   on_line_threshold = on_line_threshold, 
                                   lines_intersecting_threshold = lines_intersecting_threshold)

# %time layer3 = hough.hough_vec(filtered_layer2, max_memory = max_memory)

filtered_layer3 = filter_subspace(layer3, non_maximum_suppression_size, threshold = layer3_threshold)


# +
# [cv2.imwrite('/tmp/layer1_{}.png'.format(i), layer1[i]) for i in range(3)]
# [cv2.imwrite('/tmp/layer2_{}.png'.format(i), layer2[i]) for i in range(3)]
# [cv2.imwrite('/tmp/layer3_{}.png'.format(i), layer3[i]) for i in range(3)]
# -

# ## 5. Display Results

# +
def plot_lines(axis, ss: sub.subspaces_t, style='-'):
  resolution = sub.subspaces_resolution(ss)
  x = np.arange(*axis.get_xlim(), 2.0 / resolution)
  for a, b in sub.subspaces_to_line(ss, 1):
    axis.plot(x, -a * x - b, style)
    
def plot_intersect(axis, ss: sub.subspaces_t, style: str):
  for x, y in sub.subspaces_to_line(ss, 1):
    axis.plot(x, y, style)
    
def plot_layer(name: str, ss: sub.subspaces_t, fss: sub.subspaces_t = None, line_ss: sub.subspaces_t = None):
  # left, right, bottom, top
  extent = [-1, 1, 1, -1]
  subplot_opt = {"nrows": 1 if fss is None else 2, "ncols": 3, "sharey": True}
  first = (0) if fss is None else (0, 0)
  
  fig, axs = plt.subplots(**subplot_opt)
  fig.suptitle(name, fontsize = 32)
  fig.set_size_inches(30, 10.5 * (1 if fss is None else 2))
  
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

def progress(count, total, status='', bar_len=60):
  import sys
  filled_len = int(round(bar_len * count / float(total)))

  percents = round(100.0 * count / float(total), 1)
  bar = '=' * filled_len + '-' * (bar_len - filled_len)

  sys.stderr.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
  sys.stderr.flush()

def full_space(ss: sub.subspaces_t):
  resolution = sub.subspaces_resolution(ss)
  values = sub.subspace_axis(resolution)
  prod = np.array(np.meshgrid(values, values)).T.reshape(-1, 2)
  image = np.apply_along_axis(lambda xy: sub.subspaces_item(ss, xy), 1, prod).reshape((values.size, values.size))
  return image.T

def ss_thres(ss: sub.subspaces_t, threshold: float, threshold_max: float = 1.0):
  ss_max = sub.subspaces_max(ss)
  return [np.where((s > ss_max * threshold) & (s <= ss_max * threshold_max), s, 0) for s in ss]


# +
fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle("image", fontsize=32)
fig.set_size_inches(20, 20)
limit = 308
axs.set_ylim((limit, -limit))
axs.set_xlim((-limit, limit))
_ = axs.imshow(image, 'gray', extent=[-1, 1, 1, -1])
plot_lines(axs, filtered_layer1)
plot_intersect(axs, filtered_layer2, 'bo')
#plot_lines(axs, filtered_layer3, 'r--')

#lines = [
# (-0.7850162866449512, 0.2931596091205213),
# (0.6872964169381108, -0.2671009771986971),
#]
#x = np.arange(*axs.get_xlim(), 2.0 / resolution)
#for b, a in lines:
#  axs.plot(x, -a * x - b, 'bo')

# -

fig, axs = plot_layer("layer 0", layer0, line_ss=filtered_layer1)
plot_intersect(axs[0], filtered_layer2, style='ro')

fix, axs = plot_layer("layer 1", layer1, fss=filtered_layer1, line_ss=filtered_layer2)
print([space[space > 0].size for space in filtered_layer1])


fig, axs = plot_layer("layer 2", layer2, fss = filtered_layer2)
print([space[space > 0].size for space in filtered_layer2])

plot_layer("layer 3", layer3, fss = filtered_layer3)
print([space[space > 0].size for space in filtered_layer3])

# ## Quellen
# **[1]** *Tuytelaars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998*
#
# **[2]** *Tuytelaars, T., Proesmans, M., & Van Gool, L. (1997). The cascaded Hough transform as support for 
#      grouping and finding vanishing points and lines. Lecture Notes in Computer Science, 278–289. 
#      doi:10.1007/bfb0017873*
#
