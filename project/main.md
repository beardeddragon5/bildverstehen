# -*- coding: utf-8 -*-
# Hough-basierte Fluchtpunktberechnung
---
<div style="float: right; text-align: right"> 
    Author: Matthias Ramsauer <br />
    Datum: 05.01.2020
</div>
In dieser Studienarbeit wird der Algorithmus aus:

> Tuytelaars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998

implementiert und getestet auf die Berechnung von Fluchtpunkten. Für die Implementeriung wird Python 3.7 und
OpenCV 4.2 verwendet. Zur Darstellung und Bearbeitung des Quellcodes wurde jupyter notebook verwendet.

```python
%load_ext autoreload
%autoreload 2

import cv2
import hough
import subspace as sub
import os
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
from skimage.feature import peak_local_max
```

## 1. Konfiguration
Die Konfiguration aller Parameter die für den Algorithmus wichtig sind. Dabei kann die zu verwendete
Resolution auch auf `None` gesetzt werden um eine Resolution wie im Abschnitt 3 beschrieben ist
zu erhalten.

Image Sources:

- Google Maps
- https://earth.jsc.nasa.gov/DatabaseImages/ESC/large/ISS013/ISS013-E-18319.JPG

```python
image_path = '../images/figure3.png'
resolution = None
```

Für den Canny Algorithmus zur Reduzierung der verwendeten Daten aus dem Bild werden `canny_min` und 
`canny_max` verwendet.

```python
canny_min = 230
canny_max = 255
```

Für die Filterung der Subspaces zwischen den Hough Transformationen werden die beiden letzten
Parameter genutzt.

```python
non_maximum_suppression_size = 10
relative_filter_threshold = 0.80
```

## 2. Laden des Bildes
In diesem Abschnitt wird das oben in Abschnitt 1 ausgewählte Bild geladen. Wird dabei ein Fehler gefunden wird
der Prozess abgebrochen. Zur späteren Darstellung wird das Bild als Farbbild eingelesen und von den in
OpenCV standardmäßig verwendeteten `BGR` Farbraum in den `RGB` Farbraum konvertiert.

```python
image_path = os.path.join(os.getcwd(), image_path)
if not os.path.exists(image_path):
  raise SystemError("File {} doesn't exist".format(image_path))

image = cv2.imread(image_path)
if image is None:
  raise SystemError("File {} couldn't be loaded".format(image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
```

## 3. Vorbearbeitung des Bildes
Im zu implementierenden Algorithmus wird davon gesprochen das Bild in den ersten Subspace zu setzen. Um dieser
Anforderung nachzukommen wird zunächst falls nicht vorhanden eine `resolution` für die Subspaces ausgewählt.
Im Artikel wird eine 600x600 Auflösung der einzelnen Subspaces für 1000x1000 pixel Bilder verwendet. Wenn
keine `resolution` angegeben wird geht die Implementierung deshalb davon aus 60% der Auflösung zu verwenden.

Desweiteren muss das Bild in den ersten Subspace passen. Dabei wird das Bild in diesem Fall zunächst
quadratisch beschnitten und dann verkleinert auf die benötigte Auflösung. Von einer Beschneidung wird im
Artikel nicht gesprochen jedoch von einer Verkleinerung.

```python
min_size = min(image.shape[:2])
image = image[0:min_size, 0:min_size]

if resolution is None:
  resolution = int(0.6 * min_size)
  print("use resolution =", resolution)
image = cv2.resize(image, (resolution, resolution))

plt.imshow(image)
plt.show()
```

Nach der Größenanpassung wird das Rauschen durch einen Gausischen Weichzeichner verringert.

```python
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# gray_image = np.zeros((500, 500), np.uint8)
# cv2.line(gray_image, (0, 0), (500, 50), 255, 2)
# resolution = 500

blured_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
plt.imshow(blured_image, 'gray')
plt.show()
```

Im Artikel wird davon gesprochen, dass nur Punkte verwendet werden die bereits so aussehen als ob sie auf
einer Linie sind. Für diesen Schritt wird die OpenCV Implementierung von Canny verwendet.

```python
canny = cv2.Canny(blured_image, canny_min, canny_max)
plt.imshow(canny, cmap='gray')
plt.show()
```

## 4. Hough Durchgänge

Konvertierung des Bildes in die Subspaces Implementierung

```python
layer0 = sub.subspaces_from_image(resolution, canny)
```

Erster Hough durchlauf auf dem in subspaces konvertierten Bild. In Layer 1 werden somit mögliche Geraden
gefunden die durch Kanten des Bildes verlaufen.

```python
%time layer1 = hough.hough_vec(layer0, max_memory = 25 * 1024**3)
```

Vor jedem weiteren Hough durchgang werden zunächst die Daten gefiltered. Dazu steht im Artikel:
> ... As to the data read out at each layer, local maxima are selected. These discrete points are also the acutal data passed on to the next layer. In order to avoid closely spaced clusters of peaks, a non-maximum suppression is applied. The logarithm of peak height ist used as weightining factor for the votes it has at the next level. ...  
(Tuytellars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998)

Diese Filterung wird in dieser Implementierung folgendermaßen gelöst:

1. Ermittlung des globalen Maximum Wertes für möglichen Grenzwert
2. Umwandlung jedes Subspaces in einen [0-1] Wertebereich wobei die bisherige Verteilung beibehalten wird
3. Wie im Artikel beschrieben wird hier der $\log_2$ auf alle Elemente angewandt. Dabei werden bisherige
   Maximas abgeschwächt
4. Rücktransformierung in den Wertebereich [0-255]
5. Finden von lokalen Maxima mit möglichen Grenzwert
6. Non-Maximum Supression um Punktegruppen zu verkleinern

```python
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


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

# def local_maxima(data, min_distance, threshold_abs):
#   data_max = filters.maximum_filter(data, min_distance, mode='constant')
#   maxima = (data == data_max)
#   data_min = filters.minimum_filter(data, min_distance, mode='constant')
#   diff = ((data_max - data_min) > threshold_abs)
#   maxima[diff == 0] = 0
# 
#   labeled, num_objects = ndimage.label(maxima)
#   # xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
#   return np.where(labeled != 0, True, False)

def filter_subspace(spaces: sub.subspaces_t, nms: int, threshold: float = 0.0):
    out = []
    max_value = sub.subspaces_max(spaces)
    for space in spaces:
      local_max_mask = peak_local_max(
        space, 
        min_distance=1,
        indices=False,
        exclude_border=False,
        threshold_abs=max_value * threshold,
      )
      local_max_space = np.where(local_max_mask, space, 0)
      local_max_space = non_maximum_suppression(local_max_space, nms)
      local_max_space = np.log2(1 + local_max_space).astype(np.int32)
      out.append(local_max_space)
    return out
```

```python
filtered_layer1 = filter_subspace(layer1, non_maximum_suppression_size)
```

```python
%time layer2 = hough.hough_vec(filtered_layer1, max_memory = 22 * 1024**3)
```

```python
filtered_layer2 = filter_subspace(layer2, non_maximum_suppression_size)
```

```python
%time layer3 = hough.hough_vec(filtered_layer2, max_memory = 25 * 1024**3)
```

```python
filtered_layer3 = filter_subspace(layer3, non_maximum_suppression_size)
```

```python
[cv2.imwrite('/tmp/layer1_{}.png'.format(i), layer1[i]) for i in range(3)]
[cv2.imwrite('/tmp/layer2_{}.png'.format(i), layer2[i]) for i in range(3)]
[cv2.imwrite('/tmp/layer3_{}.png'.format(i), layer3[i]) for i in range(3)]
```

# Display Results

```python
def plot_lines(axis, ss: sub.subspaces_t, style='-'):
  resolution = sub.subspaces_resolution(ss)
  x = np.arange(*axis.get_xlim(), 2.0 / resolution)
  for b, a in sub.subspaces_to_line(ss, 1):
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

def ss_thres(ss: sub.subspaces_t, threshold: float):
  ss_max = sub.subspaces_max(ss)
  return [np.where((s > ss_max * threshold), s, 0) for s in ss]
```

```python
fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle("image", fontsize=32)
fig.set_size_inches(20, 20)
axs.set_ylim((5, -5))
axs.set_xlim((-5, 5))
_ = axs.imshow(image, 'gray', extent=[-1, 1, 1, -1])
plot_lines(axs, ss_thres(filtered_layer1, 0.75))
#plot_lines(axs, ss_thres(filtered_layer2, 0.5), 'b-')
#plot_intersect(axs, ss_thres(filtered_layer3, 0.9), 'ro')
```

```python
_ = plot_layer("layer 0", layer0) # , line_ss=ss_thres(filtered_layer1, 0.75)
```

```python
plot_layer("layer 1", layer1, fss=filtered_layer1) # , line_ss=ss_thres(filtered_layer2, 0.8)
print([space[space > 0].size for space in filtered_layer1])
#print(filtered_layer1[0].max())
#plt.hist(cv2.calcHist(filtered_layer1[0], 0, None, filtered_layer1[0].max(), [0, filtered_layer1[0].max()]))
```


```python
# flayer1 = full_space(filtered_layer1)
```

```python
'''
values = sub.subspace_axis(resolution)
fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle("full layer1", fontsize=32)
fig.set_size_inches(20, 20)
axs.set_ylim((values.max(), values.min()))
axs.set_xlim((values.min(), values.max()))
axs.imshow(flayer1, 'gray', extent = [values.min(), values.max(), values.max(), values.min()])
plot_lines(axs, ss_thres(filtered_layer2, 0.90))
'''
;
```

```python
fig, axs = plot_layer("layer 2", layer2, fss = filtered_layer2)
print([space[space > 0].size for space in filtered_layer2])
```

```python
plot_layer("layer 3", layer3, fss = filtered_layer3)
print([space[space > 0].size for space in filtered_layer3])
```

```python

```

```python

```

