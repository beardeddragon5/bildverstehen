# -*- coding: utf-8 -*-
# # Hough-basierte Fluchtpunktberechnung
# ---
# <div style="float: right; text-align: right"> 
#     Author: Matthias Ramsauer <br />
#     Datum: 05.01.2020
# </div>
#
# Abstract: 
# *In dieser Reimplementierung von Tuytellars [1] Algorithmus zur Fluchtpunktberechung werden für
# gute Bilder akzeptable Fluchtpunkte gefunden. Jedoch sind einige Fehler aufgefallen, diese könnten jedoch
# auch durch die Implementierung entstanden sein.*
#
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
import draw
import os
import numpy as np
from matplotlib import pyplot as plt
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
# - https://static.turbosquid.com/Preview/2015/03/20__21_13_06/rubikscube_A_render_full_A.jpgee5b4236-df84-4080-abc4-ee4e7a78b43fZoom.jpg
# - Bilder aus dem Kurs
# - Testbilder mit einzelnen Punkten, Linien und Geometrischen Figuren

image_path = '../images/rubix.jpg'
resolution = None
max_memory = 25 * 1024**3

# In [1,2] wird beschrieben, dass nur Punkte verwendet werden, die bereits auf einer
# Kante im Bild liegen. In [2] wird nahgelegt, dass dafür Tangenten berechnet werden.  
# Für diese Implementierung wird Canny verwendet. Nach der Filterung des Subspaces erhalten wir ähnliche 
# Ergebnisse wie im Papier [1] beschrieben.  
# Um die Kanten zu erkennen können die beiden Canny Thresholds mithilfe von `canny_min` und 
# `canny_max` eingestellt werden.

canny_min = 230
canny_max = 255

# Für die Filterung der einzelnen layer werden Bündel von lokalen Maxima durch eine non-maxima-supression verkleinert. Dafür kann man mit `non_maximum_suppression_size` die Entsprechende Nachbarregion für diesen Algorithmus einstellen.

non_maximum_suppression_size = 10

# Für das erste Layer muss ein threshold verwendet werden um nur möglichst gute Linien zu verwenden in [2] wird von einer minimalen Länge der Linien gesprochen dies lies sich für den beschränkten Zeitraum der Studienarbeit jedoch nicht so umsetzen. Das Anwenden eines Thresholds führt zumindest zu einem ähnlichen Ergebnis.

layer1_threshold = 0.3

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

resolution = int(0.6 * min_size) if resolution is None else resolution
print("use resolution =", resolution)
image = cv2.resize(image, (resolution, resolution))

plt.imshow(image)
plt.show()
# -

# Nach der Größenanpassung wird das Rauschen durch einen Gausischen Weichzeichner verringert.

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blured_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
plt.imshow(blured_image, 'gray')
plt.show()

# Im Artikel [1] wird davon gesprochen, dass nur Punkte verwendet werden die bereits so aussehen als ob sie auf
# einer Linie sind. Für diesen Schritt wird die OpenCV Implementierung von Canny verwendet.

canny = cv2.Canny(blured_image, canny_min, canny_max)
plt.imshow(canny, cmap='gray')
plt.show()

# ## 4. Hough Durchgänge
#
# Konvertierung des Bildes in die Subspaces Implementierung

layer0 = sub.subspaces_from_image(resolution, canny)


# Erster Hough Durchlauf auf dem in subspaces konvertierten Bild. In Layer 1 werden somit mögliche Geraden
# gefunden die durch Kanten des Bildes verlaufen.

# %time layer1 = hough.hough_vec(layer0, max_memory = max_memory)

# Vor jedem weiteren Hough durchgang werden zunächst die Daten gefiltered. Dazu steht im Artikel:
# > ... As to the data read out at each layer, local maxima are selected. These discrete points are also the acutal data passed on to the next layer. In order to avoid closely spaced clusters of peaks, a non-maximum suppression is applied. The logarithm of peak height ist used as weightining factor for the votes it has at the next level. ... [1]
#
# In einem anderem Artikel von Tuytelaars steht dies zur Filterung:
# > ... Only truly 'non-accidental' structures are read out at the different layers. What that means is layer dependent: for the subsequent layers these are straight lines of a  minimum length, points where at least  three straight lines intersect, and lines that contain at least three line intersections. Note that in the latter case, these can be three intersections of each time two lines. ...  [2]
#
# Für diese Implementierung wird Hauptsächlich die Filterung aus [1] verwendet, jedoch um gute Ergebnisse zu
# erhalten zum Teil auch die von [2]. Im allgemeinen scheinen die Filter aus [2] sinnvoller zu sein als der
# generische Ansatz aus [1] der unsaubere Ergebnisse erzielt.
#

def filter_subspace(spaces: sub.subspaces_t, nms: int, threshold: np.int32 = 3):
    out = []
    for space in spaces:
      # find local maxima and add nms and save in mask
      local_max_mask = peak_local_max(
        space, 
        min_distance=nms,
        indices=False,
        exclude_border=False,
        threshold_abs=threshold,
      )
      # apply mask to space
      local_max_space = np.where(local_max_mask, space, 0)
      # apply log to space
      local_max_space = np.log2(1 + local_max_space).astype(np.int32)
      out.append(local_max_space)
    return out


max_value = sub.subspaces_max(layer1)
filtered_layer1 = filter_subspace(layer1, non_maximum_suppression_size, threshold=max_value*layer1_threshold)


# %time layer2 = hough.hough_vec(filtered_layer1, max_memory = max_memory)

# Für Layer 2 ist es wichtig nur Schnittpunkte zu verwenden durch die laut [2] mindestens 3 Linien führen. Ohne diesen zusätzlichen Schritt werden zu viele Ausreißer gefunden die zwar das threshold von 3 Intersektionen
# erreichen, aber nicht am selben Punkt.
# [2] legt dabei nahe, das für jeden Subspace Wert die dafür verantwortlichen Werte gespeichert werden. In dieser Implementierung wird dies rekonstruiert was durch den in der Regel kleinen Suchraum kein Problem ist.  
# Die Diskretisierung erzeugt jedoch die Problematik, dass es sich eher zum Schnittregionen als Schnittpunkte handelt. Dies muss bei der Filterung berücksichtigt werden, vor allem in subspace zwei und drei.

def filter_intersect(
  lines: sub.subspaces_t, 
  intersects: sub.subspaces_t, 
  lines_intersecting_threshold: int = 3
):  
  resolution = sub.subspaces_resolution(lines)
  mask = sub.subspaces_create(resolution, dtype = np.bool_)
  
  # map to homogeneous linear equations
  linear_equations = [np.array([-a, -1, -b]) for a, b in sub.subspaces_peaks(lines, 1)]
  # map to homogeneous points
  homogene_points = [np.array([x, y, 1]) for x, y in sub.subspaces_peaks(intersects, 1)]
  for p in homogene_points:
    # find offset for point p and use only the maximum
    offset = max(sub.subspace_offset(resolution, p[0], p[1]))
    count = 0
    for eq in linear_equations:
      if abs(eq @ p) < offset: 
        count += 1
      if count >= lines_intersecting_threshold:
        # found minium intersections needed
        sub.subspaces_itemset(mask, p[:2], True)
        break
  # apply mask to intersection space
  return [np.where(mask_space, intersects[i], 0) for i, mask_space in enumerate(mask)]

filtered_layer2 = filter_subspace(layer2, non_maximum_suppression_size)
filtered_layer2 = filter_intersect(filtered_layer1, filtered_layer2)

# %time layer3 = hough.hough_vec(filtered_layer2, max_memory = max_memory)

filtered_layer3 = filter_subspace(layer3, non_maximum_suppression_size)

# ## 5. Display Results

limit = 5
fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle("Bild mit Fluchtpunkten und Linien", fontsize=32)
fig.set_size_inches(15, 15)
axs.set_ylim((limit, -limit))
axs.set_xlim((-limit, limit))
axs.imshow(image, 'gray', extent=[-1, 1, 1, -1])
draw.plot_lines(axs, filtered_layer1)
draw.plot_intersect(axs, filtered_layer2, 'bo')
draw.plot_lines(axs, filtered_layer3, 'r--')

draw.plot_layer("layer 0: Mit Linien aus layer 1", layer0, line_ss=filtered_layer1)
draw.plot_layer("layer 1: Mit Linien aus layer 2", layer1, line_ss=filtered_layer2)
draw.plot_layer("layer 2", layer2)
draw.plot_layer("layer 3", layer3)
print("Points in Filtered Layer 1:", [space[space > 0].size for space in filtered_layer1])
print("Points in Filtered Layer 2:", [space[space > 0].size for space in filtered_layer2])
print("Points in Filtered Layer 3:", [space[space > 0].size for space in filtered_layer3])

# ## Quellen
# **[1]** *Tuytelaars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998*
#
# **[2]** *Tuytelaars, T., Proesmans, M., & Van Gool, L. (1997). The cascaded Hough transform as support for 
#      grouping and finding vanishing points and lines. Lecture Notes in Computer Science, 278–289. 
#      doi:10.1007/bfb0017873*
#
