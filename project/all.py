# -*- coding: utf-8 -*-
# # Hough-basierte Fluchtpunktberechnung
# ---
# <div style="float: right; text-align: right"> 
#     Author: Matthias Ramsauer <br />
#     Datum: 20.01.2020
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
import draw
import os
import numpy as np
import multiprocessing as multi
from typing import NewType, List
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
# -

# ## 1. Konfiguration
# Die Konfiguration aller Parameter, die für den Algorithmus wichtig sind. Dabei kann die zu verwendende
# Resolution auch auf `None` gesetzt werden, um eine Resolution - wie im Abschnitt 3 beschrieben - zu erhalten.
#
# Image Sources:
#
# - Google Maps
# - <div style="display: inline">https://earth.jsc.nasa.gov/DatabaseImages/ESC/large/ISS013/ISS013-E-18319.JPG</div>
# - <div style="display: inline">https://static.turbosquid.com/Preview/2015/03/20__21_13_06/rubikscube_A_render_
#     full_A.jpgee5b4236-df84-4080-abc4-ee4e7a78b43fZoom.jpg</div>
# - Bilder aus dem Kurs
# - Testbilder mit einzelnen Punkten, Linien und Geometrischen Figuren

image_path = '../images/rubix.jpg'
resolution = None
max_memory = 25 * 1024**3

# In [1,2] wird beschrieben, dass nur Punkte verwendet werden, die bereits auf einer
# Kante im Bild liegen. In [2] wird nahegelegt, dass dafür Tangenten berechnet werden.  
# Für diese Implementierung wird Canny verwendet. Nach der Filterung des Subspaces erhalten wir ähnliche 
# Ergebnisse wie im Paper [1] beschrieben.  
# Um die Kanten zu erkennen können die beiden Canny Thresholds mithilfe von `canny_min` und 
# `canny_max` eingestellt werden.

canny_min = 230
canny_max = 255

# Für die Filterung der einzelnen Layer werden Bündel von lokalen Maxima durch eine non-maxima-supression verkleinert. Dafür kann man mit `non_maximum_suppression_size` die entsprechende Nachbarregion für diesen Algorithmus einstellen.

non_maximum_suppression_size = 10

# Für den ersten Layer muss ein threshold verwendet werden, um nur möglichst gute Linien zu verwenden. In [2] wird von einer minimalen Länge der Linien gesprochen. Dies lies sich für den beschränkten Zeitraum der Studienarbeit jedoch nicht so umsetzen. Das Anwenden eines Thresholds führt zumindest zu einem ähnlichen Ergebnis.

layer1_threshold = 0.3

# ## 2. Laden des Bildes
# In diesem Abschnitt wird das oben in Abschnitt 1 ausgewählte Bild geladen. Wird dabei ein Fehler gefunden wird
# der Prozess abgebrochen. Zur späteren Darstellung wird das Bild als Farbbild eingelesen und von dem in
# OpenCV standardmäßig verwendeten `BGR` Farbraum in den `RGB` Farbraum konvertiert.

# +
image_path = os.path.join(os.getcwd(), image_path)
if not os.path.exists(image_path):
  raise SystemError("File {} doesn't exist".format(image_path))

image = cv2.imread(image_path)
if image is None:
  raise SystemError("File {} couldn't be loaded".format(
    image_path))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
# -

# ## 3. Vorbearbeitung des Bildes
# Im zu implementierenden Algorithmus wird davon gesprochen, das Bild in den ersten Subspace zu setzen. Um dieser
# Anforderung nachzukommen, wird zunächst, falls nicht vorhanden, eine `resolution` für die Subspaces ausgewählt.
# Im Artikel wird eine 600x600 Auflösung der einzelnen Subspaces für 1000x1000 pixel Bilder verwendet. Wenn
# keine `resolution` angegeben wird, geht die Implementierung deshalb davon aus, 60% der Auflösung zu verwenden.
#
# Desweiteren muss das Bild in den ersten Subspace passen. Dabei wird das Bild in diesem Fall zunächst
# quadratisch beschnitten und dann auf die benötigte Auflösung verkleinert. Von einer Beschneidung wird im
# Artikel nicht gesprochen, jedoch von einer Verkleinerung.

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

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blured_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
plt.imshow(blured_image, 'gray')
plt.show()

# Im Artikel [1] wird davon gesprochen, dass nur Punkte verwendet werden, die bereits so aussehen, als ob sie auf
# einer Linie sind. Für diesen Schritt wird die OpenCV Implementierung von Canny verwendet.

canny = cv2.Canny(blured_image, canny_min, canny_max)
plt.imshow(canny, cmap='gray')
plt.show()


# Um ein kartesisches Produkt zweier Mengen zu erhalten, wird hier diese sehr einfache Funktion erstellt.

def cartesian_product(a: np.array, b: np.array):
  return np.array(np.meshgrid(a, b)).T.reshape(-1, 2)


# ### 3.1 Subspaces
# In [1,2] wurden Subspaces benutzt, um den unendlichen Raum zu diskretisieren. Der Raum wird in 3 Subspaces aufgeteilt, deren Wertebereich jeweils von [-1, 1] definiert ist. Statt der für die Hough Transformation übliche
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

# Ein Subspace ist dabei ein quadratisches, 2-dimensionales Bild aus Integer-Werten. Für die Akkumulation während der Hough Transformation wird ein großer Datentyp benötigt. Um eine reibungslose Interaktion zwischen opencv, python und numpy zu ermöglichen, wird hier ein 32-bit signed Integer verwendet.

# +
def subspace_create(resolution: int, dtype = np.int32) \
                                                -> subspace_t:
  subspace_shape = (resolution, resolution)
  return np.zeros(subspace_shape, dtype = dtype)

def subspaces_create(resolution: int, dtype = np.int32) \
                                                -> subspaces_t:
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

def subspace_inverse_pos(
  resolution: int, subspace: int, xy: tuple
) -> tuple:
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


# Um die Subspaces zu durchlaufen, wird für jede Subspace Zelle ein entsprechender Wert und die dazugehörige Größe berechnet. Dabei wird zunächst die innere Achse (Subspace 0) berechnet.
#
# $$
#     \text{inner} = \left[ \frac {2i} r \right]_{\frac {\text{resolution}} 2 \leq i \lt \frac {\text{resolution}} 2} \\
#     \text{offset_inner} = \left[ \frac 1 {(\text{resolution} - 1) / 2} \right]_{i \lt \text{resolution}}
# $$
#
# Dabei muss `inner` noch zusätzlich um `offset_inner` verschoben werden, da in `inner` nur die entsprechenden Grenzen zwischen Zellübergängen enthalten sind.
#
# Für Subspace 1 und 2 muss die Achse anders berechnet werden. Zunächst wird die halbe Achse berechnet:
#
# $$
#     outer = \left[ \frac {resolution - 1} {2i} \right]_{1\leq i \lt \frac {resolution} 2} \\
#     \text{offset_outer} = \left[ \frac {resolution - 1} {4i^2 + 4i} \right]_{1\leq i \lt \frac {resolution} 2}
# $$
#
# Bei `outer` handelt es sich wieder um die Grenzen zwischen den Zellen, die mit `offset_outer` entsprechend zentriert werden müssen. Letztendlich muss die Achse nur noch zusammengesetzt werden.

def subspace_axis(resolution: int) -> np.array:
  r = resolution - 1
  borders = resolution // 2
  
  inner = np.array([ i * (2 / r) \
                     for i in range(-borders, borders)])
  offset_inner = np.full(resolution, 1.0 / r)
  inner = inner + offset_inner
  
  outer = np.array([ r / (i * 2) for i in range(1, borders)])
  offset_outer = np.array([r / (4 * (i**2) + 4 * i) \
                           for i in range(1, borders)])  
  outer = np.concatenate((np.array([r / 2 + r / 8 ]), 
                          outer - offset_outer))
  
  return (
    np.concatenate((-outer, inner, np.flip(outer)), axis=0),
    np.concatenate(([r / 8], offset_outer, offset_inner, 
                    np.flip(offset_outer), [r / 8]), axis=0)
  )


# Berechnen der Fehlertoleranz, die für einen bestimmten Punkt in den Subspaces gilt.

def subspace_offset(resolution: int, a: float, b: float) \
                                                      -> tuple:
  values, offsets = subspace_axis(resolution)
  
  index_a = np.where((a >= (values - offsets)) & 
                     (a <= (values + offsets)))
  index_b = np.where((b >= (values - offsets)) & 
                     (b <= (values + offsets)))
  
  if index_a[0].size == 0 and \
     abs(a) > (values + offsets).max():
    offset_a = offsets[0]
  else:
    offset_a = np.take(offsets, index_a).max()

  if index_b[0].size == 0 and \
     abs(b) > (values + offsets).max():
    offset_b = offsets[0]
  else:
    offset_b = np.take(offsets, index_b).max()
  
  return (offset_a, offset_b)


# #### Subspace Bild
# Ein passendes Bild kann in den ersten Subspace eingefügt werden, um dann weiter verarbeitet zu werden.

def subspaces_from_image(resolution: int, image: np.array):
  w, h = image.shape
  if w != resolution or h != resolution:
    raise ValueError("image must be of size {0}x{0}"
                     .format(resolution))
  src = subspaces_create(resolution)
  np.add(src[0][:resolution,:resolution], image, 
         out=src[0][:resolution,:resolution])
  return src


# Um einfach Werte aus den Subspaces zu holen und zu setzen, werden entsprechend zwei Hilfsfunktionen erstellt, die mit $(a, b)$ Tupeln arbeiten können.

# +
def subspaces_item(subspaces: subspaces_t, ab: tuple) \
                                                  -> np.uint8:
  resolution = subspaces_resolution(subspaces)
  x, y, sub_id = subspace_pos(resolution, ab)
  return subspaces[sub_id].item((y, x))

def subspaces_itemset(subspaces: subspaces_t, 
                      ab: tuple, value: np.uint8):
  resolution = subspaces_resolution(subspaces)
  x, y, sub_id = subspace_pos(resolution, ab)
  subspaces[sub_id].itemset((y, x), value)


# -

# Um weiter mit Subspaces einfach arbeiten zu können, sind weitere Hilfsfunktionen erstellt worden, die entsprechend zum Addieren bzw. zur Analyse von Subspaces verwendet werden können.

# +
def subspaces_add_to(subspaces: subspaces_t, 
                     value: subspaces_t):
  for i, space in enumerate(subspaces):
    np.add(space, value[i], out=space)
    
def subspaces_max(subspaces: subspaces_t) -> np.int32:
  return np.max([space.max() for space in subspaces])

def subspaces_peaks(subspaces: subspaces_t, 
                    threshold: np.int32) -> List[tuple]:
  resolution = subspaces_resolution(subspaces)
  out = []
  for i, space in enumerate(subspaces):
    for y, x in np.array(np.where(space >= threshold)) \
                                            .T.reshape(-1, 2):
      out.append(subspace_inverse_pos(resolution, i, (x, y)))
  return out


# -

# ## 4. Hough Durchgänge
#
# Konvertierung des Bildes in die Subspaces Implementierung.

layer0 = subspaces_from_image(resolution, canny)


# ### 4.1 Multiprozessing Hough Algorithmus
#
# Die Hough Transformation wurde aufgrund des einfachen Algorithmus, der auch auf [Wikipedia]( 
# https://de.wikipedia.org/wiki/Hough-Transformation) zu finden ist, implementiert.
# Hier ist der entsprechende Pseudocode:
#
# $
# \texttt{max}_d \texttt{ := } \sqrt{bildhöhe^2 + bildbreite^2} \\
# \texttt{min}_d \texttt{ := } \texttt{max}_d \cdot -1 \\
# \texttt{houghRaum}[0 \ldots \pi][\texttt{min}_d \ldots \texttt{max}_d] := 0 \\
# \texttt{foreach pixel != 0 do} \\
# \hspace{0.5cm}\texttt{for } \alpha := 0 \texttt{ to } \pi \texttt{ do } \\
# \hspace{1.0cm}  \texttt{d := pixel}_x \cdot \cos(\alpha ) + \texttt{pixel}_y \cdot \sin(\alpha ) \\
# \hspace{1.0cm}  \texttt{houghRaum}[\alpha][\texttt{d}]++ \\
# \hspace{0.5cm}\texttt{end}\\
# \texttt{end}\\
# $
#
# Der Code musste auch auf die Subspaces angepasst werden. Der größte Unterschied ist das Durchlaufen des Raumes.
# Zunächst wird dafür eine virtuelle Achse gebildet, die jede diskrete Zelle in allen 3 Subspaces einmal durchläuft. Außerdem müssen im Gegesatz zum normalen Hough beide Achsen des Parameterraums durchlaufen werden.
# Sonst werden duch die Diskretisierung viele Geraden nicht gefunden. 
#
# Der folgende Codeblock zeigt den Durchlauf eines (x, y) Paares auf den Subspaces. Dies ist äquivalent zu diesem Teil des Pseudocodes.
#
# $
# \texttt{for } \alpha := 0 \texttt{ to } \pi \texttt{ do } \\
# \hspace{0.5cm}  \texttt{d := pixel}_x \cdot \cos(\alpha ) + \texttt{pixel}_y \cdot \sin(\alpha ) \\
# \hspace{0.5cm}  \texttt{houghRaum}[\alpha][\texttt{d}]++ \\
# \texttt{end}\\
# $
#
# Nachdem die Subspace Achse berechnet wurde, werden sie umgeformt, dass man jeweils Wertepaare `(x, x, 1)` erhält. Im nächsten Schritt werden über entsprechende Matrizen entweder `a` oder `b` für den jeweiligen Wert ausgerechnet und in einem Subspace eingefügt. Der Wert wird dabei auf `1` gesetzt. Dadurch wird ein Wert für den selben Wert von `x` o.`y` nicht doppelt gesetzt.

def _hough_vec_process(input_values):
  resolution = int(input_values[6, 0])
  Ma = input_values[0:3]
  Mb = input_values[3:6]
  
  values, _ = subspace_axis(resolution)
  values = np.c_[ values, values, np.ones(len(values)) ]
  
  space = subspaces_create(resolution, dtype = np.uint8)
  for M in [Ma, Mb]:
    ab_array = np.einsum('ij,...j', M, values)[:, :2]
    ab_array = ab_array[~np.isnan(ab_array[:, 0])]
    if ab_array.size > 0:
      # long running
      np.apply_along_axis(lambda ab: 
                          subspaces_itemset(space, ab, 1), 
                          1, ab_array)
  
  return space


# Die Bestimmung der beiden Matrizen erfolgt durch die Umformung von:
#
# $$
#     ax + b + y = 0
# $$
#
# mit der entsprechenden Umwandlung in eine Matrix Die Umformungen lauten dabei:
#
# $$
#     a = -b\frac{1}{x} + \frac{-y}{x} \\
#     b = -ax - y 
# $$
#
# Für Bildpixel, die 0 sind, wird eine invalide Matrix ausgegeben. Diese wird im späteren Verlauf herausgefiltert.

# +
nan_m = np.array([
        [np.nan, 0, 0],
        [0, np.nan, 0],
        [0, 0, np.nan]
      ])

def _value_filter_a_iterate(src_xy):
  src, x, y = src_xy
  if x == 0 or subspaces_item(src, (x, y)) == 0:
    return nan_m
  return np.array([
    [-1/x, 0, -y/x],
    [1, 0, 0],
    [0, 0, 1]
  ])

def _value_filter_b_iterate(src_xy):
  src, x, y = src_xy
  if subspaces_item(src, (x, y)) == 0:
    return nan_m
  return np.array([
    [1, 0, 0],
    [0, -x, -y],
    [0, 0, 1]
  ])


# -

# Zuletzt die Funktion `hough_vec`, die zunächst alle `(x,y)` Paare durch ein cartesisches Produkt der beiden möglichen Achsenwerte des gesamten Subspaces ermittelt (`prod`) und daraufhin parallel die Matrizen berechnet. Die invaliden Matrizen werden gefiltert und schließlich werden auf allen verfügbaren Kernen die Hough Durchgänge vollzogen. Da die Datenmenge hier extrem ansteigt und die CPU-Kerne möglichst augenutzt werden sollten, wird zudem die maximale Chunk Size ausgerechnet.
#
# Die Chunk Size gibt an, wie viele Matrizen ein Prozess ausrechnen soll. Wenn eine Chunk Size von 1 gewählt wird, resultiert dies in exzessiver Inter-Prozess-Kommunikation und die Prozessorkerne werden nicht ausgelastet. Wählt man die Chunk Size zu hoch, wird die dabei entstehende Menge an Subspaces so groß, dass
# man den verfügbaren Speicher überlastet oder sogar einen Out of Memory erhält.
#
# $$
# max(chunksize) < \frac{memory}{overhead * 3 * (resolution * resolution * sizeof(element))}
# $$
#
# Eine weitere Optimierung wäre es, die Addition der einzelnen Subspaces zum Schluss zu parallelisieren. Über eine Art Map-Reduce Verfahren. Im Augenblick werden diese zum übergeordneten Subspace dazugerechnet, sobald diese verfügbar sind. Außerdem lassen sich verschiedene wiederholende Operationen auch vorberechnen. Unter anderem die Matrizen verändern sich nicht bei der selben Resolution, genauso wie die Achsen.

def hough_vec(src: subspaces_t, max_memory: int) \
                                                -> subspaces_t:
  resolution = subspaces_resolution(src)
  cpus = multi.cpu_count() - 2
  ss = subspaces_create(resolution)
  values, _ = subspace_axis(resolution)
  
  # cartesian product
  prod = cartesian_product(values, values)
  
  # needs the most amount of time
  with multi.Pool(multi.cpu_count()) as p:
    iterate = [[src, x, y] for x, y in prod]
    chunksize = max(1, (len(iterate) // multi.cpu_count()))
    Ma_iterate = np.array(p.map(_value_filter_a_iterate, 
                            iterate, chunksize = chunksize))
    Mb_iterate = np.array(p.map(_value_filter_b_iterate, 
                            iterate, chunksize = chunksize))

  # filter out matrices where both are nan for the same pixel
  M_iterate_filter = ~np.isnan(Ma_iterate[:,0,0]) & \
                     ~np.isnan(Mb_iterate[:,0,0])
  
  Ma_iterate = Ma_iterate[M_iterate_filter].reshape(-1, 3, 3)
  Mb_iterate = Mb_iterate[M_iterate_filter].reshape(-1, 3, 3)
  
  M_iterate = np.column_stack((Ma_iterate, Mb_iterate, 
                               np.full((len(Ma_iterate), 3, 3),
                                       resolution)))

  print("done setup")
  with multi.Pool(multi.cpu_count()) as p:
    size_of_subspaces =  2 * 3 * (resolution * resolution)
    max_chunk_size = max_memory // (size_of_subspaces * cpus)
    chunksize = min(max_chunk_size, 
                    max(1,
                        (len(M_iterate) // multi.cpu_count())))
    
    print(("chunksize: {} on {} GiB total memory with " + \
           "subspaces_size = {} and max_chunk = {}")
          .format(
            chunksize, max_memory / 1024**3, 
            size_of_subspaces, max_chunk_size))
    
    iterator = p.imap_unordered(_hough_vec_process, 
                                M_iterate, 
                                chunksize = chunksize)
    
    for i, space in enumerate(iterator):
      subspaces_add_to(ss, space)
      draw.progress(i, len(M_iterate))

  return ss


# Erster Hough Durchlauf auf dem in Subspaces konvertierten Bild. In Layer 1 werden somit mögliche Geraden gefunden, die durch Kanten des Bildes verlaufen.

# %time layer1 = hough_vec(layer0, max_memory = max_memory)

# Vor jedem weiteren Hough Durchgang werden zunächst die Daten gefiltert. Dazu steht im Artikel:
# > ... As to the data read out at each layer, local maxima are selected. These discrete points are also the acutal data passed on to the next layer. In order to avoid closely spaced clusters of peaks, a non-maximum suppression is applied. The logarithm of peak height ist used as weightining factor for the votes it has at the next level. ... [1]
#
# In einem anderem Artikel von Tuytelaars steht dies zur Filterung:
# > ... Only truly 'non-accidental' structures are read out at the different layers. What that means is layer dependent: for the subsequent layers these are straight lines of a  minimum length, points where at least  three straight lines intersect, and lines that contain at least three line intersections. Note that in the latter case, these can be three intersections of each time two lines. ...  [2]
#
# Für diese Implementierung wird hauptsächlich die Filterung aus [1] verwendet, jedoch um gute Ergebnisse zu erhalten zum Teil auch die von [2]. Im Allgemeinen scheinen die Filter aus [2] sinnvoller zu sein als der generische Ansatz aus [1], der unsaubere Ergebnisse erzielt.

def filter_subspace(spaces: subspaces_t, 
                    nms: int, 
                    threshold: np.int32 = 3):
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
      local_max_space = np.log2(1 + local_max_space)\
                                              .astype(np.int32)
      out.append(local_max_space)
    return out


max_value = subspaces_max(layer1)
filtered_layer1 = filter_subspace(layer1, 
                          non_maximum_suppression_size,
                          threshold=max_value*layer1_threshold)


# %time layer2 = hough_vec(filtered_layer1, \
#                          max_memory = max_memory)

# Für Layer 2 ist es wichtig, nur Schnittpunkte zu verwenden, die laut [2] durch mindestens drei Linien führen. Ohne diesen zusätzlichen Schritt werden zu viele Ausreißer gefunden, die zwar das threshold von 3 Intersektionen erreichen, aber nicht am selben Punkt.
# [2] legt dabei nahe, dass für jeden Subspace Wert die dafür verantwortlichen Werte gespeichert werden. In dieser Implementierung wird dies rekonstruiert, was durch den in der Regel kleinen Suchraum kein Problem ist.  
# Die Diskretisierung erzeugt jedoch die Problematik, dass es sich eher um Schnittregionen als Schnittpunkte handelt. Dies muss bei der Filterung berücksichtigt werden, vor allem in Subspace 2 und 3.

def filter_intersect(lines: subspaces_t, 
                     intersects: subspaces_t, 
                     lines_intersecting_threshold: int = 3):  
  resolution = subspaces_resolution(lines)
  mask = subspaces_create(resolution, dtype = np.bool_)
  
  # map to homogeneous linear equations
  linear_equations = [np.array([-a, -1, -b]) \
                      for a, b in subspaces_peaks(lines, 1)]
  # map to homogeneous points
  homogene_points = [np.array([x, y, 1]) \
                    for x, y in subspaces_peaks(intersects, 1)]
  for p in homogene_points:
    # find offset for point p and use only the maximum
    offset = max(subspace_offset(resolution, p[0], p[1]))
    count = 0
    for eq in linear_equations:
      if abs(eq @ p) < offset: 
        count += 1
      if count >= lines_intersecting_threshold:
        # found minium intersections needed
        subspaces_itemset(mask, p[:2], True)
        break
  # apply mask to intersection space
  return [np.where(mask_space, intersects[i], 0) \
          for i, mask_space in enumerate(mask)]


filtered_layer2 = filter_subspace(layer2, 
                                  non_maximum_suppression_size)
filtered_layer2 = filter_intersect(filtered_layer1, 
                                   filtered_layer2)

# %time layer3 = hough_vec(filtered_layer2, \
#                          max_memory = max_memory)

filtered_layer3 = filter_subspace(layer3, 
                                  non_maximum_suppression_size)

# ## 5. Ergebnis
# Nach den Hough Durchgängen lassen sich bei Bildern mit markanten Linien und Fluchtpunkten
# gute Ergebnisse erzielen. Bilder ohne eindeutigen Fluchtpunkt haben oft nicht genug passende Geraden, um nach der Filterung von Layer 1 einen Fluchtpunkt zu finden, oder erzeugen viele scheinbare Fluchtpunkte. Dies könnte mitunter jedoch an unterschiedlichen Filtern liegen. Die Filterung der Subspaces ist der wichtigste und störungsanfälligste Teil des gesamten Algorithmus und leider in [1, 2] nicht genug beschrieben.

limit = 5
fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle("Bild mit Fluchtpunkten und Linien", fontsize=32)
fig.set_size_inches(9.1, 9.1)
axs.set_ylim((limit, -limit))
axs.set_xlim((-limit, limit))
axs.imshow(image, 'gray', extent=[-1, 1, 1, -1])
draw.plot_lines(axs, filtered_layer1)
draw.plot_intersect(axs, filtered_layer2, 'bo')
draw.plot_lines(axs, filtered_layer3, 'r--')

draw.plot_layer("layer 0: Mit Linien aus layer 1", 
                layer0, line_ss=filtered_layer1)
draw.plot_layer("layer 1: Mit Linien aus layer 2", 
                layer1, line_ss=filtered_layer2)
draw.plot_layer("layer 2", layer2)
draw.plot_layer("layer 3", layer3)
print("Points in Filtered Layer 1:", 
      [space[space > 0].size for space in filtered_layer1])
print("Points in Filtered Layer 2:", 
      [space[space > 0].size for space in filtered_layer2])
print("Points in Filtered Layer 3:", 
      [space[space > 0].size for space in filtered_layer3])

# ## Quellen
# **[1]** *Tuytelaars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998*
#
# **[2]** *Tuytelaars, T., Proesmans, M., & Van Gool, L. (1997). The cascaded Hough transform as support for 
#      grouping and finding vanishing points and lines. Lecture Notes in Computer Science, 278–289. 
#      doi:10.1007/bfb0017873*
#
#
