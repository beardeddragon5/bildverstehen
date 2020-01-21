# -*- coding: utf-8 -*-
# +
# %load_ext autoreload
# %autoreload 2

import subspace as sub
import numpy as np
import cv2
import multiprocessing as multi
import draw
from matplotlib import pyplot as plt


# -

# ### 4.1 Multiprozessing Hough Algorithmus
#
# Die Hough Transformation wurde aufgrund des einfachen Algorithmus der auch auf [Wikipedia]( 
# https://de.wikipedia.org/wiki/Hough-Transformation) zu finden ist implementiert.
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
# Der Code musste auf die Subspaces angepasst werden. Der größte Unterschied ist das durchlaufen des Raumes.
# Zunächst wird dafür eine virtuelle Achse gebildet die jede diskrete Zelle in allen 3 Subspaces einmal durchläuft. Außerdem muss im Gegesatz zum normalen hough beide Achsen des Parameterraums durchlaufen werden.
# Sonst werden duch die Diskretisierung viele Geraden nicht gefunden. 
#
# In dem folgenden Codeblock ist ein Durchlauf eines (x, y) paares auf den Subspaces. Dies ist equivalent zu
# diesem Teil des pseudocodes.
#
# $
# \texttt{for } \alpha := 0 \texttt{ to } \pi \texttt{ do } \\
# \hspace{0.5cm}  \texttt{d := pixel}_x \cdot \cos(\alpha ) + \texttt{pixel}_y \cdot \sin(\alpha ) \\
# \hspace{0.5cm}  \texttt{houghRaum}[\alpha][\texttt{d}]++ \\
# \texttt{end}\\
# $
#
# Nachdem die subspace Achse berechnet wurde werden sie umgeformt, dass man jeweils Wertepaare `(x, x, 1)` erhält. Im nächsten Schritt werden über entsprechende Matrizen entweder `a` oder `b` für den jeweiligen Wert ausgerechnet und in einem Subspace eingefügt. Der Wert wird dabei auf `1` gesetzt, dadurch wird ein Wert für den selben Wert von `x` o.`y` nicht doppelt gesetzt.

def _hough_vec_process(input_values):
  resolution = int(input_values[6, 0])
  Ma = input_values[0:3]
  Mb = input_values[3:6]
  
  values, _ = sub.subspace_axis(resolution)
  values = np.c_[ values, values, np.ones(len(values)) ]
  
  space = sub.subspaces_create(resolution, dtype = np.uint8)
  for M in [Ma, Mb]:
    ab_array = np.einsum('ij,...j', M, values)[:, :2]
    ab_array = ab_array[~np.isnan(ab_array[:, 0])]
    if ab_array.size > 0:
      # long running
      np.apply_along_axis(lambda ab: sub.subspaces_itemset(space, ab, 1), 1, ab_array)
  
  return space


# Die beiden Matrizen werden bestimmt durch Umformung von
#
# $$
#     ax + b + y = 0
# $$
#
# und entsprechende Umwandlung in eine Matrix. Die Umformungen lauten dabei.
#
# $$
#     a = -b\frac{1}{x} + \frac{-y}{x} \\
#     b = -ax - y 
# $$
#
# Für Bildpixel die 0 sind wird eine invalide Matrix ausgegeben, diese wird im späteren Verlauf
# herausgefiltert.

# +
nan_m = np.array([
        [np.nan, 0, 0],
        [0, np.nan, 0],
        [0, 0, np.nan]
      ])

def _value_filter_a_iterate(src_xy):
  src, x, y = src_xy
  if x == 0 or sub.subspaces_item(src, (x, y)) == 0:
    return nan_m
  return np.array([
    [-1/x, 0, -y/x],
    [1, 0, 0],
    [0, 0, 1]
  ])

def _value_filter_b_iterate(src_xy):
  src, x, y = src_xy
  if sub.subspaces_item(src, (x, y)) == 0:
    return nan_m
  return np.array([
    [1, 0, 0],
    [0, -x, -y],
    [0, 0, 1]
  ])


# -

# Zuletzt die Funktion `hough_vec` die zunächst alle `(x,y)` paare durch cartesisches Produkt der
# beiden möglichen Achsenwerte des gesamten Subspaces ermittelt (`prod`). Daraufhin parallel die
# matrizen berechnet. Die invaliden Matrizen werden gefiltert und schließlich werden auf allen
# verfügbaren Kernen die Hough durchgänge vollzogen. Da die Datenmenge hier extrem ansteigt und die
# Cpu-Kerne möglichst augenutzt werden sollten wird zudem die maximale chunk size ausgerechnet.
#
# Die Chunk Size gibt an wieviel Matrizen ein Prozess ausrechnen soll. Wenn eine chunk size von 1
# gewählt wird resultiert dies in excesiver IPC und die Prozessorkerne werden nicht ausgelastet.
# Wählt man die Chunk Size zu hoch wird die dabei entstehende Menge an Subspaces so groß, dass
# man den verfügbaren Speicher überlastet, oder sogar einen Out of Memory erhält.
#
# $$
# max(chunksize) < \frac{memory}{overhead * 3 * (resolution * resolution * sizeof(subspace\; element))}
# $$
#
# Eine weitere Optimierung wäre es die Addition der einzelnen subspaces zum Schluss zu parallelisieren.
# Über eine Art Map-Reduce Verfahren. Im Augenblick werden diese zum übergeordneten subspace dazugerechnet
# sobald diese verfügbar sind. Außerdem lassen sich verschiedene wiederholende Operationen auch vorberechnen.
# Unter andem die Matrizen verändern sich nicht bei der selben resolution, oder auch die achsen.

def hough_vec(src: sub.subspaces_t, max_memory: int) -> sub.subspaces_t:
  resolution = sub.subspaces_resolution(src)
  cpus = multi.cpu_count() - 2
  ss = sub.subspaces_create(resolution)
  values, _ = sub.subspace_axis(resolution)
  
  # cartesian product
  prod = sub.cartesian_product(values, values)
  
  # needs the most amount of time
  with multi.Pool(multi.cpu_count()) as p:
    iterate = [[src, x, y] for x, y in prod]
    chunksize = max(1, (len(iterate) // multi.cpu_count()))
    Ma_iterate = np.array(p.map(_value_filter_a_iterate, iterate, chunksize = chunksize))
    Mb_iterate = np.array(p.map(_value_filter_b_iterate, iterate, chunksize = chunksize))

  # filter out matrices where both are nan for the same pixel
  M_iterate_filter = ~np.isnan(Ma_iterate[:,0,0]) & ~np.isnan(Mb_iterate[:,0,0])
  
  Ma_iterate = Ma_iterate[M_iterate_filter].reshape(-1, 3, 3)
  Mb_iterate = Mb_iterate[M_iterate_filter].reshape(-1, 3, 3)
  
  M_iterate = np.column_stack((Ma_iterate, Mb_iterate, np.full((len(Ma_iterate), 3, 3), resolution)))

  print("done setup")
  with multi.Pool(multi.cpu_count()) as p:
    size_of_subspaces =  2 * 3 * (resolution * resolution)
    max_chunk_size = max_memory // (size_of_subspaces * cpus)
    chunksize = min(max_chunk_size, max(1, (len(M_iterate) // multi.cpu_count())))
    print("chunksize: {} on {} GiB total memory with subspaces_size = {} and max_chunk = {}".format(chunksize, max_memory / 1024**3, size_of_subspaces, max_chunk_size))
    iterator = p.imap_unordered(_hough_vec_process, M_iterate, chunksize = chunksize)
    for i, space in enumerate(iterator):
      sub.subspaces_add_to(ss, space)
      draw.progress(i, len(M_iterate))

  return ss


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
