# +
# %load_ext autoreload
# %autoreload 2

import cv2
import hough
import draw
import subspace as sub
import os
import numpy as np
from matplotlib import pyplot as plt
# -

# # Configuration

image_path = '../images/Garagen.jpg'
canny_min = 230
canny_max = 255
threshold = 0.59
resolution = 600

# # Load Image

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

# # Crop and Resize image
# If no target resolution is provided 60% of `min(width, height)` will
# be used. The 60% are infered from the 600 to 1000 ratio used in the
# paper.

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

# # Run Hough Transform

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blured_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
plt.imshow(blured_image, 'gray')
plt.show()

# Using canny to get only points that seam allready be part of an line

# +
canny = cv2.Canny(blured_image, canny_min, canny_max)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(canny, cmap='gray')
plt.show()
# -

# Input image must be converted to an subspace image. Like destribed in paper the image  
# will be rescaled and in this implementation cropped to fit into the first subspace.  
# Resulting in layer 0.

src = hough.from_image(resolution, canny)

# Get layer 1 from the result of applying the hough transformation on layer 0

# %time ss = hough.hough_vec(src)

# Apply filter to each subsequent layer of hough transformation
# > ... As to the data read out at each layer, local maxima are selected. These discrete points are also the acutal data passed on to the next layer. In order to avoid closely spaced clusters of peaks, a non-maximum suppression is applied. The logarithm of peak height ist used as weightining factor for the votes it has at the next level. ...  
# (Tuytellars et al., The Cascaded Hough Transform as an Aid in Aerial Image Interpretation, ICCV-1998)

# +
from skimage.feature import peak_local_max

def filter_subspace(spaces: sub.subspaces_t):
    out = []
    for i, space in enumerate(spaces):
      # try not to lose information of min_max in image
      max_value = sub.subspaces_max(ss) / 255
      float_space = cv2.normalize(space, None, alpha=0, beta=max_value, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      
      # applying log to float space 
      log_weights = np.log2(1 + float_space)
      
      # weight the space with results in log_weights
      weighted_space = float_space * log_weights
      
      # using min_distance=1 to get max peaks
      # non-maximum suppression is added later
      local_max_mask = peak_local_max(weighted_space, min_distance=1, indices=False, threshold_rel=0.9) # , threshold_rel=0.5
      local_max_space = np.where(local_max_mask, space, 0)
      
      out.append(cv2.normalize(local_max_space, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    return out


# +
# # %time ss = hough.hough(src)

# +
# threshold_pixel_value = int(sub.subspaces_max(ss) * threshold)
# in_ss = [np.where(space >= threshold_pixel_value, space, 0) for space in ss]

# %time ss2 = hough.hough_vec(filter_subspace(ss))
# -

# %time ss3 = hough.hough_vec(filter_subspace(ss2))

# # Display Results

def subspace_normalize(s: sub.subspace_t):
  return cv2.normalize(s, None, alpha=0, beta=255,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 


# +
from ipywidgets import interact, FloatSlider
# #+ from skimage.feature import peak_local_max

@interact(threshold = FloatSlider(threshold, min=0, max=1, step=0.05))
def plot_results(threshold: float):
  out = np.copy(image)
  [draw.g(out, ab) for ab in sub.subspaces_to_line(ss, threshold)]
  plt.imshow(out)
  plt.show()
  
  fig, axs = plt.subplots(1, 3, sharey=True)
  fig.suptitle("layer 0", fontsize=32)
  fig.set_size_inches(30, 10.5)
  for i in range(3):
    axs[i].imshow(subspace_normalize(src[i]), 'gray')

  fig, axs = plt.subplots(2, 3, sharey=True)
  fig.suptitle("layer 1", fontsize=32)
  fig.set_size_inches(30, 21)
  for i, space in enumerate(ss):
    axs[0, i].imshow(subspace_normalize(space), 'gray')
  for i, space in enumerate(filter_subspace(ss)):
    axs[1, i].imshow(subspace_normalize(space), 'gray')
  
  fig, axs = plt.subplots(2, 3, sharey=True)
  fig.suptitle("layer 2", fontsize=32)
  fig.set_size_inches(30, 21)
  for i, space in enumerate(ss2):
    axs[0, i].imshow(subspace_normalize(space), 'gray')
  for i, space in enumerate(filter_subspace(ss2)):
    axs[1, i].imshow(subspace_normalize(space), 'gray')
    
  fig, axs = plt.subplots(2, 3, sharey=True)
  fig.suptitle("layer 3", fontsize=32)
  fig.set_size_inches(30, 21)
  for i, space in enumerate(ss3):
    axs[0, i].imshow(subspace_normalize(space), 'gray')
  for i, space in enumerate(filter_subspace(ss3)):
    axs[1, i].imshow(subspace_normalize(space), 'gray')
# -




