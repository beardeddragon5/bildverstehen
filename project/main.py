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
threshold = 1.0
resolution = None

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

# +
canny = cv2.Canny(blured_image, canny_min, canny_max)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(canny, cmap='gray')
plt.show()
# -

src = hough.from_image(resolution, canny)


# %time ss = hough.hough_vec(src)

# +
# # %time ss = hough.hough(src)
# -

# # Display Results

def subspace_normalize(s: sub.subspace_t):
  return cv2.normalize(s, None, alpha=0, beta=255,
                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 


# +
from ipywidgets import interact, FloatSlider

@interact(threshold = FloatSlider(threshold, min=0, max=1, step=0.05))
def plot_results(threshold: int):
  out = np.copy(image)
  
  threshold_pixel_value = int(sub.subspaces_max(ss) * threshold)
  
  [draw.g(out, ab) for ab in sub.subspaces_to_line(ss, threshold_pixel_value)]
  
  plt.imshow(out)
  plt.show()
  
  fig, axs = plt.subplots(1, 3, sharey=True)
  fig.set_size_inches(18.5, 10.5)
  for i in range(3):
    space = ss[i]
    filtered_space = np.where(space >= threshold_pixel_value, space, 0)
    axs[i].imshow(subspace_normalize(filtered_space), 'gray')
    
  fig, axs = plt.subplots(1, 3, sharey=True)
  fig.set_size_inches(18.5, 10.5)
  for i in range(3):
    axs[i].imshow(subspace_normalize(src[i]), 'gray')
# -


