import cv2 as cv
import numpy as np
import time
import sys

def findLocalMaxima(image):
  rows, cols = image.shape
  maximas = np.zeros((rows, cols, 1), np.float32)
  for row in range(1, rows-1):
    for col in range(1, cols-1):
      v = image[row, col] > 0 and \
          image[row, col] >= image[row, col + 1] and \
          image[row, col] >= image[row, col - 1] and \
          image[row, col] >= image[row + 1, col] and \
          image[row, col] >= image[row - 1, col] and \
          image[row, col] >= image[row + 1, col + 1] and \
          image[row, col] >= image[row - 1, col - 1] and \
          image[row, col] >= image[row - 1, col + 1] and \
          image[row, col] >= image[row + 1, col - 1]
      maximas[row][col] = 1 if v else 0

  return maximas


def addOverlay(inputImage, overlay):
  rows, cols = inputImage.shape

  # add more dimensions
  dist = cv.cvtColor(inputImage, cv.COLOR_GRAY2RGB)
  overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2RGB)

  # normalize everything to have the same values
  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
  cv.normalize(overlay, overlay, 0, 1.0, cv.NORM_MINMAX)

  # create inverted pink image
  image = np.zeros((rows, cols, 3), np.float32)
  image[:] = (1.0 - 1.0, 1.0 - 0.41, 1.0 - 0.71)

  # remove values from white areas to result in pink
  dist = dist - (overlay * image)
  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
  return dist


def main():
  image = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)

  # distance transform of image
  output = cv.distanceTransform(image, cv.DIST_L1, cv.DIST_MASK_3);
  cv.normalize(output, output, 0, 1.0, cv.NORM_MINMAX)

  # helper to add mouse interaction
  def show_position(window, image):
    def _listener(event, x, y, a, b):
      cv.displayOverlay(window, "{} {} color: {}".format(x, y, image[y][x]), 0)
    return _listener

  # show input image
  cv.imshow('input', image)
  cv.setMouseCallback('input', show_position('input', image))

  # add local maxima to input image
  dist = addOverlay(image, findLocalMaxima(output))

  # show input with overlay
  cv.imshow('input with overlay', dist)
  cv.setMouseCallback('input with overlay', show_position('input with overlay', dist))

  # show distance transform result
  cv.imshow('output', output)
  cv.setMouseCallback('output', show_position('output', output))

  cv.waitKey(0)
  cv.destroyAllWindows()

main()
