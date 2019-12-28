'''
Distance transform sample.
Usage:
  distrans.py [<image>]
Keys:
  ESC   - exit
  v     - toggle voronoi mode
'''


import numpy as np
import cv2 as cv

if __name__ == '__main__':
    import sys
    try: fn = sys.argv[1]
    except: fn = './BinaryObjectsI.png'

    img = cv.imread(fn, 0)
    # cm = make_cmap('jet')
    need_update = True
    voronoi = False

    def update(dummy=None):
        global need_update
        need_update = False
        thrs = cv.getTrackbarPos('threshold', 'distrans')
        dist = cv.distanceTransform(img, cv.DIST_L2, 5)
        # if voronoi:
        #     vis = cm[np.uint8(labels)]
        # else:
        #     vis = cm[np.uint8(dist*2)]
        # vis[mark != 0] = 255
        cv.imshow('distrans', dist)

    def invalidate(dummy=None):
        global need_update
        need_update = True

    cv.namedWindow('distrans')
    cv.createTrackbar('threshold', 'distrans', 60, 255, invalidate)
    update()


    while True:
        ch = 0xFF & cv.waitKey(50)
        if ch == 27:
            break
        if ch == ord('v'):
            voronoi = not voronoi
            print('showing', ['distance', 'voronoi'][voronoi])
            update()
        if need_update:
            update()
    cv.destroyAllWindows()
