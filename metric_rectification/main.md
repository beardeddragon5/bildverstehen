---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import utils
```

# Bild 1

```python
img = cv2.imread('IMG 1.png')

p1 = [1866.0, 530.0, 1.0]
p2 = [2771.0, 974.0, 1.0]
p3 = [1067.0, 961.0, 1.0]
p4 = [2112.0, 1646.0, 1.0]

l1 = np.cross(p1, p2)
l2 = np.cross(p3, p4)
m1 = np.cross(p1, p3)
m2 = np.cross(p2, p4)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()

plt.imshow(img)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 1 (mit Punkten).png', dpi = 300)
plt.show()
```

```python
h = utils.getAffineMatrix(l1, l2, m1, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
```

```python
start = time.time()
affineImg1 = utils.transformImage(img, h)
end = time.time()
print('Laufzeit: {:5.3f}s'.format(end - start))

affineImg1 = cv2.cvtColor(affineImg1, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 1 (affin) (1).png', affineImg1)
affineImg1 = cv2.cvtColor(affineImg1, cv2.COLOR_BGR2RGB)

plt.imshow(affineImg1)
plt.axis('off')
plt.show()

plt.imshow(affineImg1)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 1 (affin mit Punkten) (1).png', dpi = 300)
plt.show()
```

```python
start = time.time()
affineImg2 = cv2.warpPerspective(img, np.array(h), (img.shape[1], img.shape[0]))
end = time.time()
print('Laufzeit: {:5.3f}s'.format(end - start))

affineImg2 = cv2.cvtColor(affineImg2, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 1 (affin) (2).png', affineImg2)
affineImg2 = cv2.cvtColor(affineImg2, cv2.COLOR_BGR2RGB)

plt.imshow(affineImg2)
plt.axis('off')
plt.show()

plt.imshow(affineImg2)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 1 (affin mit Punkten) (2).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
```

```python
l1 = np.cross(p1, p2)
m1 = np.cross(p1, p3)
l2 = np.cross(p1, p4)
m2 = np.cross(p3, p2)

h = utils.getMetricMatrix(l1, m1, l2, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
```

```python
start = time.time()
metricImg1 = utils.transformImage(affineImg1, h)
end = time.time()
print('Laufzeit: {:5.3f}s'.format(end - start))

metricImg1 = cv2.cvtColor(metricImg1, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 1 (metrisch) (1).png', metricImg1)
metricImg1 = cv2.cvtColor(metricImg1, cv2.COLOR_BGR2RGB)

plt.imshow(metricImg1)
plt.axis('off')
plt.show()

plt.imshow(metricImg1)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p1[0], p4[0]], [p1[1], p4[1]])
plt.plot([p3[0], p2[0]], [p3[1], p2[1]])
plt.axis('off')
plt.savefig('IMG 1 (metrisch mit Punkten) (1).png', dpi = 300)
plt.show()
```

```python
start = time.time()
metricImg2 = cv2.warpPerspective(affineImg2, np.array(h), (img.shape[1], img.shape[0]))
end = time.time()
print('Laufzeit: {:5.3f}s'.format(end - start))

metricImg2 = cv2.cvtColor(metricImg2, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 1 (metrisch) (2).png', metricImg2)
metricImg2 = cv2.cvtColor(metricImg2, cv2.COLOR_BGR2RGB)

plt.imshow(metricImg2)
plt.axis('off')
plt.show()

plt.imshow(metricImg2)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p1[0], p4[0]], [p1[1], p4[1]])
plt.plot([p3[0], p2[0]], [p3[1], p2[1]])
plt.axis('off')
plt.savefig('IMG 1 (metrisch mit Punkten) (2).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
```

# Bild 2

```python
img = cv2.imread('IMG 2.png')

p1 = [1639.0, 199.0, 1.0]
p2 = [1888.0, 151.0, 1.0]
p3 = [1598.0, 1657.0, 1.0]
p4 = [1849.0, 1678.0, 1.0]
p5 = [2251.0, 463.0, 1.0]
p6 = [2524.0, 201.0, 1.0]
p7 = [3424.0, 1801.0, 1.0]
p8 = [3799.0, 1530.0, 1.0]

l1 = np.cross(p1, p3)
l2 = np.cross(p2, p4)
m1 = np.cross(p5, p7)
m2 = np.cross(p6, p8)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()

plt.imshow(img)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.scatter(p8[0], p8[1])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.plot([p5[0], p7[0]], [p5[1], p7[1]])
plt.plot([p6[0], p8[0]], [p6[1], p8[1]])
plt.axis('off')
plt.savefig('IMG 2 (mit Punkten).png', dpi = 300)
plt.show()
```

```python
h = utils.getAffineMatrix(l1, l2, m1, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
p5 = utils.transformPoint(p5, h)
p6 = utils.transformPoint(p6, h)
p7 = utils.transformPoint(p7, h)
p8 = utils.transformPoint(p8, h)
```

```python
affineImg = cv2.warpPerspective(img, np.array(h), (img.shape[1], img.shape[0]))

affineImg = cv2.cvtColor(affineImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 2 (affin).png', affineImg)
affineImg = cv2.cvtColor(affineImg, cv2.COLOR_BGR2RGB)

plt.imshow(affineImg)
plt.axis('off')
plt.show()

plt.imshow(affineImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.scatter(p8[0], p8[1])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.plot([p5[0], p7[0]], [p5[1], p7[1]])
plt.plot([p6[0], p8[0]], [p6[1], p8[1]])
plt.axis('off')
plt.savefig('IMG 2 (affin mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
print(p8)
```

```python
l1 = np.cross(p1, p2)
m1 = np.cross(p1, p3)
l2 = np.cross(p5, p6)
m2 = np.cross(p5, p7)

h = utils.getMetricMatrix(l1, m1, l2, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
p5 = utils.transformPoint(p5, h)
p6 = utils.transformPoint(p6, h)
p7 = utils.transformPoint(p7, h)
p8 = utils.transformPoint(p8, h)
```

```python
metricImg = cv2.warpPerspective(affineImg, np.array(h), (img.shape[1], img.shape[0]))

metricImg = cv2.cvtColor(metricImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 2 (metrisch).png', metricImg)
metricImg = cv2.cvtColor(metricImg, cv2.COLOR_BGR2RGB)

plt.imshow(metricImg)
plt.axis('off')
plt.show()

plt.imshow(metricImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.scatter(p8[0], p8[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p5[0], p6[0]], [p5[1], p6[1]])
plt.plot([p5[0], p7[0]], [p5[1], p7[1]])
plt.axis('off')
plt.savefig('IMG 2 (metrisch mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
print(p8)
```

# Bild 3

```python
img = cv2.imread('IMG 3.png')

p1 = [1416.0, 286.0, 1.0]
p2 = [3049.0, 674.0, 1.0]
p3 = [1231.0, 479.0, 1.0]
p4 = [2980.0, 932.0, 1.0]
p5 = [2693.0, 685.0, 1.0]
p6 = [3508.0, 987.0, 1.0]
p7 = [3093.0, 2134.0, 1.0]

l1 = np.cross(p1, p2)
l2 = np.cross(p3, p4)
m1 = np.cross(p1, p3)
m2 = np.cross(p2, p4)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()

plt.imshow(img)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 3 (mit Punkten).png', dpi = 300)
plt.show()
```

```python
h = utils.getAffineMatrix(l1, l2, m1, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
p5 = utils.transformPoint(p5, h)
p6 = utils.transformPoint(p6, h)
p7 = utils.transformPoint(p7, h)
```

```python
affineImg = cv2.warpPerspective(img, np.array(h), (img.shape[1], img.shape[0]))

affineImg = cv2.cvtColor(affineImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 3 (affin).png', affineImg)
affineImg = cv2.cvtColor(affineImg, cv2.COLOR_BGR2RGB)

plt.imshow(affineImg)
plt.axis('off')
plt.show()

plt.imshow(affineImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 3 (affin mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
```

```python
l1 = np.cross(p1, p2)
m1 = np.cross(p1, p3)
l2 = np.cross(p6, p5)
m2 = np.cross(p6, p7)

h = utils.getMetricMatrix(l1, m1, l2, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
p5 = utils.transformPoint(p5, h)
p6 = utils.transformPoint(p6, h)
p7 = utils.transformPoint(p7, h)
```

```python
metricImg = cv2.warpPerspective(affineImg, np.array(h), (img.shape[1], img.shape[0]))

metricImg = cv2.cvtColor(metricImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 3 (metrisch).png', metricImg)
metricImg = cv2.cvtColor(metricImg, cv2.COLOR_BGR2RGB)

plt.imshow(metricImg)
plt.axis('off')
plt.show()

plt.imshow(metricImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.scatter(p5[0], p5[1])
plt.scatter(p6[0], p6[1])
plt.scatter(p7[0], p7[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p6[0], p5[0]], [p6[1], p5[1]])
plt.plot([p6[0], p7[0]], [p6[1], p7[1]])
plt.axis('off')
plt.savefig('IMG 3 (metrisch mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
```

# Bild 4

```python
img = cv2.imread('IMG 4.png')

p1 = [78.0, 215.0, 1.0]
p2 = [433.0, 86.0, 1.0]
p3 = [361.0, 442.0, 1.0]
p4 = [731.0, 237.0, 1.0]

l1 = np.cross(p1, p2)
l2 = np.cross(p3, p4)
m1 = np.cross(p1, p3)
m2 = np.cross(p2, p4)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()

plt.imshow(img)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 4 (mit Punkten).png', dpi = 300)
plt.show()
```

```python
h = utils.getAffineMatrix(l1, l2, m1, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
```

```python
affineImg = cv2.warpPerspective(img, np.array(h), (img.shape[1], img.shape[0]))

affineImg = cv2.cvtColor(affineImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 4 (affin).png', affineImg)
affineImg = cv2.cvtColor(affineImg, cv2.COLOR_BGR2RGB)

plt.imshow(affineImg)
plt.axis('off')
plt.show()

plt.imshow(affineImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p3[0], p4[0]], [p3[1], p4[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p2[0], p4[0]], [p2[1], p4[1]])
plt.axis('off')
plt.savefig('IMG 4 (affin mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
```

```python
l1 = np.cross(p1, p2)
m1 = np.cross(p1, p3)
l2 = np.cross(p1, p4)
m2 = np.cross(p3, p2)

h = utils.getMetricMatrix(l1, m1, l2, m2)
```

```python
p1 = utils.transformPoint(p1, h)
p2 = utils.transformPoint(p2, h)
p3 = utils.transformPoint(p3, h)
p4 = utils.transformPoint(p4, h)
```

```python
metricImg = cv2.warpPerspective(affineImg, np.array(h), (img.shape[1], img.shape[0]))

metricImg = cv2.cvtColor(metricImg, cv2.COLOR_RGB2BGR)
cv2.imwrite('IMG 4 (metrisch).png', metricImg)
metricImg = cv2.cvtColor(metricImg, cv2.COLOR_BGR2RGB)

plt.imshow(metricImg)
plt.axis('off')
plt.show()

plt.imshow(metricImg)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(p3[0], p3[1])
plt.scatter(p4[0], p4[1])
plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
plt.plot([p1[0], p3[0]], [p1[1], p3[1]])
plt.plot([p1[0], p4[0]], [p1[1], p4[1]])
plt.plot([p3[0], p2[0]], [p3[1], p2[1]])
plt.axis('off')
plt.savefig('IMG 4 (metrisch mit Punkten).png', dpi = 300)
plt.show()
```

```python
print(p1)
print(p2)
print(p3)
print(p4)
```
