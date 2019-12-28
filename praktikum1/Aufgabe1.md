# Aufgabe1

## a) Gegeben seien die Geraden `y = 4x + 1` und `y = 4x − 3`. Berechnen Sie deren Schnittpunkt *S*

1) In die Form a·x + b·y + c = 0 umwandeln
```
      y = 4x + 1
<=>   0 = 4x - y + 1
 =>   g1 = (4, -1, 1)^T

      y = 4x - 3
<=>   0 = 4x - y - 3
 =>   g2 =  (4, -1, -3)^T
```
2) Schnittpunkt durch Kreuzprodukt
```
          | i   j   k |   / -1 * -3  - (-1) *  1 \   /  4 \
g1 x g2 = | 4  -1   1 | = |  4 *  1  -   4  * -3 | = | 16 | = S
          | 4  -1  -3 |   \  4 * -1  -   4  * -1 /   \  0 /
```

## b) Berechnen Sie die Gerade l durch S und P = (5, 8)T
```
        | i   j  k |   /  16   - 0      \   /  16 \   /   4 \
S x P = | 4  16  0 | = |   0   - 4      | = |  -4 | = |  -1 | = l
        | 5   8  1 |   \ 4 * 8 - 5 * 16 /   \ -48 /   \ -12 /

    0 = 4x - y - 12
<=> y = 4x - 12
```
## c) Überprüfen Sie, ob der Punkt Q = (10, 6)^T auf der Geraden l liegt.
```
                  /   4 \
     (10, 6, 1) · |  -1 |      = 0
                  \ -12 /
<=>  10 * 4 + 6 * -1 + 1 * -12 = 0
<=>  22                        = 0  ERROR
```
## d) Geben Sie die homogenen Koordinaten von einem Kreis an, der den nicht-homogenen Mittelpunkt (2, 3) und den Radius 4 hat.
- Verifizieren Sie, dass der homogene Punkt (6, 3, 1) auf dem Kreis liegt.
```
Allgemein: (x-a·k)^2 + (y-b·k)^2 = r^2·k^2
 => (x - 2 * k)^2 + (y - 3 * k)^2 = 16 * k^2
 => (6 - 2 * 1)^2 + (3 - 3 * 1)^2 = 16 * 1^2
<=> 16 = 16
```
- Verifizieren Sie, dass der homogene Punkt (3, 3i, 0) auf dem Kreis liegt.
```
 => (x - 2 * k)^2 + (y - 3 * k)^2 = 16 * k^2
 => (3 - 2 * 0)^2 + (3i - 3 * 0)^2 = 16 * 0^2
<=> 0 = 0
```

## e) Gegeben sei die 3-d Rotationsmatrix
```
    / 0.4534747 −0.8528619  0.2588190 \
R = | 0.6606284  0.1267112 −0.7399421 |
    \ 0.5982731  0.5065283  0.6208852 /
```
1. Überzeugen Sie sich davon, dass es sich um eine Rotationsmatrix handelt.
```
det(R) =
  + 0.4534747 * 0.1267112 * 0.6208852
  + (−0.8528619) * (−0.7399421) * 0.5982731
  + 0.2588190 * 0.6606284 * 0.5065283
  - 0.2588190 * 0.1267112 * 0.5982731
  - (−0.8528619) * 0.6606284 * 0.6208852
  - 0.4534747 * (−0.7399421) * 0.5065283
  = 1
```
2. Berechnen Sie die Rotationswinkel α, β, γ.

