# Simple QR decomposition package.

## features of this package:
* QR decomposition using Hauseholder reflexion (`transform` method)
* OLS using QR decomposition (`OLS` method)

# Some examples:

you can find some more examples on tail of **QR.py** file.

```python
from QR import QR

qr = QR()

A = np.array([
    [1,2,20],
    [4,5,155],
    [7,8,10]
], dtype='float64')
b = np.array([
    -1,
    2,
    3
], dtype='float64')

x, residual_norm = qr.OLS(A, b)
```
