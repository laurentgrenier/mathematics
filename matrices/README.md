# matrices


> Each formulas URLs are generated using the website https://www.codecogs.com/latex/eqneditor.php  
## Introduce matrices
### A matrix
![\[\begin{cases} & 2a + 3b = 8 \\ & 10a + b = 13 \end{cases}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bcases%7D%20%26%202a%20&plus;%203b%20%3D%208%20%5C%5C%20%26%2010a%20&plus;%20b%20%3D%2013%20%5Cend%7Bcases%7D)  
can be written as a matrix \
![\[\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} 8 \\ 13 \end{pmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%202%20%26%203%20%5C%5C%2010%20%26%201%20%5Cend%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%20a%20%5C%5C%20b%20%5Cend%7Bpmatrix%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%208%20%5C%5C%2013%20%5Cend%7Bpmatrix%7D)

```python
import numpy as np
m = np.array([[2, 3], [10, 1]])
```

## Multiply by a basis vector
![\[\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 10 \end{pmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%202%20%26%203%20%5C%5C%2010%20%26%201%20%5Cend%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%201%20%5C%5C%200%20%5Cend%7Bpmatrix%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%202%20%5C%5C%2010%20%5Cend%7Bpmatrix%7D)


```python
import numpy as np
m = np.array([[2, 3], [10, 1]])
e1 = [1, 0]
m.dot(e1)
```
> **Linear algebra** \
> Linear algebra is a mathematical system for manipulating vectors in spaces described by vectors.

## How matrices transform space

![\[A = \begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix}\]](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cbegin%7Bpmatrix%7D%202%20%26%203%20%5C%5C%2010%20%26%201%20%5Cend%7Bpmatrix%7D)

![\[r = \begin{bmatrix} a \\ b \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?r%20%3D%20%5Cbegin%7Bbmatrix%7D%20a%20%5C%5C%20b%20%5Cend%7Bbmatrix%7D)

![\[{r}' = \begin{bmatrix} 8 \\ 13 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%7Br%7D%27%20%3D%20%5Cbegin%7Bbmatrix%7D%208%20%5C%5C%2013%20%5Cend%7Bbmatrix%7D)

![\[A.r = {r}'\]](https://latex.codecogs.com/gif.latex?A.r%20%3D%20%7Br%7D%27)

### rule 1 
![\[A.(n.r) = n.{r}'\]](https://latex.codecogs.com/gif.latex?A.%28n.r%29%20%3D%20n.%7Br%7D%27)

### rule 2
![\[A.(r+s) = A.r + A.s\]](https://latex.codecogs.com/gif.latex?A.%28r&plus;s%29%20%3D%20A.r%20&plus;%20A.s)

### rule 3
![\[A.(n.\hat{e}_{1} + m.\hat{e}_{2}) = n.A.\hat{e}_{1} + m.A.\hat{e}_{2} = n{e_{1}}'+ m{e_{2}}'\]](https://latex.codecogs.com/gif.latex?A.%28n.%5Chat%7Be%7D_%7B1%7D%20&plus;%20m.%5Chat%7Be%7D_%7B2%7D%29%20%3D%20n.A.%5Chat%7Be%7D_%7B1%7D%20&plus;%20m.A.%5Chat%7Be%7D_%7B2%7D%20%3D%20n%7Be_%7B1%7D%7D%27&plus;%20m%7Be_%7B2%7D%7D%27)

## Types of matrix transformation

### Identity matrix
![\[\begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%201%20%26%200%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
r = np.array([1, 2])
identity = np.array([[1, 0], [0, 1]])
np.dot(identity, r)
```

### Scalling matrix
#### Upscale
![\[\begin{bmatrix} 3 & 0\\ 0 & 2 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%203%20%26%200%5C%5C%200%20%26%202%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
r = np.array([1, 2])
upscaling = np.array([[3, 0], [0, 2]])
np.dot(upscaling, r)
```

#### Downscale
![\[\begin{bmatrix} \frac{1}{3} & 0\\ 0 & \frac{1}{2} \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B1%7D%7B3%7D%20%26%200%5C%5C%200%20%26%20%5Cfrac%7B1%7D%7B2%7D%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
r = np.array([1, 2])
downscaling = np.array([[1/3, 0], [0, 1/2]])
np.dot(downscaling, r)
```

### Invert matrix
![\[\begin{bmatrix} -1 & 0\\ 0 & 2 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20-1%20%26%200%5C%5C%200%20%26%202%20%5Cend%7Bbmatrix%7D)
 or 
![\[\begin{bmatrix} -1 & 0\\ 0 & -1 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20-1%20%26%200%5C%5C%200%20%26%20-1%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
r = np.array([1, 2])
invert = np.array([[-1, 0], [0, -1]])
np.dot(invert, r)
```

### Shears matrices
If we want to keep e1 but move e2 to e2_prime, we can write a shear matrix: 

![\[\begin{bmatrix} 1 & 1\\ 0 & 1 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%201%20%26%201%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
shear = np.array([[1, 0], [1, 1]])
np.dot(shear, e1)
```

### Mirror matrices 

![\[\begin{bmatrix} 0 & 1\\ 1 & 0 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%200%20%26%201%5C%5C%201%20%26%200%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
mirror_a = np.array([[0, 1], [1, 0]])
np.dot(mirror_a, e1)
```

![\[\begin{bmatrix} -1 & 0\\ 0 & 1 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20-1%20%26%200%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
mirror_b = np.array([[0, -1], [-1, 0]])
np.dot(mirror_b, e1)
```

![\[\begin{bmatrix} 0 & -1\\ -1 & 0 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%200%20%26%20-1%5C%5C%20-1%20%26%200%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
mirror_c = np.array([[-1, 0], [0, 1]])
np.dot(mirror_c, e1)
```

![\[\begin{bmatrix} 1 & 0\\ 0 & -1 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%201%20%26%200%5C%5C%200%20%26%20-1%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
mirror_d = np.array([[1, 0], [0, -1]])
np.dot(mirror_d, e1)
```

### Rotate matrix
![\[\begin{bmatrix} cos(\theta) & sin(\theta)\\ -sin(\theta) & cos(\theta) \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20cos%28%5Ctheta%29%20%26%20sin%28%5Ctheta%29%5C%5C%20-sin%28%5Ctheta%29%20%26%20cos%28%5Ctheta%29%20%5Cend%7Bbmatrix%7D)

For example, if we want to rotate by 90° anticlockwise. 

![\[\begin{bmatrix} cos(-90) & sin(-90)\\ -sin(-90) & cos(-90) \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20cos%28-90%29%20%26%20sin%28-90%29%5C%5C%20-sin%28-90%29%20%26%20cos%28-90%29%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%200%20%26%20-1%20%5C%5C%201%20%26%200%20%5Cend%7Bbmatrix%7D)

```python
import numpy as np
e1 = np.array([1, 0])
def rotation_matrix(angle):
    return [[round(np.cos(angle), 3), round(np.sin(angle), 3)], [-round(np.sin(angle), 3), round(np.cos(angle), 3)]]
np.dot(rotation_matrix(-np.pi/2), e1)
```

## Determinants and inverses
The determinant change the scope of the space.

![\[det(A)=ad - bc\]](https://latex.codecogs.com/gif.latex?det%28A%29%3Dad%20-%20bc)

```python
import numpy as np
G = np.array([[1, -1], [-1, 3]])
np.linalg.det(G)
```

### Using the determinant for a matrix inverse calculation

Let ![\[A = \begin{pmatrix} a & b\\ c & d \end{pmatrix}\]](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cbegin%7Bpmatrix%7D%20a%20%26%20b%5C%5C%20c%20%26%20d%20%5Cend%7Bpmatrix%7D)

![\[\frac{1}{det(A)}\begin{pmatrix} d & -b\\ -c & a \end{pmatrix} = A^{-1}\]](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7Bdet%28A%29%7D%5Cbegin%7Bpmatrix%7D%20d%20%26%20-b%5C%5C%20-c%20%26%20a%20%5Cend%7Bpmatrix%7D%20%3D%20A%5E%7B-1%7D)

## Singular matrix
A matrix is singular if an inverse exists.

## Einstein summation convention
Les multiply two matrix. \
![\[\begin{pmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & ... & ... & ...\\ ... & ... & ... & ...\\ a_{n1} & ... & ... & a_{nn} \end{pmatrix} \begin{pmatrix} b_{11} & b_{12} & ... & b_{1n}\\ b_{21} & ... & ... & ...\\ ... & ... & ... & ...\\ b_{n1} & ... & ... & b_{nn} \end{pmatrix}\]](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bpmatrix%7D%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%20...%20%26%20a_%7B1n%7D%5C%5C%20a_%7B21%7D%20%26%20...%20%26%20...%20%26%20...%5C%5C%20...%20%26%20...%20%26%20...%20%26%20...%5C%5C%20a_%7Bn1%7D%20%26%20...%20%26%20...%20%26%20a_%7Bnn%7D%20%5Cend%7Bpmatrix%7D%20%5Cbegin%7Bpmatrix%7D%20b_%7B11%7D%20%26%20b_%7B12%7D%20%26%20...%20%26%20b_%7B1n%7D%5C%5C%20b_%7B21%7D%20%26%20...%20%26%20...%20%26%20...%5C%5C%20...%20%26%20...%20%26%20...%20%26%20...%5C%5C%20b_%7Bn1%7D%20%26%20...%20%26%20...%20%26%20b_%7Bnn%7D%20%5Cend%7Bpmatrix%7D)

From that multiplication, we can extract one step as an example \
![\[(ab)_{23} = a_{21}b_{13} + a_{22}b_{23} + ... + a_{2n}b_{n3}\]](https://latex.codecogs.com/gif.latex?%28ab%29_%7B23%7D%20%3D%20a_%7B21%7Db_%7B13%7D%20&plus;%20a_%7B22%7Db_%7B23%7D%20&plus;%20...%20&plus;%20a_%7B2n%7Db_%7Bn3%7D)

Then, we can deduct the Einstein summation of the matrix multiplication \
![\[\sum_{j} a_{ij}b_{jk} = a_{ij}b_{jk}\]](https://latex.codecogs.com/gif.latex?%5Csum_%7Bj%7D%20a_%7Bij%7Db_%7Bjk%7D%20%3D%20a_%7Bij%7Db_%7Bjk%7D)

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 0, 1]])
B = np.array([[1, 1, 0],[0, 1, 1], [1, 0, 1]])
np.dot(A, B)
```


## Transposed matrix
![\[b_{ij} = a_{ji}\]](https://latex.codecogs.com/gif.latex?b_%7Bij%7D%20%3D%20a_%7Bji%7D)

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def my_transpose(A):
    (n, m) = A.shape

    B = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i, j]

    return B

print("not transposed: ", A)
print("transposed: ", np.array_equal(my_transpose(A), np.transpose(A)))
```

![\[A^{T}A = I\]](https://latex.codecogs.com/gif.latex?A%5E%7BT%7DA%20%3D%20I)

