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

