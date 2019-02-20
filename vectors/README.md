# vectors
Basics about vectors

> Each formulas URLs are generated using the website https://www.codecogs.com/latex/eqneditor.php  
## Operations with vectors
### Addition
![\[\begin{bmatrix} r_{1} \\ r_{2} \end{bmatrix} + \begin{bmatrix} s_{1} \\ s_{2} \end{bmatrix} = \begin{bmatrix} r_{1} + s_{1} \\ r_{2} + s_{2} \end{bmatrix}\]
](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20%5C%5C%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Cbegin%7Bbmatrix%7D%20s_%7B1%7D%20%5C%5C%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20&plus;%20s_%7B1%7D%20%5C%5C%20r_%7B2%7D%20&plus;%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D)  


```python
import numpy as np
r = np.array([[1], [2]])
s = np.array([[3], [4]])
sum = r + s
```

#### Associativity
(r+s)+t = r+(s+t)

```python
import numpy as np
r = np.array([[1], [2]])
s = np.array([[3], [4]])
t = np.array([[5], [6]])
np.array_equal(r+(s+t), r+(s+t))
```


### Multiplication
#### By a scalar

![\[2\times \begin{bmatrix} r_{1} \\ r_{2} \end{bmatrix} = \begin{bmatrix} 2\times r_{1}\\ 2\times r_{2} \end{bmatrix}\]
](https://latex.codecogs.com/gif.latex?2%5Ctimes%20%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20%5C%5C%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%202%5Ctimes%20r_%7B1%7D%5C%5C%202%5Ctimes%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D)  

## Modulus and inner product

### Modulus
The scalar measure of a vector.

Let's 
![\[r = \begin{bmatrix} a \\ b \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?r%20%3D%20%5Cbegin%7Bbmatrix%7D%20a%20%5C%5C%20b%20%5Cend%7Bbmatrix%7D)

![\left | r \right | = \sqrt{a^{2} + b^{2}}](https://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20r%20%5Cright%20%7C%20%3D%20%5Csqrt%7Ba%5E%7B2%7D%20&plus;%20b%5E%7B2%7D%7D)

```python
import numpy as np
r = np.array([[1], [2]])

np.linalg.norm(r)
```

### Dot product
Let's ![\[r = \begin{bmatrix} r_{1} \\ r_{2} \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?r%20%3D%20%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20%5C%5C%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D) 
and ![\[s = \begin{bmatrix} s_{1} \\ s_{2} \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?s%20%3D%20%5Cbegin%7Bbmatrix%7D%20s_%7B1%7D%20%5C%5C%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D)

![\[r.s = r_{1}s_{1} \times r_{2}s_{2}\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20r_%7B1%7Ds_%7B1%7D%20%5Ctimes%20r_%7B2%7Ds_%7B2%7D)