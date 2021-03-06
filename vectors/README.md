# vectors
Basics about vectors

> Each formulas URLs are generated using the website https://www.codecogs.com/latex/eqneditor.php  
## Operations with vectors
### Addition
![\[\begin{bmatrix} r_{1} \\ r_{2} \end{bmatrix} + \begin{bmatrix} s_{1} \\ s_{2} \end{bmatrix} = \begin{bmatrix} r_{1} + s_{1} \\ r_{2} + s_{2} \end{bmatrix}\]
](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20%5C%5C%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Cbegin%7Bbmatrix%7D%20s_%7B1%7D%20%5C%5C%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20&plus;%20s_%7B1%7D%20%5C%5C%20r_%7B2%7D%20&plus;%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D)  


```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])
sum = r + s
```

#### Associativity
(r+s)+t = r+(s+t)

```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])
t = np.array([5,6])
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
r = np.array([1,2])

np.linalg.norm(r)
```

### Dot product
Let's ![\[r = \begin{bmatrix} r_{1} \\ r_{2} \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?r%20%3D%20%5Cbegin%7Bbmatrix%7D%20r_%7B1%7D%20%5C%5C%20r_%7B2%7D%20%5Cend%7Bbmatrix%7D) 
and ![\[s = \begin{bmatrix} s_{1} \\ s_{2} \end{bmatrix}\]](https://latex.codecogs.com/gif.latex?s%20%3D%20%5Cbegin%7Bbmatrix%7D%20s_%7B1%7D%20%5C%5C%20s_%7B2%7D%20%5Cend%7Bbmatrix%7D)

![\[\[r.s = r_{1}s_{1} + r_{2}s_{2}\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20r_%7B1%7Ds_%7B1%7D%20&plus;%20r_%7B2%7Ds_%7B2%7D)

```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])

np.dot(r,s)
```
#### Commutativity
![\[r.s = s.r\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20s.r)

```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])

assert np.dot(r,s) == np.dot(s,r)
```

#### distributivity
![\[r.(s+t) = r.s + r.t\]](https://latex.codecogs.com/gif.latex?r.%28s&plus;t%29%20%3D%20r.s%20&plus;%20r.t)

```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])
t = np.array([5,6])

assert np.dot(r,s+t) == np.dot(r,s)+np.dot(r,t)
```

#### associativity over scalar multiplication
![\[r.(as) = a(r.s)\]](https://latex.codecogs.com/gif.latex?r.%28as%29%20%3D%20a%28r.s%29)


```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])
a = 2

assert np.dot(r, a * s) == a*np.dot(r, s)
```

#### cosine rule
![mathematics_vectors_001](images/mathematics_vectors_001.png "Vectors")

![\[c^{2} = a^{2} + b^{2} - 2ab.cos\theta\]](https://latex.codecogs.com/gif.latex?c%5E%7B2%7D%20%3D%20a%5E%7B2%7D%20&plus;%20b%5E%7B2%7D%20-%202ab.cos%5Ctheta)

![\[\left |r \right |\]](https://latex.codecogs.com/gif.latex?%5Cleft%20%7Cr%20%5Cright%20%7C) is the **modulus**. It's a number. \
**r** is a vector

![\[\left |r-s \right |^{2} = \left |r \right |^{2} + \left |s \right |^{2} - 2 \left |r \right |\left |s \right |.cos\theta\]](https://latex.codecogs.com/gif.latex?%5Cleft%20%7Cr-s%20%5Cright%20%7C%5E%7B2%7D%20%3D%20%5Cleft%20%7Cr%20%5Cright%20%7C%5E%7B2%7D%20&plus;%20%5Cleft%20%7Cs%20%5Cright%20%7C%5E%7B2%7D%20-%202%20%5Cleft%20%7Cr%20%5Cright%20%7C%5Cleft%20%7Cs%20%5Cright%20%7C.cos%5Ctheta)

```python
import numpy as np
import math
r = np.array([1,2])
s = np.array([3,4])
teta = math.acos(np.dot(r,s) / (np.linalg.norm(r)*np.linalg.norm(s)))
round(np.linalg.norm(r-s)**2, 3) == round(np.linalg.norm(r)**2 + np.linalg.norm(s)**2 - 2*np.linalg.norm(r)*np.linalg.norm(s)*math.cos(teta),3)
```


![\[r.s = \left |r \right |\left |s \right |cos(\theta )\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20%5Cleft%20%7Cr%20%5Cright%20%7C%5Cleft%20%7Cs%20%5Cright%20%7Ccos%28%5Ctheta%20%29)

```python
import numpy as np
import math
r = np.array([1,2])
s = np.array([3,4])
teta = math.acos(np.dot(r,s) / (np.linalg.norm(r)*np.linalg.norm(s)))
round(float(np.dot(r, s)), 3) == round(np.linalg.norm(r)*np.linalg.norm(s)*math.cos(teta),3)
```
##### Specific cases
If **r** and **s** are orthogonal, **r.s = 0** \
If **r** and **s** are going into the same direction, ![\[r.s = \left |r \right |\left |s \right |\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20%5Cleft%20%7Cr%20%5Cright%20%7C%5Cleft%20%7Cs%20%5Cright%20%7C) \
If **r** and **s** are going into opposite directions, ![\[r.s = - \left |r \right |\left |s \right |\]](https://latex.codecogs.com/gif.latex?r.s%20%3D%20-%20%5Cleft%20%7Cr%20%5Cright%20%7C%5Cleft%20%7Cs%20%5Cright%20%7C) \

### Vectors projection

![mathematics_vectors_002](images/mathematics_vectors_002.png "Vector projection")

#### Scalar projection
![\[\frac{r.s}{\left | r \right |} = \left | s \right | \times cos(\Theta)\]](https://latex.codecogs.com/gif.latex?%5Cfrac%7Br.s%7D%7B%5Cleft%20%7C%20r%20%5Cright%20%7C%7D%20%3D%20%5Cleft%20%7C%20s%20%5Cright%20%7C%20%5Ctimes%20cos%28%5CTheta%29)

```python
import numpy as np
import math
r = np.array([1,2])
s = np.array([3,4])
teta = math.acos(np.dot(r,s) / (np.linalg.norm(r)*np.linalg.norm(s)))
round(float(np.dot(r, s))/np.linalg.norm(r), 3) == round(np.linalg.norm(s)*math.cos(teta), 3)
```

#### Vector projection
![\[\frac{r.s}{\left | r \right |\left | r \right |} = \frac{r.s}{r.r}\]](https://latex.codecogs.com/gif.latex?%5Cfrac%7Br.s%7D%7B%5Cleft%20%7C%20r%20%5Cright%20%7C%5Cleft%20%7C%20r%20%5Cright%20%7C%7D%20%3D%20%5Cfrac%7Br.s%7D%7Br.r%7D)

```python
import numpy as np
r = np.array([1,2])
s = np.array([3,4])
round(float(np.dot(r, s)/(np.linalg.norm(r)*np.linalg.norm(r))), 3) == round(float(np.dot(r, s)/np.dot(r, r)),3)
```

## Basis vectors

### notation 

![\[\hat{e_{1}}\]](https://latex.codecogs.com/gif.latex?%5Chat%7Be_%7B1%7D%7D)
the _hat_ means that the vector is of unit length.

### orthogonal vectors
**r** and **s** are orthogonal if the dot product **r.s = 0**. \
That means that **r** doesn't exist in the basis **s**.

> Basis vectors is a set of vectors that: 
> * are not linear combination of each other
> * span the space. The space is then n-dimensional
