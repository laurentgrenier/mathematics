import math
import numpy as np
import matplotlib.pyplot as plt


print("\nvalues")
r = np.array([1, 2])
s = np.array([3, 4])
t = np.array([5, 6])
v1 = np.array([1, 2])
v2 = np.array([-2, 1])

vectors = np.array([r, s, t, v1, v2])
origin = [0], [0]
plt.quiver(*origin, vectors[:, 0], vectors[:, 1], color=['b', 'g', 'r', 'c', 'm'], scale=21)
plt.show()

teta = math.acos(np.dot(r,s) / (np.linalg.norm(r)*np.linalg.norm(s)))

a = 2


print("\tr=", r)
print("\ts=", s)

print("\naddition")
print("\nassociativity: ", np.array_equal(r+(s+t), r+(s+t)))

print("\nmultiplication")
print("\nby a scalar: ", 2*r)


print("\nModulus and inner product")
print("\nModulus: ", np.linalg.norm(r))

print("\nDot product: ", np.dot(r,s))
print("commutativity: ", np.dot(r,s) == np.dot(s,r))
print("distributivity: ", np.dot(r,s+t) == np.dot(r,s)+np.dot(r,t))
print("associativity over scalar multiplication: ", np.dot(r, a * s) == a*np.dot(r, s))

print("\nCosine rules")
print("first rule: ", round(np.linalg.norm(r-s)**2, 3) ==
      round(np.linalg.norm(r)**2 + np.linalg.norm(s)**2 - 2*np.linalg.norm(r)*np.linalg.norm(s)*math.cos(teta),3))
print("second rule: ", round(float(np.dot(r, s)), 3) == round(np.linalg.norm(r)*np.linalg.norm(s)*math.cos(teta),3))

print("\nVectors projections")
print("Scalar projection: ", round(float(np.dot(r, s))/np.linalg.norm(r), 3) ==
    round(np.linalg.norm(s)*math.cos(teta), 3))
print("Vector projection: ", round(float(np.dot(r, s)/(np.linalg.norm(r)*np.linalg.norm(r))), 3) ==
   round(float(np.dot(r, s)/np.dot(r, r)),3))

print("\nBasis vectors")
print("unit length vector: ", [1,0])
print("r and s are linearly dependent: ", np.dot(r, s) != 0)
print("v1 and v2 are linearly independent: ", np.dot(v1, v2) == 0)

