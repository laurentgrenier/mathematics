import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import math

print("\nvalues")
r = np.array([1,2])
s = np.array([3,4])
t = np.array([5,6])


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

vectors = np.array([r, s])
origin = [0], [0] # origin point
plt.plot([])
plt.quiver(*origin, vectors[:,0], vectors[:,1], color=['r', 'b', 'g'], scale=21)
plt.show()