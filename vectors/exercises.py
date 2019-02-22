import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import math

print("\nvalues")
r = np.array([1,2])
s = np.array([3,4])
t = np.array([5,6])
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