import math
import numpy as np

print("\nvalues")
m = np.array([[2, 3], [10, 1]])
A = np.array([[3, 2], [2, 3]])
r = np.array([1, 2])
s = np.array([-2, -2])
x1 = 3
x2 = 2
r_prime = np.array([7, 8])

e1 = np.array([1, 0])
e2 = np.array([0, 1])

print("\tm=", m)
print("\te1=", e1)
print("\te2=", e2)

print("\nMatrices")
print("multiply by a basis vector: ", np.array_equal(m.dot(e1), np.array([2, 10])))

print("\nHow matrices transform space")
print("true equation: ", np.array_equal(np.dot(A, r), r_prime))
print("rule 1: ", np.array_equal(np.dot(A, x1 * r), x1 * r_prime))
print("rule 2: ", np.array_equal(np.dot(A, r + s), np.dot(A, r) + np.dot(A, s)))
print("rule 3: ", np.array_equal(np.dot(A, x1 * e1 + x2 * e2), x1 * np.dot(A, e1) + x2 * np.dot(A, e2)))

