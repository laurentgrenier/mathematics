import math
import numpy as np

print("\nvalues")
m = np.array([[2, 3], [10, 1]])
A = np.array([[3, 2], [2, 3]])
r = np.array([1, 2])
s = np.array([-2, -2])
identity = np.array([[1, 0], [0, 1]])
upscaling = np.array([[3, 0], [0, 2]])
downscaling = np.array([[1/3, 0], [0, 1/2]])
invert = np.array([[-1, 0], [0, -1]])


x1 = 3
x2 = 2
r_prime = np.array([7, 8])

e1 = np.array([1, 0])
e2 = np.array([0, 1])

mirror_a = np.array([[0, 1], [1, 0]])
mirror_b = np.array([[0, -1], [-1, 0]])
mirror_c = np.array([[-1, 0], [0, 1]])
mirror_d = np.array([[1, 0], [0, -1]])
shear = np.array([[1, 0], [1, 1]])

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

print("\nTypes of matrix transformation")
print("Identity matrix: ", np.array_equal(np.dot(identity, r), r))
print("Upscaling matrix: ", np.array_equal(np.greater(np.dot(upscaling, r), r), [True, True]))
print("Downscaling matrix: ", np.array_equal(np.less(np.dot(downscaling, r), r), [True, True]))
print("Inversion matrix: ", np.array_equal(np.dot(np.dot(invert, r), invert), r))

print("mirror matrices a: ", np.dot(mirror_a, e1))
print("mirror matrices b: ", np.dot(mirror_d, e1))
print("mirror matrices c: ", np.dot(mirror_c, e1))
print("mirror matrices d: ", np.dot(mirror_d, e1))

print("shears matrix: ", np.dot(shear, e1))


def rotation_matrix(angle):
    return [[round(np.cos(angle), 3), round(np.sin(angle), 3)], [-round(np.sin(angle), 3), round(np.cos(angle), 3)]]


print("rotation matrix: ", np.array_equal(np.dot(rotation_matrix(-3*np.pi/2), np.dot(rotation_matrix(-np.pi/2), e1)), e1))
