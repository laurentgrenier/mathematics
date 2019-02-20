import numpy as np

print("\nvalues")
r = np.array([[1], [2]])
s = np.array([[3], [4]])
t = np.array([[5], [6]])
print("\tr=", r)
print("\ts=", s)

print("\naddition")
print("\nassociativity: ", np.array_equal(r+(s+t), r+(s+t)))

print("\nmultiplication")
print("\nby a scalar: ", 2*r)


print("\nModulus and inner product")
print("\nModulus: ", np.linalg.norm(r))

