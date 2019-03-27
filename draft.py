import numpy as np
import numpy.linalg as la
r = np.array([1, 2, 3])
s = np.array([4, 5, 6])


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B_matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

print(A.shape)
print(A[:, 1].shape)
print(r.shape)
print(A.shape)

verySmallNumber = 1e-14
print(float(verySmallNumber))

def gsBasis4(A):
    B = np.array(A, dtype=np.float_)  # Make B as a copy of A, since we're going to alter it's values.
    # The zeroth column is easy, since it has no other vectors to make it normal to.
    # All that needs to be done is to normalise it. I.e. divide by its modulus, or norm.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # For the first column, we need to subtract any overlap with our new zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])
    # Now we need to repeat the process for column 2.
    # Insert two lines of code, the first to subtract the overlap with the zeroth vector,
    # and the second to subtract the overlap with the first.
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]

    # Again we'll need to normalise our new vector.
    # Copy and adapt the normalisation fragment from above to column 2.
    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])

    # Finally, column three:
    # Insert code to subtract the overlap with the first three vectors.
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]

    # Now normalise if possible
    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    # Finally, we return the result:
    return B


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    # for each columns of B
    for i in range(B.shape[1]):
        # substract the overlap from the previous columns
        for j in range(i):
            # substract the overlap
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        # Now normalise if possible
        if la.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])

    return B

V = np.array([[1,2,3],
              [1,0,1],
              [1,1,-1]], dtype=np.float_)

V_rebased = gsBasis(V)
print(V_rebased)

print(1/np.sqrt(6))

# GRADED FUNCTION
# This is the cell you should edit and submit.

# In this function, you will return the transformation matrix T,
# having built it out of an orthonormal basis set E that you create from Bear's Basis
# and a transformation matrix in the mirror's coordinates TE.
def build_reflection_matrix(bearBasis) : # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # Use the gsBasis function on bearBasis to get the mirror's orthonormal basis.
    E = gsBasis(bearBasis)
    # Write a matrix in component form that perform's the mirror's reflection in the mirror's basis.
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0],
                   [0, -1]])
    # Combine the matrices E and TE to produce your transformation matrix.
    T =  E @ TE
    # Finally, we return the result. There is no need to change this line.
    return T


def build_reflection_matrix_3(bearBasis) : # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # Use the gsBasis function on bearBasis to get the mirror's orthonormal basis.
    E = gsBasis(bearBasis)
    # Write a matrix in component form that perform's the mirror's reflection in the mirror's basis.
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, -1]])
    # Combine the matrices E and TE to produce your transformation matrix.
    T =  E @ TE
    # Finally, we return the result. There is no need to change this line.
    return T

print("reflexion matrix: ", build_reflection_matrix_3(V))