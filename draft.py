import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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


def test_zip():
    plt.style.use('ggplot')
    e1 = np.array([1, 0, 0])
    r  = np.array([2, 3, 5])
    origin=np.array([0, 0, 0])

    print("concatenate :", e1)
    all = np.array([e1, r])

    X, Y, Z = zip(origin, e1, r)
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    print("U:", U)
    print("V:", V)
    print("W:", W)


    test1=[1, 2, 3]
    test2=[4, 5, 6]
    test3=[7, 8, 9]

    A, B, C=zip(test1, test2, test3)
    print(A, B, C)




A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def my_transpose(A):
    (n, m) = A.shape

    B = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i, j]

    return B

print("not transposed: ", A)
print("transposed: ", np.array_equal(my_transpose(A), np.transpose(A)))




def test_meshgrid_001():
    xs, ys = np.meshgrid(np.arange(0,6,0.05), np.arange(0,6,0.05))
    zs = 2 * xs ** 2 * ys
    h = plt.contourf(xs, ys, zs)
    plt.show()

def test_meshgrid_002():
    xs, ys = np.meshgrid(np.arange(-4,4,0.001), np.arange(-4,4,0.001))
    zs = 3 * (1 - xs) ** 2 * np.exp(-xs ** 2 - (ys + 1) ** 2)
    - 10 * (xs / 5 - xs ** 3 - ys ** 5) * np.exp(-xs ** 2 - ys ** 2)
    - 1/3 * np.exp(-(xs + 1) ** 2 - ys ** 2)

    h = plt.contourf(xs, ys, zs)
    plt.show()


def test_3d_plot():
    X, Y = np.meshgrid(np.arange(-4, 4, 0.001), np.arange(-4, 4, 0.001))
    Z = 3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2)
    - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2)
    - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



def neural_output_by_hand():
    # First set up the network.
    sigma = np.tanh
    W = np.array([[-2, 4, -1], [6, 0, -3]])
    b = np.array([0.1, -2.5])

    # Define our input vector
    x = np.array([0.3, 0.4, 0.1])

    # Calculate the values by hand,
    # and replace a1_0 and a1_1 here (to 2 decimal places)
    # (Or if you feel adventurous, find the values with code!)

    a1_0, a1_1 = sigma(W @ x + b)

    a1 = np.array([a1_0, a1_1])

    return a1

sigma = lambda z: 1 / (1 + np.exp(-z))

def plot_logistic_function():
    values_range = 10
    xs = np.array([x-int(values_range / 2) for x in range(values_range+1)])
    ys = sigma(xs)
    plt.plot(xs, ys)
    plt.show()

def random_view():
    print("random: ", np.random.randn(6, 1))

random_view()


import numpy as np
import scipy.misc

g0 = lambda x: 1
g2 = lambda x: 1 - np.power(x,2) / scipy.special.factorial(2)
g4 = lambda x: 1 - np.power(x,2) / scipy.special.factorial(2) + np.power(x,4) / scipy.special.factorial(4)

gn = lambda nx: np.array([np.power(-1,(n % 4) / 2) * np.power(nx[1],n) / scipy.special.factorial(n) for n in np.arange(0, nx[0]+1, 2)]).sum()
print("gn: ", gn((0,3)))
print("gn: ", gn((1,3)))
print("gn: ", gn((2,3)))
print("gn: ", gn((3,3)))
print("gn: ", gn((4,3)))
print(4%4)

