import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
G = np.array([[1, -1], [-1, 3]])
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


A = np.array([[5, 2, 0], [2, -7, 0], [2, 3, -5]])
B = np.array([[7/40, 1/16, 0], [1/16, -5/32, 0], [1/8, -9/80, -1/5]])

print("verification: ", np.linalg.inv(A))
print("calculated: ", B)
print("determinant: ", np.linalg.det(G))

A = np.array([[1, 2, 3], [4, 0, 1]])
B = np.array([[1, 1, 0],[0, 1, 1], [1, 0, 1]])
print("multiply matrix: ", np.dot(A, B))


def mirror_transformation_example():
    plt.style.use('ggplot')
    e1 = np.array([0, 0, 0, 1, 0, 0])
    r  = np.array([0, 0, 0, 2, 3, 5])
    origin=np.array([0, 0, 0])

    all = np.array([e1, r])

    X, Y, Z, U, V, W = zip(*all)
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    print("U:", U)
    print("V:", V)
    print("W:", W)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, colors=[(0.2, 0.5, 0.2), (0.8, 0.2, 0.2)], lw=2)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    plt.show()


def transformation():
    # let's a vector r
    r = np.array([2, 3, 5])

    # an origin basis
    basis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# transformation matrix
T = np.array([[0.9, 0.8], [-1, 0.35]])
v0 = np.array([0.5, 1])

def draw(vectors):
    # colors
    blue1 = (0.27450980392156865, 0.396078431372549, 0.5372549019607843)
    magenta = (0.9882352941176471, 0.4588235294117647, 0.8588235294117647)
    green = (0.6862745098039216, 0.8588235294117647, 0.5215686274509804)
    orange = (0.8549019607843137, 0.6705882352941176, 0.45098039215686275)
    blue2 = (0.47843137254901963, 0.6823529411764706, 0.8431372549019608)
    bear_white = (0.89, 0.856, 0.856)
    bear_black = (0.141, 0.11, 0.11)
    lightgrey = (220/255, 218/255, 212/255)

    # limits
    xmin, xmax, ymin, ymax = (-10, 10, -10, 10)

    # graph
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(1)
    plt.axis('off')

    # origin axes
    ax.arrow(xmin, 0, -xmin+xmax, 0, lw=2, color=lightgrey, zorder=1, head_width=0.1)
    ax.arrow(0, ymin, 0, -ymin+ymax, lw=2, color=lightgrey, zorder=2, head_width=0.1)

    # draw vectors
    for v in vectors:
        ax.arrow(0, 0, v[0], v[1], lw=3, color=magenta, zorder=5, head_width=0.1)


    xs = [v[0] for v in vectors]
    ys = [v[1] for v in vectors]
    ax.plot(xs, ys, color=orange, label='line 1', lw=2)

    return ax

def n_transform(v, T, n):
    result=[v]

    for i in range(1, n+1):
        print("i:", i)
        print("result[i-1]: ", result[i-1])
        result.append(T @ result[i-1])

    return result


# vs = n_transform(v0, T, 0)

# draw(vs)
# plt.show()
L=[[1, 1], [0, 2]]
eVals, eVecs =  np.linalg.eig(L)
# -1 means reverse list
order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues

print("eigenvalues: ",eVals)
print("eigenvectors: ",eVecs)
print("order: ", order)
print("eigenvalues ordered: ",eVals[order])
print("eigenvectors ordered: ",eVecs[order])


# Replace the ??? here with the probability of clicking a link to each website when leaving Website F (FaceSpace).
L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]])


# initialize the r vector
r = 100 * np.ones(L.shape[0]) / L.shape[0]
print("r initialized: ", r)

# Apply the transformation
r_1 = L @ r
print("r after one transformation applyed: ", r_1)

 # We'll call this one L2, to distinguish it from the previous L.
L2 = np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])

def transfo_n(L):
    r = 100 * np.ones(L.shape[0]) / L.shape[0]

    lastR = r
    r = L @ r
    i = 0
    # Apply a transformation T n times until the r stay unchanged
    while np.linalg.norm(lastR - r) > 0.01:
        lastR = r
        r = L @ r
        i = i + 1

    return np.round(r, 2)


# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)
def pageRank(linkMatrix, d):
    n = linkMatrix.shape[0]

    # init r
    r = 100 * np.ones(n) / n

    # init the lastR
    lastR = r

    # build the new link matrix with the damping parameter
    M = d * linkMatrix + (1 - d) / n * np.ones([n, n])

    # first iteration
    r = M @ r

    while np.linalg.norm(lastR - r) > 0.01:
        lastR = r
        r = M @ r

    return r

def eigens(M):
    np.set_printoptions(suppress=True, linewidth=400)
    vals, vecs = np.linalg.eig(M)

    print("vals: \n", vals)
    print("vecs: \n", vecs)

def my_transpose(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            B[j, i] = A[i, j]

    return B

def draw(vectors, transposed):
    # colors
    blue1 = (0.27450980392156865, 0.396078431372549, 0.5372549019607843)
    magenta = (0.9882352941176471, 0.4588235294117647, 0.8588235294117647)
    green = (0.6862745098039216, 0.8588235294117647, 0.5215686274509804)
    orange = (0.8549019607843137, 0.6705882352941176, 0.45098039215686275)
    blue2 = (0.47843137254901963, 0.6823529411764706, 0.8431372549019608)
    bear_white = (0.89, 0.856, 0.856)
    bear_black = (0.141, 0.11, 0.11)
    lightgrey = (220/255, 218/255, 212/255)

    # limits
    xmin, xmax, ymin, ymax = (-10, 10, -10, 10)

    # graph
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(1)
    plt.axis('off')

    # origin axes
    ax.arrow(xmin, 0, -xmin+xmax, 0, lw=2, color=lightgrey, zorder=1, head_width=0.1)
    ax.arrow(0, ymin, 0, -ymin+ymax, lw=2, color=lightgrey, zorder=2, head_width=0.1)

    # draw vectors
    for v in vectors:
        ax.arrow(0, 0, v[0], v[1], lw=3, color=magenta, zorder=5, head_width=0.1,label='original')

    for r in transposed:
        ax.arrow(0, 0, r[0], r[1], lw=3, color=orange, zorder=5, head_width=0.1,label='transposed')

    ax.legend( handles=[ax.arrow, ], loc='upper left' )

    return ax

A=np.array([[1,3], [4, 2]])

print("not transposed: ", A)
print("transposed: ", np.transpose(A))

draw(A, np.transpose(A))
plt.show()