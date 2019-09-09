import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
iris = datasets.load_iris()
print('data shape is {}'.format(iris.data.shape))
print('class shape is {}'.format(iris.target.shape))
import numpy as np
import scipy


x0 = np.array([[1,0], [1,0]])
x1 = np.array([[0,1], [0,1]])


X = iris.data[:, :2] # use first two version for simplicity
y = iris.target

def display_iris():
    iris = datasets.load_iris()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000',  '#00FF00', '#0000FF'])

    K = 3
    x = X[-1]

    fig, ax = plt.subplots(figsize=(4,4))
    for i, iris_class in enumerate(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']):
        idx = y==i
        ax.scatter(X[idx,0], X[idx,1],
                   c=cmap_bold.colors[i], edgecolor='k',
                   s=20, label=iris_class);
    ax.set(xlabel='sepal length (cm)', ylabel='sepal width (cm)')
    ax.legend();

def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product"""
    distance = np.sqrt((x0 - x1) @ (x0 - x1)) # <-- EDIT THIS to compute the distance between x0 and x1
    return distance

def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product"""
    angle = np.arccos((x0 @ x1) / np.sqrt((x0 @ x0) * (x1 @ x1))) # <-- EDIT THIS to compute angle between x0 and x1
    return angle


def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """
    N, D = X.shape
    M, _ = Y.shape

    scipy_distance_matrix = scipy.spatial.distance_matrix(X, Y)

    loop_distance_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            loop_distance_matrix[i,j] = np.sqrt((X[i] - Y[j]) @ (X[i] - Y[j]))

    custom_distance_matrix = np.zeros((N,M))
    W = X.T - Y.T
    print(W)
    custom_distance_matrix = np.sqrt(W.T @ W)

    print("scipy_distance_matrix: ", scipy_distance_matrix)
    print("loop_distance_matrix: ", loop_distance_matrix)
    print("custom_distance_matrix: ", custom_distance_matrix)

#    distance_matrix = distance  # <-- EDIT THIS to compute the correct distance matrix.
    return loop_distance_matrix


def KNN(k, X, y, x):
    """K nearest neighbors
    k: number of nearest neighbors
    X: training input locations
    y: training labels
    x: test input
    """
    N, D = X.shape

    # number of predictable unique values
    num_classes = len(np.unique(y))

    # distance from the given point to each points of the traning set
    dist = scipy.spatial.distance_matrix(X, x)

    print("dist: ", dist)

    # Next we make the predictions
    # initialize a vector with a value for each class
    ypred = np.zeros(num_classes)

    print("num_classes:" , num_classes)

    # the K nearest neighbors
    knn = y[np.argsort(dist)][:k]
    print("knn:", knn)
    classes = knn  # find the labels of the k nearest neighbors
    print("y: ", y)
    print("classes: ", classes)
    for c in np.unique(classes):
        ypred[c] = (classes == c).sum()  # <-- EDIT THIS to compute the correct prediction


    return np.argmax(ypred)

def test_on_iris():
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets
    iris = datasets.load_iris()
    print('data shape is {}'.format(iris.data.shape))
    print('class shape is {}'.format(iris.target.shape))

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000',  '#00FF00', '#0000FF'])

    # we keep the two nearest neighbors
    K = 2

    # keep only the 4th first values
    X = iris.data[:90, :2] # use first two version for simplicity
    y = iris.target[:90]


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    step = 0.1

    # create grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    # init the test output vector
    ypred = []

    # create a matrix of points
    grid_points = np.array([xx.ravel(), yy.ravel()]).T

    for point in grid_points:
        # because each row is a point, we need to reshape each point from (2,1) to (1,2)
        ypred.append(KNN(K, X, y, point.reshape(1,2)))

    fig, ax = plt.subplots(figsize=(4,4))

    print(len(ypred))

    ax.pcolormesh(xx, yy, np.array(ypred).reshape(xx.shape), cmap=cmap_light)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.show()


def projection_matrix_1d(b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        b: ndarray of dimension (D, 1), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    D, _ = b.shape
    ### Edit the code below to compute a projection matrix of shape (D,D)
    P = (b @ b.T) / np.sqrt(b.T @ b)**2 # <-- EDIT THIS
    return P
    ###


# ===YOU SHOULD EDIT THIS FUNCTION===
def project_1d(x, b):
    """Compute the projection matrix onto the space spanned by `b`
    Args:
        x: the vector to be projected
        b: ndarray of dimension (D, 1), the basis for the subspace

    Returns:
        y: ndarray of shape (D, 1) projection of x in space spanned by b
    """
    p = projection_matrix_1d(b) @ x  # <-- EDIT THIS
    return p


# Projection onto a general (higher-dimensional) subspace
# ===YOU SHOULD EDIT THIS FUNCTION===
def projection_matrix_general(B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    P = B @ np.linalg.inv(B.T @ B) @ B.T  # <-- EDIT THIS
    return P


# ===YOU SHOULD EDIT THIS FUNCTION===
def project_general(x, B):
    """Compute the projection matrix onto the space spanned by the columns of `B`
    Args:
        x: ndarray of dimension (D, 1), the vector to be projected
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        p: projection of x onto the subspac spanned by the columns of B; size (D, 1)
    """
    p = projection_matrix_general(B) @ x  # <-- EDIT THIS
    return p

def test_proj_1D():
    b = np.array([2, 1]).reshape(-1,1)
    x = np.array([1, 2]).reshape(-1,1)
    print(project_1d(x, b))


def test_proj_2D():
    B = np.array([[0,0], [1,2], [2,2]])
    x = np.array([4,2,1])
    print(projection_matrix_general(B))

test_proj_2D()

#test_proj_2D()
