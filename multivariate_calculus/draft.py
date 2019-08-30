import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

# set function
f = lambda x,y:-((x-4.1)**2 + (y-0.9)**2) - 30*np.exp(-((x-1.5)**2 + (y-3.7)**2)/(2*1.4))


def display_a_contour_plot(f):
    # create the figure
    fig, ax = plt.subplots()

    # set axes
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])
    ax.set_aspect(1)

    # draw the contours
    X, Y = np.meshgrid(np.arange(0, 6, 0.05), np.arange(0, 6, 0.05))
    ax.contour(X, Y, f(X, Y), 10)

    # display
    plt.show()


def display_in_3D(f):
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    # Make data.
    X, Y = np.meshgrid(np.arange(0, 6, 0.05), np.arange(0, 6, 0.05))
    Z = f(X,Y)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

def parabolic():
    f = lambda x: -2*x**2 + 10*x -1

    X=np.arange(0,6,0.1)
    Xprime = np.arange(4, 5, 0.1)

    Y=f(X)

    fprime = lambda x:-4*x + 10
    Yprimes = fprime(Xprime)

    bounds = [(0,6)]
    x0 = differential_evolution(f, bounds)
    print("x0: ", x0.fun)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(X,Y)
    ax.plot(Xprime, Yprimes)
    plt.show()

# x0 = differential_evolution(lambda xs: f(xs[0], xs[1]), ((0,6),(0,6))).x


def test_yabox():
    from yabox.problems import Levy
    problem = Levy()
    problem.plot3d()

def plot_3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    f = lambda x,y: np.exp(-(x**2 +y**2))
    f_prime_x = lambda x,y: -2 * x * np.exp(-(x**2 +y**2))
    f_prime_y = lambda x,y: -2 * y * np.exp(-(x**2 +y**2))

    # values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Z
    Z = f(X, Y)

    # Points
    p_1 = (-1, 1, 0)
    p_2 = (2, 2, 0)
    p_3 = (0, 0, 0)

    # Jacobians
    J_1 = [f_prime_x(p_1[0],p_1[1]), f_prime_y(p_1[0],p_1[1])]
    J_2 = [f_prime_x(p_2[0], p_2[1]), f_prime_y(p_2[0], p_2[1])]
    J_3 = [f_prime_x(p_3[0], p_3[1]), f_prime_y(p_3[0], p_3[1])]

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # drawing the function
    ax.plot_wireframe(X, Y, Z, color='grey')

    # drawing the Jacobians
    ax.quiver(p_1[0], p_1[1],p_1[2], J_1[0], J_1[1], 0)

    plt.show(block=True)
    plt.interactive(False)


def animated_graph():
    import math

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation


    def beta_pdf(x, a, b):
        return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
                / (math.gamma(a) * math.gamma(b)))


    class UpdateDist(object):
        def __init__(self, ax, prob=0.5):
            self.success = 0
            self.prob = prob
            self.line, = ax.plot([], [], 'k-')
            self.x = np.linspace(0, 1, 200)
            self.ax = ax

            # Set up plot parameters
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 15)
            self.ax.grid(True)

            # This vertical line represents the theoretical value, to
            # which the plotted distribution should converge.
            self.ax.axvline(prob, linestyle='--', color='black')

        def init(self):
            self.success = 0
            self.line.set_data([], [])
            return self.line,

        def __call__(self, i):
            # This way the plot can continuously run and we just keep
            # watching new realizations of the process
            if i == 0:
                return self.init()

            # Choose success based on exceed a threshold with a uniform pick
            if np.random.rand(1,) < self.prob:
                self.success += 1
            y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
            print("y type: ", type(y))
            self.line.set_data(self.x, y)
            print("call !")
            return self.line,

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    fig, ax = plt.subplots()
    ud = UpdateDist(ax, prob=0.7)
    anim = FuncAnimation(fig, ud, frames=np.arange(100), init_func=ud.init,
                         interval=100, blit=True)
    plt.show()

def my_animation():
    from matplotlib.animation import FuncAnimation


    def f(xs, i):
        if i > 20:
            return [np.cos(x) for x in xs]
        return [i for x in xs]

    class UpdateGraph(object):
        def __init__(self, ax):
            self.ax = ax
            self.xs =np.arange(-3, 3, 0.1)
            self.line, = ax.plot([], [], 'k-')
            # Set up plot parameters
            self.ax.set_xlim(-3, 3)
            self.ax.set_ylim(0, 1000)


        def init(self):
            self.line.set_data([], [])
            return self.line,

        def __call__(self, i):
            if i == 0:
                return self.init()

            ys = f(self.xs, i)
            print("y type: ", type(ys))

            self.line.set_data(self.xs, ys)
            return self.line,


    fig, ax = plt.subplots()
    update_function = UpdateGraph(ax)
    anim = FuncAnimation(fig, update_function, frames=np.arange(100), init_func=update_function.init,
                         interval=100, blit=True)
    plt.show()

# PACKAGE: DO NOT EDIT THIS CELL
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import time
import timeit

def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape

    mean = np.zeros((D,1))
    ### Edit the code; iterate over the dataset and compute the mean vector.
    for n in range(N):
        # Update the mean vector
        mean = mean + (X[:,n] / N).reshape(D,1)
    ###

    return mean

def test_mean_naive():
    image_shape = (64, 64)
    # Load faces data
    dataset = fetch_olivetti_faces('./')
    faces = dataset.data.T

    print('Shape of the faces dataset: {}'.format(faces.shape))
    print('{} data points'.format(faces.shape[1]))

    mean = mean_naive(faces)
    print("mean: {}".format(mean.shape))


def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N)
    where D is the dimension and N is the number of data points"""
    D, N = X.shape

    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))

    ### Update covariance
    mean = mean_naive(X)

    # for each sample
    for n in range(N):
        # calculate for each axes
        for i in range(D):
            # the new value with each other axes
            for j in range(D):
                # covariance 0 is the one of x with each other dimensions j
                # divided by N-1 like it is in Numpy Cov instead of N
                covariance[i,j] = covariance[i,j] + (X[i, n] - mean[i][0]) * (X[j, n] - mean[j][0]) / (N-1)
    ###
    return covariance


A = np.array([[1,4,2],[3,8,3],[1,2,1],[1,2,1]]).T
B = np.array([[1,2], [4,2], [1,2]])

# print("cov with np.cov: ", np.cov(A))
# print("cov with cov_naive: ", cov_naive(A))


def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D, 1))
    ### Update mean here
    mean = np.mean(X, axis=1).reshape(D, 1)

    ###
    return mean


def cov(X):
    "Compute the covariance for a dataset"
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    ### Edit the code to compute the covariance matrix

    ### Update covariance_matrix here
    mu = mean(X)
    M = X - mean(X)
    covariance_matrix =  (M @ M.T) / (N-1)

    ###
    return covariance_matrix

A = np.array([[1,4,2],[3,8,3],[2,5,2]]).T
print("np.cov: ", np.cov(A))
print("cov: ", cov(A))

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces('./')
faces = dataset.data.T

#np.testing.assert_almost_equal(mean(faces), mean_naive(faces), decimal=6)
np.testing.assert_almost_equal(cov(faces), cov_naive(faces))

