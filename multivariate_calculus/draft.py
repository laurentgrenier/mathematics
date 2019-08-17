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




