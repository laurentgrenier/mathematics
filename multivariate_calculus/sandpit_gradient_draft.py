# Following imports pylab notebook without giving the user rubbish messages
import os, sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from scipy.misc import imread
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import ipywidgets as widgets
from IPython.display import display, Markdown

matplotlib.rcParams['figure.subplot.left'] = 0


# matplotlib.rcParams['figure.figsize'] = (7, 6)

class Sandpit:
    def __init__(self, f, simulation=False):
        # Default options
        self.game_mode = 0  # 0 - Jacobian, 1 - Depth Only, 2 - Steepest Descent
        self.grad_length = 1 / 5
        self.grad_max_length = 1
        self.arrowhead_width = 0.1
        self.arrow_placement = 2  # 0 - tip, 1 - base, 2 - centre, 3 - tail
        self.tol = 0.15  # Tolerance
        self.markerColour = (1, 0.85, 0)
        self.contourCM = LinearSegmentedColormap.from_list("Cmap", [
            (0., 0.00505074, 0.191104),
            (0.155556, 0.0777596, 0.166931),
            (0.311111, 0.150468, 0.142758),
            (0.466667, 0.223177, 0.118585),
            (0.622222, 0.295886, 0.094412),
            (0.777778, 0.368595, 0.070239),
            (0.822222, 0.389369, 0.0633324),
            (0.866667, 0.410143, 0.0564258),
            (0.911111, 0.430917, 0.0495193),
            (0.955556, 0.451691, 0.0426127),
            (1., 0.472465, 0.0357061)
        ], N=256)
        self.start_text = "**Click anywhere in the sandpit to place the dip-stick.**"
        self.win_text = "### Congratulations!\nWell done, you found the phone."

        # Initialisation variables
        self.revealed = False
        self.handler_map = {}
        self.nGuess = 0
        self.simulation = simulation

        # Parameters
        self.f = f  # Contour function

        # (x,y) coordinates of the global minimum of the function.
        x0 = self.x0 = differential_evolution(lambda xs: f(xs[0], xs[1]), ((0, 6), (0, 6))).x

        # (x,y) coordinates of the global minimum of the inverse function.
        # i.e the coordinates of the global maximum of the function
        x1 = differential_evolution(lambda xs: -f(xs[0], xs[1]), ((0, 6), (0, 6))).x

        # get z of the minimum function
        f0 = f(x0[0], x0[1])

        # get z of the maximum function
        f1 = f(x1[0], x1[1])

        # normalization of z from [0,1] to [0,-9]
        self.f = lambda x, y: 8 * (f(x, y) - f1) / (f1 - f0) - 1

        self.df = lambda x, y: np.array(
            [self.f(x + 0.01, y) - self.f(x - 0.01, y), self.f(x, y + 0.01) - self.f(x, y - 0.01)]) / 0.02

        self.d2f = lambda x, y: np.array([
            [self.df(x + 0.01, y)[0] - self.df(x - 0.01, y)[0], self.df(x, y + 0.01)[0] - self.df(x, y - 0.01)[0]],
            [self.df(x + 0.01, y)[1] - self.df(x - 0.01, y)[1], self.df(x, y + 0.01)[1] - self.df(x, y - 0.01)[1]]
        ]) / 0.02

    def draw(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 6])
        self.ax.set_ylim([0, 6])
        self.ax.set_aspect(1)
        self.fig.canvas.mpl_connect('button_press_event', lambda e: self.onclick(e))
        self.drawcid = self.fig.canvas.mpl_connect('draw_event', lambda e: self.ondraw(e))

        self.leg = self.ax.legend(handles=[], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Depths:")
        img = imread("readonly/sand.png")
        self.ax.imshow(img, zorder=0, extent=[0, 6, 0, 6], interpolation="bilinear")

        print(self.start_text)

        self.showContours()

    def onclick(self, event):
        print("click !!")
        if (event.button != 1):
            return
        x = event.xdata
        y = event.ydata
        self.placeArrow(x, y)

        if np.linalg.norm(self.x0 - [x, y]) <= self.tol:
            self.showContours()
            return
        lx = minimize(lambda xs: self.f(xs[0], xs[1]), np.array([x, y])).x
        if np.linalg.norm(lx - [x, y]) <= self.tol:
            self.local_min(lx[0], lx[1])
            return

        i = 5
        if self.game_mode == 2:
            while i > 0:
                i = i - 1
                dx = self.next_step(self.f(x, y), self.df(x, y), self.d2f(x, y))

                self.ax.plot([x, x + dx[0]], [y, y + dx[1]], '-', zorder=15, color=(1, 0, 0, 0.5), ms=6)
                x += dx[0]
                y += dx[1]

                if x < 0 or x > 6 or y < 0 or y > 6:
                    break

                self.placeArrow(x, y, auto=True)

                if np.linalg.norm(self.x0 - [x, y]) <= self.tol:
                    self.showContours()
                    break

                lx = minimize(lambda xs: self.f(xs[0], xs[1]), np.array([x, y])).x
                if np.linalg.norm(lx - [x, y]) <= self.tol:
                    self.local_min(lx[0], lx[1])
                    break

    def random_click(self):
        x = np.random.random() * 6.
        y = np.random.random() * 6.

        if np.linalg.norm(self.x0 - [x, y]) <= self.tol:
            return True

        lx = minimize(lambda xs: self.f(xs[0], xs[1]), np.array([x, y])).x
        if np.linalg.norm(lx - [x, y]) <= self.tol:
            return False

        i = 5
        if self.game_mode == 2:
            while i > 0:
                i = i - 1
                dx = self.next_step(self.f(x, y), self.df(x, y), self.d2f(x, y))

                x += dx[0]
                y += dx[1]

                if x < 0 or x > 6 or y < 0 or y > 6:
                    return False

                if np.linalg.norm(self.x0 - [x, y]) <= self.tol:
                    return True

                lx = minimize(lambda xs: self.f(xs[0], xs[1]), np.array([x, y])).x
                if np.linalg.norm(lx - [x, y]) <= self.tol:
                    return False

        return False

    def ondraw(self, event):
        self.fig.canvas.mpl_disconnect(self.drawcid)  # Only do this once, then self destruct the event.
        self.displayMsg(self.start_text)

    def placeArrow(self, x, y, auto=False):
        d = -self.df(x, y) * self.grad_length
        dhat = d / np.linalg.norm(d)
        d = d * np.clip(np.linalg.norm(d), 0, self.grad_max_length) / np.linalg.norm(d)

        if self.arrow_placement == 0:  # tip
            off = d + dhat * 1.5 * self.arrowhead_width
        elif self.arrow_placement == 1:  # head
            off = d
        elif self.arrow_placement == 2:  # centre
            off = d / 2
        else:  # tail
            off = np.array((0, 0))

        if auto:
            self.ax.plot([x], [y], 'yo', zorder=25, color="red", ms=6)
        else:
            self.nGuess += 1

            p, = self.ax.plot([x], [y], 'yo', zorder=25, label=
            str(self.nGuess) + ") %.2fm" % self.f(x, y), color=self.markerColour, ms=8, markeredgecolor="black")

            if (self.nGuess <= 25):
                self.ax.text(x + 0.2 * dhat[1], y - 0.2 * dhat[0], str(self.nGuess))

                self.handler_map[p] = HandlerLine2D(numpoints=1)

                self.leg = self.ax.legend(handler_map=self.handler_map, bbox_to_anchor=(1.05, 1), loc=2,
                                          borderaxespad=0., title="Depths:")

                if (self.nGuess == 22 and not self.revealed):
                    self.displayMsg("**Hurry Up!** The supervisor has calls to make.")
            elif not self.revealed:
                self.showContours()
                self.displayMsg(
                    "**Try again.** You've taken too many tries to find the phone. Reload the sandpit and try again.")

        if self.game_mode != 1:
            self.ax.arrow(x - off[0], y - off[1], d[0], d[1],
                          linewidth=1.5, head_width=self.arrowhead_width,
                          head_starts_at_zero=False, zorder=20, color="black")

    def showContours(self):
        if self.revealed:
            return
        x0 = self.x0
        X, Y = np.meshgrid(np.arange(0, 6, 0.05), np.arange(0, 6, 0.05))
        self.ax.contour(X, Y, self.f(X, Y), 10, cmap=self.contourCM)
        img = imread("readonly/phone2.png")
        self.ax.imshow(img, zorder=30,
                       extent=[x0[0] - 0.375 / 2, x0[0] + 0.375 / 2, x0[1] - 0.375 / 2, x0[1] + 0.375 / 2],
                       interpolation="bilinear")
        self.displayMsg(self.win_text)
        self.revealed = True

    def local_min(self, x, y):
        img = imread("readonly/nophone.png")
        self.ax.imshow(img, zorder=30, extent=[x - 0.375 / 2, x + 0.375 / 2, y - 0.375 / 2, y + 0.375 / 2],
                       interpolation="bilinear")
        if not self.revealed:
            self.displayMsg("**Oh no!** You've got stuck in a local optimum. Try somewhere else!")

    def displayMsg(self, msg):
        print(msg)


def sandpit_gradient(next_step):
    a = np.random.rand(4, 4) * np.outer(np.arange(4) + 1., np.arange(4) + 1.) ** -2
    φx = 2 * np.pi * np.random.rand(4, 4)
    φy = 2 * np.pi * np.random.rand(4, 4)
    fn = lambda n, m, x, y: a[n, m] * np.cos(np.pi * n * x / 6 + φx[n, m]) * np.cos(np.pi * m * y / 6 + φy[n, m])
    ff = lambda x, y: (fn(0, 0, x, y) + fn(0, 1, x, y) + fn(0, 2, x, y) + fn(0, 3, x, y) +
                       fn(1, 0, x, y) + fn(1, 1, x, y) + fn(1, 2, x, y) + fn(1, 3, x, y) +
                       fn(2, 0, x, y) + fn(2, 1, x, y) + fn(2, 2, x, y) + fn(2, 3, x, y) +
                       fn(3, 0, x, y) + fn(3, 1, x, y) + fn(3, 2, x, y) + fn(3, 3, x, y) +
                       (1 - (x * (x - 6) * y * (y - 6)) / (81)) ** 7 / 9
                       )

    # ff = lambda x, y: 10*np.exp(-((x-4)**2 + (y-4)**2)/4)


    def launch_game():
        sp = Sandpit(ff)
        sp.game_mode = 2
        sp.next_step = next_step
        sp.win_text = """
        ### Congratulations!
        Well done, you found the phone.
    
        You may run this example again to find the phone in a different landscape.
                """
        sp.draw()

    def launch_simulation(id):
        a = np.random.rand(4, 4) * np.outer(np.arange(4) + 1., np.arange(4) + 1.) ** -2
        φx = 2 * np.pi * np.random.rand(4, 4)
        φy = 2 * np.pi * np.random.rand(4, 4)
        fn = lambda n, m, x, y: a[n, m] * np.cos(np.pi * n * x / 6 + φx[n, m]) * np.cos(np.pi * m * y / 6 + φy[n, m])
        ff = lambda x, y: (fn(0, 0, x, y) + fn(0, 1, x, y) + fn(0, 2, x, y) + fn(0, 3, x, y) +
                           fn(1, 0, x, y) + fn(1, 1, x, y) + fn(1, 2, x, y) + fn(1, 3, x, y) +
                           fn(2, 0, x, y) + fn(2, 1, x, y) + fn(2, 2, x, y) + fn(2, 3, x, y) +
                           fn(3, 0, x, y) + fn(3, 1, x, y) + fn(3, 2, x, y) + fn(3, 3, x, y) +
                           (1 - (x * (x - 6) * y * (y - 6)) / (81)) ** 7 / 9
                           )

        import time
        sp = Sandpit(ff)
        sp.game_mode = 2
        sp.next_step = next_step

        win = False

        MAX_TURN = 10
        i = MAX_TURN
        turn_count = 0
        while i > 0:
            i -= 1
            turn_count += 1

            win = sp.random_click()
            print("win = ", win)
            if win:
                print("{} turn(s) \n YOU WIN !".format(turn_count))
                break

        if not win:
            print("{} turn(s) \n GAME OVER !".format(turn_count))

        return { "id":id, "gamma": gamma,"turn_count":turn_count, "win":win }

    def plot_3d():
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        # get the points
        x = np.linspace(0, 6, 100)
        y = np.linspace(0, 6, 100)

        # populate the whole grid
        X, Y = np.meshgrid(x, y)

        x0 = differential_evolution(lambda xs: ff(xs[0], xs[1]), ((0, 6), (0, 6))).x

        # (x,y) coordinates of the global minimum of the inverse function.
        # i.e the coordinates of the global maximum of the function
        x1 = differential_evolution(lambda xs: -ff(xs[0], xs[1]), ((0, 6), (0, 6))).x

        # get z of the minimum function
        f0 = ff(x0[0], x0[1])

        # get z of the maximum function
        f1 = ff(x1[0], x1[1])

        # normalized function
        f = lambda x, y: 8 * (ff(x, y) - f1) / (f1 - f0) - 1

        # differential
        df = lambda x, y: np.array(
            [f(x + 0.01, y) - f(x - 0.01, y), f(x, y + 0.01) - f(x, y - 0.01)]) / 0.02


        class Point:
            def __init__(self, x, y, z = 0):
                self.x = x
                self.y = y
                self.z = z

        p = Point(2, 3)
        p.z = f(p.x,p.y)

        df_values = df(p.x,p.y)
        jacobian = Point(p.x + df_values[0], p.y + df_values[0], f(p.x + df_values[0], p.y + df_values[1]))

        p1 = Point(1, 3)
        p1.z = f(p1.x, p1.y)
        df_values = df(p1.x, p1.y)
        jacobian1 = Point(p1.x + df_values[0], p1.y + df_values[0], f(p1.x + df_values[0], p1.y + df_values[1]))

        p2 = Point(3, 3)
        p2.z = f(p2.x, p2.y)
        df_values = df(p2.x, p2.y)
        jacobian2 = Point(p2.x + df_values[0], p2.y + df_values[0], f(p2.x + df_values[0], p2.y + df_values[1]))

        print("Jacobian: ({},{},{})".format(jacobian.x, jacobian.y, jacobian.z))


        Z = ff(X, Y)
        Z_norm = f(X, Y)
        Z_inv = -ff(X, Y)

        # plotting the results
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        # drawing the function
        #ax.plot_surface(X, Y, Z, cmap=cm.jet)
        #ax.plot_surface(X, Y, Z, cmap=cm.summer)
        ax.plot_surface(X, Y, Z_norm, cmap=cm.summer)
        ax.plot([p.x, jacobian.x], [p.y, jacobian.y], zs=[p.z, jacobian.z], color="red", zorder=16)
        ax.plot([p1.x, jacobian1.x], [p1.y, jacobian1.y], zs=[p1.z, jacobian1.z], color="orange", zorder=16)
        ax.plot([p2.x, jacobian2.x], [p2.y, jacobian2.y], zs=[p2.z, jacobian2.z], color="green", zorder=16)


        cset = ax.contour(X, Y, Z, zdir='z', levels=20, offset=0, cmap=cm.jet)
        # cset = ax.contour(X, Y, Z, zdir='x', offset=-4, cmap=cm.jet)
        # cset = ax.contour(X, Y, Z, zdir='y', offset=-80, cmap=cm.jet)


        plt.show()

    def stats(results):
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("sandpit_stats.csv")
        print("win - loose count: {} - {}".format(df[df["win"] == True].shape[0], df[df["win"] == False].shape[0]))
        print(df)
        print("parties count: ", df.shape[0])

    gamma = 0.1

    results = []

    while gamma < 3:
        gamma += 0.1
        SIM_COUNT = 1000
        sim_id = 0

        while sim_id < SIM_COUNT:
            sim_id += 1
            result = launch_simulation(sim_id)
            results.append(result)

    print("result: ", results)
    stats(results)

def next_step_old1(f, J, H) :
    return np.array([1, 0])

def next_step_old2(f, J, H):
    gamma = 0.5
    return -gamma * J

def next_step(f, J, H) :

    return -np.linalg.inv(H) @ J

gamma = 0.5

def next_step(f, J, H) :
    #gamma = 0.5
    step = -np.linalg.inv(H) @ J
    if step @ -J <= 0 or np.linalg.norm(step) > np.random.random() :
        step = -gamma * J

    return step


def compare_differential():
    f = lambda x,y: np.exp(-(x**2 + y**2))
    df = lambda x, y: np.array(
        [f(x + 0.01, y) - f(x - 0.01, y), f(x, y + 0.01) - f(x, y - 0.01)]) / 0.02
    J = lambda x,y: np.array(
        [-2*x*np.exp(-(x**2 + y**2)), -2*y*np.exp(-(x**2 + y**2))])

    print("f:", f(2,2))
    print("df:", df(2, 2))
    print("J:", J(2, 2))

# compare_differential()
# sandpit_gradient(next_step)

def stats():
    from multivariate_calculus.lib.drawing import Drawing

    df = pd.read_csv("sandpit_stats.csv", index_col='id')

    win_by_gamma = df[df["win"] == True].groupby(["gamma"]).size()
    mean = win_by_gamma.mean()

    draw = Drawing(1, 1, xlim=(-0.2, 3), ylim=(-1, 1000))
    plt.plot([-0.2, 3], [mean,mean], linestyle='dashed', color='grey')
    draw.plot(win_by_gamma.index.values, win_by_gamma, 'frequency', color='violet')
    draw.show()

stats()

def stats_2():
    from multivariate_calculus.lib.drawing import Drawing

    df = pd.read_csv("sandpit_stats.csv", index_col='id')

    mean_turn_count_by_gamma = df[df["win"] == True].groupby(["gamma"])["turn_count"].mean()
    mean = mean_turn_count_by_gamma.mean()

    draw = Drawing(1, 1, xlim=(-0.2, 3), ylim=(-1, 6))
    plt.plot([-0.2, 3], [mean,mean], linestyle='dashed', color='grey')
    draw.plot(mean_turn_count_by_gamma.index.values, mean_turn_count_by_gamma, 'velocity', color='violet')
    draw.show()



    print(mean_turn_count_by_gamma)

stats_2()



