import matplotlib.pyplot as plt

class Drawing():
    def __init__(self, nrows, ncols, xlim=(-8,8), ylim=(-2,2), width=16):
        self.nrows = nrows
        self.ncols = ncols
        self.xlim = xlim
        self.ylim = ylim
        self.axes = []
        self.width = width

        self.__create_axes()

    def __create_axes(self):
        """ Set axes and figure styles """
        fig = plt.figure(figsize=(self.width, self.width / (self.nrows/self.ncols)))
        fig.patch.set_facecolor('violet')
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        medium_grey = (0.4, 0.4, 0.4)

        for row in range(self.nrows):
            for col in range(self.ncols):
                index = row * self.ncols + col + 1

                ax = fig.add_subplot(self.nrows, self.ncols, index)
                ax.patch.set_facecolor('none')
                ax.spines['left'].set_position('zero')
                ax.spines['bottom'].set_position('zero')
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.spines['left'].set_color(medium_grey)
                ax.spines['bottom'].set_color(medium_grey)
                ax.tick_params(axis='x', colors=medium_grey)
                ax.tick_params(axis='y', colors=medium_grey)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.set_facecolor((0.1, 0.1, 0.1))
                ax.set_xlim(self.xlim[0], self.xlim[1])
                ax.set_ylim(self.ylim[0], self.ylim[1])
                self.axes.append(ax)

    def plot(self, X, Y, title, color='pink', index=0):
        self.axes[index].plot(X, Y, color=color)
        self.axes[index].set_title(title, loc='right', color='0.7')

    def show(self):
        plt.show()

