import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

markers = {'SGD': '', 'SNGD': '', 'DSNGD': '', 'MAP': '', 'DSNGD_map': '', 'DSNGD_Theta': '', 'DSNGD_sgd': '',
           'CSNGD': '', 'CSNGD2': '', 'AdaGrad': '', 'MOD10': '','MOD50': '','MOD100': '','MOD500': '','MOD1000': '',
           'MOD5': '','MOD20': '','MOD200': '','MOD2000': '', 'MEGD': ''}
linestyles = {'SGD': '-', 'SNGD': ':', 'DSNGD': '-.', 'MAP': '-.', 'DSNGD_map': ':', 'DSNGD_Theta': '-', 'DSNGD_sgd': '-',
              'CSNGD': '--','CSNGD2': '--', 'AdaGrad': '-',
              'MOD10': '-.', 'MOD100': '-.', 'MOD1000': '-.', 'MOD50': '-.', 'MOD500': '-.',
              'MOD5': '-.', 'MOD20': '-.', 'MOD200': '-.', 'MOD2000': '-.', 'MEGD': '-.'}

markersizes= {'SGD': '1', 'SNGD': '1', 'DSNGD': '1', 'MAP': '1', 'DSNGD_map': '1', 'DSNGD_Theta': '1', 'DSNGD_sgd': '1',
              'CSNGD': '1', 'CSNGD2': '1', 'AdaGrad': '1','MOD10': '1','MOD50': '1','MOD100': '1','MOD500': '1',
              'MOD5': '1','MOD20': '1','MOD200': '1','MOD2000': '1','MOD1000': '1','MEGD': '1'}

color = {'SGD': 'C1', 'SNGD': 'C4', 'DSNGD': 'C2', 'MAP': 'C3', 'DSNGD_map': None, 'DSNGD_Theta': None, 'DSNGD_sgd': None,
         'CSNGD': 'C0','CSNGD2': 'C0', 'AdaGrad': 'C5','MOD5': 'C1', 'MOD20': 'C2', 'MOD200': 'C4', 'MOD2000': 'C3',
         'MOD10': 'C1', 'MOD100': 'C2', 'MOD1000': 'C5', 'MOD50': 'C3', 'MOD500': 'C5', 'MEGD': 'C9'}

alpha = {'SGD': 1, 'SNGD': 1, 'DSNGD': 1, 'MAP': 1, 'DSNGD_map': 1, 'DSNGD_Theta': 1, 'DSNGD_sgd': 1,
               'CSNGD': 1, 'AdaGrad': 1,'MOD10': 0.9, 'MOD100': 0.5, 'MOD1000': 1, 'MOD50': 0.7, 'MOD500': 1}

ratio = {'SGD': 1, 'SNGD': 1, 'DSNGD':  1, 'MAP': 1,
         'DSNGD_map':  1, 'DSNGD_Theta':  1, 'DSNGD_sgd':  1,
               'CSNGD': 1, 'AdaGrad':  1,
         'MOD10':  0.9, 'MOD100':  0.3, 'MOD1000':  1, 'MOD50':  3.5, 'MOD500':  0.1}


def plot_lines(x, graphs, labels, x_labels=False, y_labels=False, low_lines=None, high_lines=None):
    rows = graphs.shape[0]
    columns = graphs.shape[1]
    grid_num = rows*100 + columns*10 + 1
    # plt.figure(figsize=(14, 3.9))
    plt.figure(figsize=(14, 3.9))
    # plt.suptitle(measure.name + ' function, k=' + str(n_classes))
    for row in range(rows):
        for col in range(columns):
            ax = plt.subplot(grid_num + columns * row + col)
            # if x_labels:
            #     plt.xlabel(x_labels[i])
            # if y_label and i==0:
            #     plt.ylabel(y_label[0], rotation=0)
            # if row == 0:
            #     ax.set_ylim([0.001, 3.])
            # elif row == 1:
            #     ax.set_ylim([0.01, 5.])
            # else:
            #     ax.set_ylim([0.01, 10])
            if col != 0:
                ax.get_yaxis().set_visible(False)
            elif y_labels:
                plt.ylabel(y_labels[row], rotation=0)
            if row != rows-1:
                ax.get_xaxis().set_visible(False)
            elif x_labels:
                plt.xlabel(x_labels[col])

            lines = graphs[row, col]
            for l in range(len(lines)):
                line = lines[l]
                l_line = low_lines[row, col][l]
                h_line = high_lines[row, col][l]
                name = labels[l]
                if (line > 0).all():
                    plt.semilogy()
                    plt.plot(x, line, label=name if col + row == 0 else "",
                             color=color[name], linestyle=linestyles[name], marker=markers[name],
                             markersize=markersizes[name])
                    # uncomment below line to fill quartiles
                    plt.fill_between(x, l_line, h_line, facecolor=color[name], alpha=0.35)
            plt.figlegend(loc=8, ncol=len(labels))
    plt.show()


def plot_surface_3D(X, Y, Z, x_label=False, y_label=False, z_label=False, title=''):
    # plt.figure(figsize=(14, 3.9))
    fig = plt.figure()
    plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.axes(projection='3d')
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label, rotation=0)
    if z_label:
        ax.set_zlabel(z_label, rotation=0)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.Spectral, linewidth=0, antialiased=False)

    # Customize axes.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.set_xlim(0, 5000)
    ax.xaxis.set_major_locator(LinearLocator(2))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def surpass_args(arr, part):
    args = []
    for p in part:
        args.append(surpass_arg(0, arr, p))
    return np.array(args, dtype=float)


def surpass_arg(i, arr, value):
    if i > (len(arr) - 1):
        return len(arr)-1
    if np.amax(arr[i:]) <= value:
        return i
    i = np.argmax(arr[i:]) + i + 1
    return surpass_arg(i, arr, value)
