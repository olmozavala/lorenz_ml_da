# %% Reshaping because of the way yy is saved not sure why it is like this
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

def plot_x_y_da(xx, yy, title='', save_path='', show_figure=True):
    yy_oz = np.zeros((yy.shape[0], 3))
    for i in range(yy_oz.shape[0]):
        yy_oz[i,:] = [yy[i][0], yy[i][1], yy[i][2]]
    # Plot xx in 3D using matplotlib, include yy locations with red dots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xx[:, 0], xx[:, 1], xx[:, 2])
    # Small maker size
    ax.scatter(yy_oz[:, 0], yy_oz[:, 1], yy_oz[:, 2], c='r', s=3)
    ax.set_title(title)
    if save_path != '':
        plt.savefig(join('imgs',save_path))
    # plt.show()
    if show_figure:
        plt.show()
    plt.close()
    
def plot_x_3d(xx, color='orange', linewidth=0.1, title='', save_path='', show_figure=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot with Small line width
    ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], linewidth=linewidth, color=color) 
    ax.set_title(title)
    if save_path != '':
        plt.savefig(join('imgs',save_path))
    if show_figure:
        plt.show()
    plt.close()