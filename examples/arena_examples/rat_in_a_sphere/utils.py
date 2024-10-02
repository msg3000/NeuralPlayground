import numpy as np
from neuralplayground.plotting.plot_utils import make_plot_rate_map
from neuralplayground.backend import SingleSim
from neuralplayground.comparison import GridScorer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from projection_agents import RatOnTangent, RatOnLogarithmicTangent
from scipy import interpolate
import os

def compute_spatial_info(p: np.ndarray, X: np.ndarray):

    """ Compute spatial information index
        p : numpy.ndarray -> probability map of location occupancy
        X : numpy.ndarray -> 2d array of firing rate per location
    """

    p = p.flatten()
    F_rates = np.abs(X.flatten()) # abs due to log
    mean_rate = F_rates.mean()
    non_zero = F_rates > 0 
    non_zero_rates = F_rates[non_zero]
    sp_info = non_zero_rates * np.log2(non_zero_rates / mean_rate) / mean_rate
    sp_info = p[non_zero] * sp_info

    return np.sum(sp_info)

def compute_sparsity_info(p: np.ndarray, X: np.ndarray):

    """ Sparsity index
    """

    p, X = p.flatten(), np.abs(X.flatten())
    return (np.sum(p*X)**2) / np.sum(p*(X**2))

def make_boxplot(ax, data, labels, x_label, y_label):
    sns.boxplot(
        data=data,
        ax=ax,
        notch=True,
        palette='pastel',  
        width=0.6,         
        linewidth=2,     
        flierprops=dict(marker='o', markersize=5, linestyle='none', markerfacecolor='red')
    )
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(f'Comparison of {y_label} across {x_label}', fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    sns.despine(offset=10, trim=True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    return ax


