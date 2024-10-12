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

N_STACKS = N_SLICES = 36

def extract_gravity(variant_path):
    base_name = os.path.basename(variant_path)
    gravity_str = base_name.split('_')[1]
    gravity_value = float(gravity_str)/100
    return gravity_value

def read_in_models():

    sim_manager = SingleSim()
    models_dir = 'models'
    variants = [os.path.join(models_dir, name) for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name)) and name.startswith('gravity_')]
    variants = sorted(variants, key=extract_gravity)
    models = []
    for variant in variants:
        agent, env, _ = sim_manager.load_results(os.path.join(variant, "spherical"))
        orth_agent, env, _ = sim_manager.load_results(os.path.join(variant, "orth_proj"))
        log_agent, env, _ = sim_manager.load_results(os.path.join(variant, "log_proj"))
        vert_agent, env, _ = sim_manager.load_results(os.path.join(variant, "vert_proj"))
        models.append((agent, orth_agent, log_agent, vert_agent, float(variant.split("_")[1][2:])/100))
    return models

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

def make_boxplot(ax, df, x_key, y_key, hue_key, x_label, y_label):
    sns.boxplot(
        data=df,
        x=x_key,
        y=y_key,
        hue=hue_key,
        ax=ax,
        palette='pastel',
        width=0.6,
        linewidth=2,
        flierprops=dict(marker='o', markersize=5, linestyle='none', markerfacecolor='red')
    )
    ax.set_title(f'Comparison of {y_label} across {x_label}', fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    sns.despine(offset=10, trim=True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title='Spatial representation')
    return ax


def plot_on_3d_surface(ax, grid_cell, n_slices, n_stacks, heading=False):

    # Plot 3d-hemisphere
    phi = np.linspace(np.pi/2, np.pi, n_stacks)
    theta = np.linspace(0, 2*np.pi, n_slices)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    # Apply rate maps to colours
    norm = plt.Normalize(grid_cell.min(), grid_cell.max())
    ax.plot_surface(x,y,z, facecolors = plt.cm.jet(norm(grid_cell.T)), linewidth=0, antialiased=False, alpha=0.8)
    if heading :
        ax.set_title("Euclidean grid cells")
    return ax


def plot_rotation_correlation_curve(ax, corr_dict, heading=False):
    
    angles = list(corr_dict.keys())
    correlation_values = list(corr_dict.values())

    f = interpolate.interp1d(angles, correlation_values, kind='cubic', fill_value="extrapolate")

    x_dense = np.linspace(0, 180, 1000)
    y_dense = f(x_dense)

    # Create the plot
    ax.plot(x_dense, y_dense, '-', label='Interpolated')
    ax.plot(angles, correlation_values, 'ro', label='Original data')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Correlation Value')
    if heading:
        ax.set_title('Rotation Correlation Values')
    ax.grid(True)
    ax.set_xticks(angles)


def proj_3d_map(grid_cell, axes='yz', grid_id = None, model_name = None):
    phi = np.linspace(np.pi/2, np.pi, N_STACKS)
    theta = np.linspace(0, 2*np.pi, N_SLICES)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    x,y,z = x.flatten(), y.flatten(), z.flatten()
    grid_cell = grid_cell.flatten()

    num_bins = 36
    x_edges = np.linspace(-1, 1, num_bins)
    y_edges = np.linspace(-1, 1, num_bins)
    z_edges = np.linspace(-1, 0, num_bins)

    # Compute 2D histogram
    hist, y_edges, z_edges = np.histogram2d(y, z, bins=[y_edges, z_edges], weights=grid_cell)

    # Normalize by counts to get average firing rate per bin
    counts, _, _ = np.histogram2d(y, z, bins=[y_edges, z_edges])
    average_firing_rate = hist / counts
    average_firing_rate = np.nan_to_num(average_firing_rate)  # Replace NaNs with zero

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(average_firing_rate.T, origin='lower', extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
            aspect='auto', cmap='jet')
    plt.colorbar(label='Average Firing Rate')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Firing Rate Heatmap on XY Plane')
    plt.savefig(f"results/test_proj/proj_{model_name}_{grid_id}")

