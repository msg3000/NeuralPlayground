from neuralplayground.arenas import Sphere
from neuralplayground.agents import Stachenfeld2018, RatInASphere
from neuralplayground.backend import episode_based_training_loop
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


N_SLICES = N_STACKS = 36

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

def plot_combined_results(ax, eig_number, grid_cell, orth_grid_cell, log_grid_cell, corr_dict, heading = False):

    plot_on_3d_surface(ax[0], grid_cell, N_SLICES, N_STACKS, heading)
    make_plot_rate_map(grid_cell, ax[1], "Spherical coordinates" if heading else "", "Azimuthal", "Polar", "Firing rate")
    make_plot_rate_map(orth_grid_cell, ax[2], "Orthogonal projection" if heading else "", "x", "y", "Firing rate")
    make_plot_rate_map(log_grid_cell, ax[3], "Logarithmic projection" if heading else "", "x", "y", "Firing rate")
    plot_rotation_correlation_curve(ax[4], corr_dict, heading=heading)

def plot_rotation_correlation_curve(ax, corr_dict, heading=False):
    
    angles = list(corr_dict.keys())
    correlation_values = list(corr_dict.values())

    f = interpolate.interp1d(angles, correlation_values, kind='cubic', fill_value="extrapolate")

    # Create a denser set of x values for a smoother curve
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


def compile_result(grid_cells, orth_grid_cells, log_grid_cells, eig_numbers, save_path):

    fig, axes = plt.subplots(len(eig_numbers), 5, figsize=(20, 24))
    GridScorer_Stachenfeld2018 = GridScorer(N_STACKS + 1)


    for idx, x in enumerate(eig_numbers):
        axes[idx,0].axis('off')
        axes[idx,0] = fig.add_subplot(len(eig_numbers), 5, 5*idx + 1, projection = '3d')
        axes[idx,0].view_init(80, -90)
        _, grid_field_props = GridScorer_Stachenfeld2018.get_scores(grid_cells[idx])
        plot_combined_results(axes[idx], x, grid_cells[idx], orth_grid_cells[idx], log_grid_cells[idx], grid_field_props['rotationCorrVals'], heading = idx == 0)
    fig.savefig(f"results/{save_path}")
    plt.close()

def read_in_models():

    sim_manager = SingleSim()
    models_dir = 'models'
    variants = [os.path.join(models_dir, name) for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name)) and name.startswith('gravity_')]
    models = []
    for variant in variants:
        agent, env, _ = sim_manager.load_results(os.path.join(variant, "spherical"))
        orth_agent, env, _ = sim_manager.load_results(os.path.join(variant, "orth_proj"))
        log_agent, env, _ = sim_manager.load_results(os.path.join(variant, "log_proj"))
        models.append((agent, orth_agent, log_agent, variant[-1:-4:-1]))
    return models

def compile_all_results(models, eigs):

    for model in models:
        agent, orth_agent, log_agent, gravity= model
        grid_cells = agent.get_rate_map_matrix(agent.srmat, eigs)
        orth_grid_cells = orth_agent.get_rate_map_matrix(orth_agent.srmat, eigs)
        log_grid_cells = log_agent.get_rate_map_matrix(log_agent.srmat, eigs)
        compile_result(grid_cells, orth_grid_cells, log_grid_cells, eigs, f"eigs_gravity_{gravity}")

if __name__ == "__main__":
    models = read_in_models()
    eigs = [1,5,10,15,20,50]
    compile_all_results(models, eigs)

