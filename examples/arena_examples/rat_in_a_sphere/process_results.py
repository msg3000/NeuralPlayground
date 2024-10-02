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
from utils import *

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
        orth_agent, env, _ = sim_manager.load_results(os.path.join(variant, "vert_proj"))
        log_agent, env, _ = sim_manager.load_results(os.path.join(variant, "log_proj"))
        models.append((agent, orth_agent, log_agent, variant.split("_")[1][2:]))
    return models

def compile_all_results(models, eigs):

    grid_cell_spatial = []
    grid_cell_sparsity = []
    labels = []

    for model in models:
        agent, orth_agent, log_agent, gravity= model

        # Plot specific eigenvector results
        grid_cells = agent.get_rate_map_matrix(agent.srmat, eigs)
        orth_grid_cells = orth_agent.get_rate_map_matrix(orth_agent.srmat, eigs)
        log_grid_cells = log_agent.get_rate_map_matrix(log_agent.srmat, eigs)
        #compile_result(grid_cells, orth_grid_cells, log_grid_cells, eigs, f"eigs_vert_gravity_{gravity}")

        # Compute gridness hist
        grid_cells = agent.get_rate_map_matrix(agent.srmat)
        orth_grid_cells = orth_agent.get_rate_map_matrix(orth_agent.srmat)
        log_grid_cells = log_agent.get_rate_map_matrix(log_agent.srmat)
        # compile_gridness_hist(grid_cells, orth_grid_cells, log_grid_cells, gravity)

        labels.append(f"g = 0.{gravity}")
        spatial_info = []
        sparsity_info = []

        p = agent.freq_map / np.sum(agent.freq_map)

        for grid_cell in grid_cells:
            spatial_info.append(compute_spatial_info(p, grid_cell))
            sparsity_info.append(compute_sparsity_info(p, grid_cell))
        
        grid_cell_spatial.append(spatial_info)
        grid_cell_sparsity.append(sparsity_info)

        #for grid_id, grid_cell in zip(eigs, grid_cells):
        #   proj_3d_map(grid_cell, grid_id=grid_id, model_name=gravity)
    fig, ax = plt.subplots(figsize=(16,8))
    ax = make_boxplot(ax, grid_cell_spatial, labels, "Gravity strength", "Spatial information")
    plt.savefig("results/spatial_info")
    plt.close()

    fig, ax = plt.subplots(figsize=(16,8))
    ax = make_boxplot(ax, grid_cell_sparsity, labels, "Gravity strength", "Sparsity index")
    plt.savefig("results/sparsity_info")
    plt.close()




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

def compile_gridness_hist(grid_cells, orth_grid_cells, log_grid_cells, gravity):
    GridScorer_Stachenfeld2018 = GridScorer(N_STACKS + 1)
    grid_scores, orth_scores, log_scores = [], [], []
    for grid_cell, orth_grid_cell, log_grid_cell in zip(grid_cells, orth_grid_cells, log_grid_cells):
        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(grid_cell)
        grid_scores.append(grid_field_props['gridscore'])

        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(orth_grid_cell)
        orth_scores.append(grid_field_props['gridscore'])

        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(log_grid_cell)
        log_scores.append(grid_field_props['gridscore'])
    fig, ax = plt.subplots(1,3)
    sns.histplot(grid_scores, ax=ax[0], bins=20, kde=True, color='red'), ax[0].set_title("Spherical coordinates")
    sns.histplot(orth_scores, ax=ax[1], bins=20, kde=True, color='green'), ax[1].set_title("Orthogonal projection")
    sns.histplot(log_scores, ax=ax[2], bins=20, kde=True, color='blue'), ax[2].set_title("Logarithmic projection")

    fig.savefig(f"results/grid_v_hist_{gravity}")

if __name__ == "__main__":
    models = read_in_models()
    eigs = [1,5,10,15,20,50]
    compile_all_results(models, eigs)

