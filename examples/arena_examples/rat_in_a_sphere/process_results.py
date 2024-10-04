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

def plot_grid_spherical_res(ax, grid_cell, sac_plot, grid_score, corr_dict, heading = False):

    plot_on_3d_surface(ax[0], grid_cell, N_SLICES, N_STACKS, heading)
    make_plot_rate_map(grid_cell, ax[1], "Spherical coordinates" if heading else "", "Azimuthal", "Polar", "Firing rate")
    make_plot_rate_map(sac_plot, ax[2], f"SAC (gridscore = {grid_score:.3f})", "", "", "Firing rate")
    plot_rotation_correlation_curve(ax[3], corr_dict, heading=heading)

def plot_grid_proj_res(ax, orth_grid_cell, log_grid_cell, vert_grid_cell, heading = False):

    make_plot_rate_map(orth_grid_cell, ax[0], "Orthogonal projection (XY)" if heading else "", "x", "y", "Firing rate")
    make_plot_rate_map(log_grid_cell, ax[1], "Logarithmic projection (XY)" if heading else "", "x", "y", "Firing rate")
    make_plot_rate_map(vert_grid_cell, ax[2], "Orthogonal projection (YZ)" if heading else "", "y", "z", "Firing rate")

def format_qualitative_results(grid_cells, orth_grid_cells, log_grid_cells, vert_grid_cells, eig_numbers, save_path):
    os.makedirs(f"results/{save_path}", exist_ok=True)

    fig, axes = plt.subplots(len(eig_numbers), 4, figsize=(20, 24))
    GridScorer_Stachenfeld2018 = GridScorer(N_STACKS + 1)

    for idx, x in enumerate(eig_numbers):
        axes[idx,0].axis('off')
        axes[idx,0] = fig.add_subplot(len(eig_numbers), 4, 4*idx + 1, projection = '3d')
        axes[idx,0].view_init(80, -90)
        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(grid_cells[idx])
        plot_grid_spherical_res(axes[idx], grid_cells[idx], sac, grid_field_props["gridscore"],
                                grid_field_props['rotationCorrVals'], heading = idx == 0)
    fig.savefig(f"results/{save_path}/sphere_cells.pdf")
    plt.close()

    fig, axes = plt.subplots(len(eig_numbers), 3, figsize=(20, 24))
    GridScorer_Stachenfeld2018 = GridScorer(N_STACKS + 1)

    for idx, x in enumerate(eig_numbers):
        axes[idx,0].axis('off')
        plot_grid_proj_res(axes[idx], orth_grid_cells[idx], log_grid_cells[idx], vert_grid_cells[idx], heading = idx == 0)
    fig.savefig(f"results/{save_path}/proj_cells.pdf")
    plt.close()



def compile_all_results(models, eigs):

    grid_cell_spatial = []
    grid_cell_sparsity = []
    labels = []

    for model in models:
        agent, orth_agent, log_agent, vert_agent, gravity= model

        # Plot specific eigenvector results
        grid_cells = agent.get_rate_map_matrix(agent.srmat, eigs)
        orth_grid_cells = orth_agent.get_rate_map_matrix(orth_agent.srmat, eigs)
        log_grid_cells = log_agent.get_rate_map_matrix(log_agent.srmat, eigs)
        vert_grid_cells = vert_agent.get_rate_map_matrix(vert_agent.srmat, eigs)
        format_qualitative_results(grid_cells, orth_grid_cells, log_grid_cells, 
                                   vert_grid_cells, eigs, f"gravity_{gravity}")

        # Compute gridness hist
        grid_cells = agent.get_rate_map_matrix(agent.srmat)
        orth_grid_cells = orth_agent.get_rate_map_matrix(orth_agent.srmat)
        log_grid_cells = log_agent.get_rate_map_matrix(log_agent.srmat)
        vert_grid_cells = vert_agent.get_rate_map_matrix(vert_agent.srmat)
        #compile_gridness_hist(grid_cells, orth_grid_cells, log_grid_cells, vert_grid_cells, gravity)

        # Spatial info and sparsity info
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
    plt.savefig("results/spatial_info.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(16,8))
    ax = make_boxplot(ax, grid_cell_sparsity, labels, "Gravity strength", "Sparsity index")
    plt.savefig("results/sparsity_info.pdf")
    plt.close()



def compile_gridness_hist(grid_cells, orth_grid_cells, log_grid_cells, vert_grid_cells, gravity):
    GridScorer_Stachenfeld2018 = GridScorer(N_STACKS + 1)
    grid_scores, orth_scores, log_scores, vert_scores = [], [], [], []
    for grid_cell, orth_grid_cell, log_grid_cell, vert_grid_cell in zip(grid_cells, orth_grid_cells, log_grid_cells, vert_grid_cells):
        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(grid_cell)
        grid_scores.append(grid_field_props['gridscore'])

        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(orth_grid_cell)
        orth_scores.append(grid_field_props['gridscore'])

        sac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(log_grid_cell)
        log_scores.append(grid_field_props['gridscore'])

        ac, grid_field_props = GridScorer_Stachenfeld2018.get_scores(vert_grid_cell)
        vert_scores.append(grid_field_props['gridscore'])

    fig, ax = plt.subplots(1,4, figsize = (15, 8))
    sns.histplot(grid_scores, ax=ax[0], bins=20, kde=True, color='red'), ax[0].set_title("Spherical coordinates")
    sns.histplot(orth_scores, ax=ax[1], bins=20, kde=True, color='green'), ax[1].set_title("Orthogonal projection")
    sns.histplot(log_scores, ax=ax[2], bins=20, kde=True, color='blue'), ax[2].set_title("Logarithmic projection")
    sns.histplot(vert_scores, ax=ax[3], bins=20, kde=True, color='purple'), ax[3].set_title("Vertical projection")

    fig.savefig(f"results/gravity_{gravity}/gridness_hist.pdf")

if __name__ == "__main__":
    models = read_in_models()
    eigs = [1,5,10,15,20,50]

    compile_all_results(models, eigs)

