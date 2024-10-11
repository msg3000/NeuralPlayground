import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt 
import matplotlib as mpl
from neuralplayground.plotting.plot_utils import make_plot_rate_map
from neuralplayground.agents import Stachenfeld2018
from neuralplayground.arenas import Sphere
import random
from typing import Union

class RatOnTangent(Stachenfeld2018):
    def __init__(self, agent_name: str = "SR", discount: float = 0.9, threshold: float = 0.000001, lr_td: float = 0.01, room_width: float = 12, room_depth: float = 12, state_density: float = 1, twoD: bool = True, replicable: bool = True, **mod_kwargs):
        super().__init__(agent_name, discount, threshold, lr_td, room_width, room_depth, state_density, twoD, **mod_kwargs)
        self.replicable = replicable
        if self.replicable:
            np.random.seed(42)
        self.freq_map = np.zeros(self.n_state)

    def obs_to_state(self, pos: ndarray):
        # Compute the bin size in both x and y directions
        bin_size_x = self.room_width / (self.width - 1)
        bin_size_y = self.room_depth / (self.depth - 1)

        # Map the observed position to the closest bin indices
        x_index = int(np.round((pos[0] - (-self.room_width / 2)) / bin_size_x))
        y_index = int(np.round((pos[1] - (-self.room_depth/2)) / bin_size_y))

        # Clamp indices to ensure they are within valid bounds
        x_index = np.clip(x_index, 0, self.width-1)
        y_index = np.clip(y_index, 0, self.depth-1)

        # Convert 2D (x_index, y_index) to a single state index in the linear state space
        curr_state = y_index * self.width + x_index

        return curr_state
    
    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [
                obs,
            ]

        if len(obs) == 0:
            action = None
        else:
            # Random policy
            action = np.random.uniform(-1,1,3)
            self.next_state = self.obs_to_state(obs[0][:2])
            self.freq_map[self.next_state] += 1
            action = np.array(action)
        return action
    
    
    def get_rate_map_matrix(
        self,
        sr_matrix=None,
        eigen_vector: Union[int, list, tuple] = None,
    ):
        if sr_matrix is None:
            sr_matrix = self.successor_rep_solution()
        if eigen_vector is None:
            eigen_vector = np.arange(self.width * self.depth)
        evals, evecs = np.linalg.eig(sr_matrix)
        if isinstance(eigen_vector, int):
            return evecs[:, eigen_vector].reshape((self.depth, self.width)).real
        r_out_im = [evecs[:, evec_idx].reshape((self.depth, self.width)).real for evec_idx in eigen_vector]
        return r_out_im
    
    
    def plot_rate_map(
        self,
        sr_matrix=None,
        eigen_vectors: Union[int, list, tuple] = None,
        ax: mpl.axes.Axes = None,
        save_path: str = None,
    ):
        if eigen_vectors is None:
            eigen_vectors = random.randint(5, 19)

        if isinstance(eigen_vectors, int):
            rate_map_mat= self.get_rate_map_matrix(sr_matrix, eigen_vector=eigen_vectors)

            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(4, 5))
            make_plot_rate_map(rate_map_mat, ax, "Rate map: Eig" + str(eigen_vectors), "x", "y", "Firing rate")
        else:
            if ax is None:
                f, ax = plt.subplots(1, len(eigen_vectors), figsize=(4 * len(eigen_vectors), 5))
            if isinstance(ax, mpl.axes.Axes):
                ax = [ax,]

            rate_map_mats = self.get_rate_map_matrix(sr_matrix, eigen_vector=eigen_vectors)
            for i, rate_map_mat in enumerate(rate_map_mats):
                make_plot_rate_map(rate_map_mat, ax[i], "Rate map: " + "Eig" + str(eigen_vectors[i]), "x", "y", "Firing rate")
        
        
        if save_path is None:
            pass
        else:
            plt.savefig(save_path, bbox_inches="tight")
            return ax
    
           
    def plot_freq_map(
        self,
        ax: mpl.axes.Axes = None,
        save_path: str = None
    ):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(4,5))

        make_plot_rate_map(self.freq_map.reshape(self.resolution_width, self.resolution_depth), ax, "Frequency Map", "x", "y", "Frequency")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
           
        return ax
    
class RatOnLogarithmicTangent(RatOnTangent):
    
    def __init__(self, agent_name: str = "SR", discount: float = 0.9, threshold: float = 0.000001, lr_td: float = 0.01, room_width: float = 12, room_depth: float = 12, state_density: float = 1, twoD: bool = True, replicable: bool = True, **mod_kwargs):
        super().__init__(agent_name, discount, threshold, lr_td, room_width, room_depth, state_density, twoD, replicable, **mod_kwargs)

    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [
                obs,
            ]

        if len(obs) == 0:
            action = None
        else:
            # Random policy
            action = np.random.uniform(-1,1,3)
            self.next_state = self.obs_to_state(obs[0]) # Only pass in geometric coordinates
            self.freq_map[self.next_state] += 1
            action = np.array(action)
        return action
    
    def obs_to_state(self, pos: ndarray):
        logarithmic_pos = Sphere.logarithmic_map(np.zeros(3), pos)[:2]
        return super().obs_to_state(logarithmic_pos)
    
class RatOnVertical(RatOnTangent):
    
    def __init__(self, agent_name: str = "SR", discount: float = 0.9, threshold: float = 0.000001, lr_td: float = 0.01, room_width: float = 12, room_depth: float = 12, state_density: float = 1, twoD: bool = True, replicable: bool = True, **mod_kwargs):
        super().__init__(agent_name, discount, threshold, lr_td, room_width, room_depth, state_density, twoD, replicable, **mod_kwargs)
        # Modify vertical for even state density
        self.depth = int(2 * self.room_depth * self.state_density)
        self.n_state = int(self.depth * self.width)
        self.obs_history = []
        if twoD:
            self.create_transmat(self.state_density, "2D_env")

    def obs_to_state(self, pos: ndarray):
        # Compute the bin size in both x and y directions
        bin_size_y = self.room_width / (self.width - 1)
        bin_size_z = self.room_depth / (self.depth - 1)

        # Map the observed position to the closest bin indices
        y_index = int(np.round((pos[0] - (-self.room_width / 2)) / bin_size_y))
        z_index = self.depth -1 - int(np.round((pos[1] - (-self.room_depth)) / bin_size_z))

        # Clamp indices to ensure they are within valid bounds
        y_index = np.clip(y_index, 0, self.width-1)
        z_index = np.clip(z_index, 0, self.depth-1)

        # Convert 2D (x_index, y_index) to a single state index in the linear state space
        curr_state = z_index * self.width + y_index

        return curr_state
    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [
                obs,
            ]

        if len(obs) == 0:
            action = None
        else:
            # Random policy
            action = np.random.uniform(-1,1,3)
            self.next_state = self.obs_to_state(obs[0][1:]) # Only pass in geometric coordinates
            self.freq_map[self.next_state] += 1
            action = np.array(action)
        return action
    

