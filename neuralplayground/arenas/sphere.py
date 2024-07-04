import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.plotting.plot_utils import make_plot_trajectories

class Sphere(Environment):
    """
    """

    def __init__(
        self,
        environment_name: str = "sphere",
        n_stacks: int = 16,
        n_slices: int = 16,
        **env_kwargs,
    ):
        """Initialise the class

        Parameters
        ----------
        environment_name: str
            Name of the specific instantiation of the Simple2D class
        time_step_size: float
            time_step_size * global_steps will give a measure of the time in the experimental setting
        agent_step_size: float
            Step size used when the action is a direction in x,y coordinate (normalize false in step())
            Agent_step_size * global_step_number will give a measure of the distance in the experimental setting
        arena_x_limits: float
            Size of the environment in the x direction
        arena_y_limits: float
            Size of the environment in the y direction
        """
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.n_stacks = n_stacks
        self.n_slices = n_slices
        self.state_dims_labels = ["x_pos", "y_pos", "z_pos"]
        self.reset()

    @staticmethod
    def exponential_map(p: np.ndarray, v: np.ndarray):

        v_norm = np.linalg.norm(v)
        return np.cos(v_norm)*p + np.sin(v_norm)*v/v_norm
    
    @staticmethod
    def project_to_tangent(point: np.ndarray, vector: np.ndarray):

        proj = np.dot(point, vector) * point
        return point - proj
    
    @staticmethod
    def to_spherical(point: np.ndarray):

        phi = np.arccos(point[2])
        theta = np.arctan2(point[1], point[0])
        theta = theta + 2*np.pi if theta < 0 else theta
        return theta, phi
    
    @staticmethod
    def to_extrinsic(theta: float, phi: float):
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

    
    def normalize_state(self, current_state):
        theta, phi = self.to_spherical(current_state)
        theta = round(theta*self.n_stacks/2*np.pi) * 2*np.pi / self.n_stacks
        phi = round(theta*self.n_slices/2*np.pi) * 2*np.pi / self.n_slices
        return self.to_extrinsic(theta, phi)

    

    def step(self, action: None, normalize_step: bool = True):
        """Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.

        Parameters
        ----------
        action: None or ndarray (2,)
            Array containing the action of the agent, in this case the delta_x and detla_y increment to position
        normalize_step: bool
            If true, the action is normalized to have unit size, then scaled by the agent step size

        Returns
        -------
        reward: float
            The reward that the animal receives in this state
        new_state: ndarray
            Update the state with the updated vector of coordinate x and y of position and head directions respectively
        observation: ndarray
            Array of the observation of the agent in the environment
        """
        if action is None:
            new_state = self.state
        else:
            # Project random direction onto instantaneous tangent plane
            action = self.project_to_tangent(self.state, action)

            if normalize_step:
                action = action / np.linalg.norm(action)

            # Take a unit step along tangent vector in tangent plane --> action and project back to sphere
            sphere_proj = self.exponential_map(self.state, action)

            # Approximate to discretised space
            new_state = self.normalize_state(sphere_proj)
            
            new_state, valid_action = self.validate_action(self.state, action, new_state)

        self.state = np.asarray(new_state)
        observation = self.make_observation()
        self._increase_global_step()
        reward = self.reward_function(action, self.state)
        transition = {
            "action": action,
            "state": self.state,
            "next_state": new_state,
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(transition)
        return observation, new_state, reward

    def validate_action(self, pre_state, action, new_state):
        """Check if the new state is crossing any walls in the arena.

        Parameters
        ----------
        pre_state : (2,) 2d-ndarray
            2d position of pre-movement
        new_state : (2,) 2d-ndarray
            2d position of post-movement

        Returns
        -------
        new_state: (2,) 2d-ndarray
            corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
            wall, new_state will be corrected to a valid place without crossing the wall
        crossed_wall: bool
            True if the change in state crossed a wall and was corrected
        """
       
        if new_state[2] > 0:
            return pre_state, True
        
        return new_state

    def plot_trajectory(
        self,
        history_data: list = None,
        ax=None,
        return_figure: bool = False,
        save_path: str = None,
        plot_every: int = 1,
    ):
        """Plot the Trajectory of the agent in the environment

        Parameters
        ----------
        history_data: list of interactions
            if None, use history data saved as attribute of the arena, use custom otherwise
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
            axis from subplot from matplotlib where the trajectory will be plotted.
        return_figure: bool
            If true, it will return the figure variable generated to make the plot
        save_path: str, list of str, tuple of str
            saving path of the generated figure, if None, no figure is saved

        Returns
        -------
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
            Modified axis where the trajectory is plotted
        f: matplotlib.figure
            if return_figure parameters is True
        """
        # Use or not saved history
        if history_data is None:
            history_data = self.history

        # Generate Figure
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Make plot of positions
        if len(history_data) != 0:
            state_history = [s["state"] for s in history_data]
            x = []
            y = []
            for i, s in enumerate(state_history):
                # if i % plot_every == 0:
                #     if i + plot_every >= len(state_history):
                #         break
                x.append(s[0])
                y.append(s[1])
            ax = make_plot_trajectories(self.arena_limits, np.asarray(x), np.asarray(y), ax, plot_every)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax

    def render(self, history_length=30):
        """Render the environment live through iterations as in OpenAI gym"""
        f = plt.figure()
        ax = plt.axes(projection = "3d")
        phi = np.linspace(0, 2*np.pi, self.n_stacks)
        theta = np.linspace(0, np.pi, self.n_slices)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        ax.plot_wireframe(x,y,z, color ='green')
        canvas = FigureCanvas(f)
        history = self.history[-history_length:]
        ax = self.plot_trajectory(history_data=history, ax=ax)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
        print(image.shape)
        cv2.imshow("2D_env", image)
        cv2.waitKey(10)
