import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.plotting.plot_utils import make_plot_trajectories_3d

class Sphere(Environment):
    """
    Methods (Some in addition to Environment class)
    ----------
    __init__(self, environment_name="2DEnv", **env_kwargs):
        Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and moves
        the agent in the supplied direction with a fixed step size
    plot_trajectory(self, history_data=None, ax=None):
        Plot the Trajectory of the agent in the environment. In addition to environment class.
    validate_action(self, pre_state, action, new_state):
        Check if the new state is crossing the bounds of the lower hemisphere.
    render(self, history_length=30):
        Render the environment live through iterations as in OpenAI gym.
    exponential_map(p, v):
        Apply the exponential map to a point v in the tangent space of p.
    to_spherical(point):
        Given a point on the unit sphere returns its spherical coordiates.
    to_extrinsic(theta, phi):
        Convert intrinsic representation as spherical coordinates to 3D coordinates.
    project_to_tangent(point, vector):
        Computes orthogonal projection of vector onto point.

    Attributes (Some in addition to the Environment class)
    ----------
    state: ndarray
        Contains the x, y coordinate of the position and head direction of the agent (will be further developed)
        head_direction: ndarray
                Contains the x and y Coordinates of the position
        position: ndarray
                Contains the x and y Coordinates of the position
    history: list of dicts
        Saved history over simulation steps (action, state, new_state, reward, global_steps)
    global_steps: int
        Counter of the number of steps in the environment
    arena_x_limits: float
        Size of the environment in the x direction (width)
    arena_y_limits: float
        Size of the environment in the y direction (depth)
    room_width: int
        Size of the environment in the x direction
    room_depth: int
        Size of the environment in the y direction
    metadata: dict
        Dictionary containing the metadata
    state_dims_labels: list
        List of the labels of the dimensions of the state
    observation_space: gym.spaces
        specify the range of observations as in openai gym
    action_space: gym.spaces
        specify the range of actions as in openai gym
    wall_list: list
        List of the walls in the environment
    observation: ndarray
        Fully observable environment, make_observation returns the state
        Array of the observation of the agent in the environment (Could be modified as the environments are evolves)
    agent_step_size: float
         Size of the step when executing movement, agent_step_size*global_steps will give
         a measure of the total distance traversed by the agent
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
        n_stacks: int
            Number of even discrete intervals to discretise polar angles
        n_slices: int
            Number of even discrete intervals to discretise azimuthal angles
        
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
        return vector - proj
    
    @staticmethod
    def to_spherical(point: np.ndarray):
        phi = np.arccos(point[2]) 
        theta = np.arctan2(point[1], point[0])
        theta = theta + 2*np.pi if theta < 0 else theta
        return phi, theta
    
    @staticmethod
    def to_extrinsic(theta: float, phi: float):
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

    
    def normalize_state(self, current_state):
        phi, theta = self.to_spherical(current_state)
        dphi, dtheta = np.pi/(2*(self.n_stacks-1)), 2*np.pi/(self.n_slices-1)
        theta = round(theta/dtheta) * dtheta
        phi = np.pi/2 + round((phi - np.pi/2)/dphi) * dphi
        return (self.to_extrinsic(theta, phi), (phi, theta))

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        """Reset the environment variables

        Parameters
        ----------
        random_state: bool
            If True, sample a new position uniformly within the arena, use default otherwise
        custom_state: np.ndarray
            If given, use this array to set the initial state

        Returns
        ----------
        observation: ndarray
            Because this is a fully observable environment, make_observation returns the state of the environment
            Array of the observation of the agent in the environment (Could be modified as the environments are evolves)

        self.state: ndarray (2,)
            Vector of the x and y coordinate of the position of the animal in the environment
        """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        if random_state:
            self.state = np.asarray(
                [
                    np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                    np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1]),
                ]
            )
        else:
            self.state = (np.asarray([0,0,-1]), self.to_spherical(np.array([0,0,-1])))

        if custom_state is not None:
            self.state = custom_state
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    

    def step(self, action = None, normalize_step: bool = True):
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
            action = self.project_to_tangent(self.state[0], action)


            if normalize_step:
                action = action / np.linalg.norm(action)

            # Take a unit step along tangent vector in tangent plane --> action and project back to sphere
            sphere_proj = self.exponential_map(self.state[0], 0.1*action)
            
            # Approximate to discretised space
            new_state = self.normalize_state(sphere_proj)
            
            new_state, valid_action = self.validate_action(self.state, action, new_state)

        self.state = new_state
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
       
        if new_state[0][2] > 0:
            return pre_state, True
        
        return new_state, False

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
            z = []
            for i, s in enumerate(state_history):
                # if i % plot_every == 0:
                #     if i + plot_every >= len(state_history):
                #         break
                x.append(s[0][0])
                y.append(s[0][1])
                z.append(s[0][2])
            ax = make_plot_trajectories_3d(np.asarray(x), np.asarray(y), np.asarray(z), ax, plot_every)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax

    def render(self, history_length=30, save_dir = None, frame_num = 1):
        """Render the environment live through iterations as in OpenAI gym"""

        f = plt.figure()
      
        ax = plt.axes(projection = "3d")  
        ax.view_init(90, -90)
        phi = np.linspace(np.pi/2, np.pi, self.n_stacks)
        theta = np.linspace(0, 2*np.pi, self.n_slices)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        ax.plot_surface(x,y,z, color ='blue', alpha = 0.5)
        canvas = FigureCanvas(f)
        history = self.history[:history_length]
        ax = self.plot_trajectory(history_data=history, ax=ax)
    
        if save_dir is not None:
            plt.savefig(save_dir)
            plt.close(f)
        else:
            plt.show()
        # canvas.draw()
        # image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        # image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
        # print(image.shape)
        # cv2.imshow("2D_env", image)
        # cv2.waitKey(10)