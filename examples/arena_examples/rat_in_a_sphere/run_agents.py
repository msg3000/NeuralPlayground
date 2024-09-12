from neuralplayground.arenas import Sphere
from neuralplayground.agents import Stachenfeld2018, RatInASphere
from neuralplayground.backend import episode_based_training_loop
from neuralplayground.plotting.plot_utils import make_plot_rate_map
from neuralplayground.backend import SingleSim
from neuralplayground.comparison import GridScorer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from projection_agents import RatOnTangent, RatOnLogarithmicTangent

# Global variables
ENV_CLASS = Sphere
TRAINING_LOOP = episode_based_training_loop
SIM_ID = "SR_rat_in_a_sphere"
TRAINING_LOOP_PARAMS = {"t_episode": 1000, "n_episode": 1000}

def run_spherical_agent(n_slices = 36, n_stacks = 36, gravity_mag = 0.15):

    env_params = {"n_stacks": n_stacks,
             "n_slices": n_slices,
             "gravity_mag": gravity_mag,
             "step_size": 0.1}

    agent_class = RatInASphere

    agent_params = {"discount":  0.9,
                "threshold": 1e-6,
                "lr_td":  1e-2,
                "n_slices": n_slices,
                "n_stacks": n_stacks}
    
    sim_spherical = SingleSim(simulation_id = SIM_ID,
                agent_class = agent_class,
                agent_params = agent_params,
                env_class = ENV_CLASS,
                env_params = env_params,
                training_loop = TRAINING_LOOP,
                training_loop_params = TRAINING_LOOP_PARAMS)
    
    sim_spherical.run_sim(f"models/gravity_{gravity_mag}/spherical")


def run_ortho_proj_agent(n_slices = 100, n_stacks = 100, gravity_mag = 0.15):

    env_params = {"n_stacks": n_stacks,
             "n_slices": n_slices,
             "gravity_mag": gravity_mag,
             "step_size": 0.1}

    agent_params = {"room_width" : 2,
                "room_depth" : 2,
                "state_density" : 25.5}

    agent_class = RatOnTangent
    
    sim_orth = SingleSim(simulation_id = SIM_ID,
                agent_class = agent_class,
                agent_params = agent_params,
                env_class = ENV_CLASS,
                env_params = env_params,
                training_loop = TRAINING_LOOP,
                training_loop_params = TRAINING_LOOP_PARAMS)
    
    sim_orth.run_sim(f"models/gravity_{gravity_mag}/orth_proj")

def run_log_proj_agent(n_slices = 100, n_stacks = 100, gravity_mag = 0.15):

    env_params = {"n_stacks": n_stacks,
             "n_slices": n_slices,
             "gravity_mag": gravity_mag,
             "step_size": 0.1}
   
    agent_params = {"room_width" : 3,
                "room_depth" : 3,
                "state_density" : 12}

    agent_class = RatOnLogarithmicTangent
    
    sim_orth = SingleSim(simulation_id = SIM_ID,
                agent_class = agent_class,
                agent_params = agent_params,
                env_class = ENV_CLASS,
                env_params = env_params,
                training_loop = TRAINING_LOOP,
                training_loop_params = TRAINING_LOOP_PARAMS)
    
    sim_orth.run_sim(f"models/gravity_{gravity_mag}/log_proj")



if __name__ == "__main__":

    # Read in gravity param
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--gravity_mag', type=float, default=0.15, help='Magnitude of gravity')
    args = parser.parse_args()
    gravity_mag = args.gravity_mag

    print("========= Beginning spherical agent =========== ")
    run_spherical_agent(gravity_mag=gravity_mag)
    print("========= Spherical agent complete =========== ")
    print("========= Beginning orthogonal projection agent =========== ")
    run_ortho_proj_agent(gravity_mag=gravity_mag)
    print("========= Orthogonal projection agent complete =========== ")
    print("========= Beginning logarithmic projection agent =========== ")
    run_log_proj_agent(gravity_mag=gravity_mag)
    print("========= Logarithmic projection agent complete =========== ")












