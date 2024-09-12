#!/bin/bash

#SBATCH --job_name=rat_sim
#SBATCH --output=/home-mscluster/mgoolam/NeuralPlayground/examples/arena_examples/rat_in_a_sphere/out.txt
#SBATCH --ntasks = 1
#SBATCH --partition=stampede
#SBATCH -N 1

python3 ~/NeuralPlayground/examples/arena_examples/rat_in_a_sphere/run_agents.py --gravity_mag 0.15
