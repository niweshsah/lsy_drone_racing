# Autonomous Drone Racing Project Course
install dependancy & acados

activate virtual env at first

    mamba activate race

## Vanilla MPCC
### Level0: average 3.46s
    python ./scripts/sim.py --config level0.toml --controller mpcc.py --n_runs 10 -g

## MPCC with trajectory translation (Final)
### Level2 (stable): success rate 91% average 6.05s
    python ./scripts/sim.py --config level2.toml --controller mpcc_traj_translation_simplified.py --n_runs 10 -g
### Level2 (leader board): success rate 48% average 4.70s
    python ./scripts/sim.py --config level2.toml --controller mpcc_traj_translation_real_fast.py --n_runs 10 -g

## Experiments Results
### Pareto front of trials in Bayesian Optimiation (100 runs/trial)
<img width="500" src="results/Pareto_front.png">

### Violin plots of some sets of parameters from Bayesian Optimization
successful rollouts from 100 runs

<img width="800" src="results/violin_plot.png">

### Video Demostrations (under ./results)
Video of realworld deployment using the fast parameter set. Lap time 5.75 sec.

- 🎬 results/real_5_75sec.mp4

Simulation Demos using stable parameter set. Average lap time 6.05 sec.

- 🎬 results/sim_demos.mp4

## Method Illustrations
### Gate Translation Visualization
Stick to one planned trajectory, shift the whole trajectory toward the gate offset, then smooth the translation using Gaussian weights.

<img width="400" src="results/gate_translation.svg">

### Obstacle Cost Design
Red area: Gaussian shaped obstacle cost

Grey area: Suppress q_c, μ when approaching obstacles: force the drone to slow down, allow deviation from original trajectory

<img width="400" src="results/obst_cost.svg">

### Gate Translation Demo Snapshots
Examples of trajectory translation under different gate configurations.

<img width="400" height="400" src="results/translation_demo_gate_1.png">
<img width="400" height="400" src="results/translation_demo_gate_2.png">

<img width="400" height="400" src="results/translation_demo_gate_3.png">
<img width="400" height="400" src="results/translation_demo_gate_4.png">


## Reinforcement Learning:
### Level2: success rate 65% average 4.91s
    python ./scripts/sim.py --config level2.toml --controller rl_controller.py --n_runs 10 -g
### Level0: 2.99s
    python ./scripts/sim.py --config level0.toml --controller rl_controller_fast.py --n_runs 10 -g
