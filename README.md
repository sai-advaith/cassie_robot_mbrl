# Cassie project
Export `PYTHONPATH` to `cassie_project/` before running any code in this repository.

## Model Learning

This section primarily deals with `model_learning/` directory of the repository i.e. learning the dynamics of Cassie.

- The code to train the transformer model with default hyperparameters is included in `model_learning_transformer.py`. The code includes support for [wandb](https://wandb.ai/) to visualize and log results.

- To change any hyperparameter/get more information about specific arguments use `python3 model_learning_transformer.py -h` command.

- `model_learning/model_learning_sweep.py` is example code to perform hyperparameter sweep with grid search using [wandb](https://wandb.ai/).

- `transformer.py`, `representation.py`, and `positional_encoding.py` include code for different architectures used to learn robot dynamics.

## Planner
This section primarily deals with `planner/` directory of the repository i.e. the trajectory planning of Cassie.

- The code to run the gradient-CEM planner for Cassie with default hyperparameters is included in `grad_planner.py`.

- To change any hyperparameter/get more information about specific arguments use `python3 grad_planner.py -h` command.

- `gradient_cem.py` is the planner currently used with `grad_planner.py` and `cassie_dynamics.py` is the retraining and state prediction module used with `grad_planner.py`.
- `reward.py` is the reward function used during the planning phase
- `mpc_model_learning.py`, `mpc_cem.py`, `mpc_cem_gd.py`, and `cem_gd_planner.py` are implementations for other types of planners like CEM and CEM-gd.

## Utils
This section primarily deals with `utils/` directory of the repository i.e. the utility code used in planner, model learning, and analytical gradient collection.

- Code to collect analytical gradient and specific action phase included in `utils/get_analytic_gradient.py` and `utils/get_analytic_phase.py`

- Model learning and planner utility included in `model_learning_utility.py` and `planner_utility.py`

## Simulator
This section deals with the simulator code wrapper for Cassie's mujoco simulator.

- Simulator wrapper for mujoco simulator included in `simulator/evaluate_expert.py`. This includes code to execute particular action and send the PD targets to mujoco interface.
- `actor_iter1143.pt` is the LSTM based expert policy for Cassie.

#### Note
The states and actions for training the model have to be collected manually using code in `RSS-2020-learning-memory-based-control/simulator_code` in aws instance `cassie_project_gpu`. Refer to `README.md` in that directory for specific instructions.

Email [advaith.maddipatla@gmail.com](mailto:advaith.maddipatla@gmail.com) for any questions
