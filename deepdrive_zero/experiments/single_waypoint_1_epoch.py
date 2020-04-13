import os

from deepdrive_zero.constants import FPS
from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """
Simplest environment. Should reach 98% average angle accuracy in about 5 minutes

Angle accuracy graph:
https://photos.app.goo.gl/GvGfa2ibgAC6V1f49
https://i.imgur.com/blv5WdY.jpg

Full results:
https://docs.google.com/spreadsheets/d/1nQb33naseYJ7-gFW1YyzHb6Mffx-H63gquryD_WibTk/edit?usp=sharing
"""


env_config = dict(
    env_name='deepdrive-2d-one-waypoint-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=3.3e-5,
    gforce_penalty_coeff=0.006 * 5,
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.02,
    speed_reward_coeff=0.50,
    gforce_threshold=None,
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
)

net_config = dict(
    hidden_units=(64, 64),
    activation=torch.nn.Tanh
)

eg = ExperimentGrid(name=experiment_name)
eg.add('env_name', env_config['env_name'], '', False)
pso = env_config['physics_steps_per_observation']
effective_horizon_seconds = 10
eg.add('gamma', 1 - pso / (effective_horizon_seconds * FPS))  # Lower gamma so seconds of effective horizon remains at 10s with current physics steps = 12 * 1/60s * 1 / (1-gamma)
eg.add('epochs', 1)
eg.add('steps_per_epoch', 500)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config,
              num_eval_episodes=10)