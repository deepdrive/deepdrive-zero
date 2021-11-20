import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """Last train run where we did speed => collision => comfort => yield 
phases of fine-tuning ended up in a local minima where yield was only
happening correctly about half the time (due to not forgetting old 
straight agent tendency to yield). So trying to start from scratch with
high level goals like waypoint finding, avoiding collisions, and yielding
incented in the first phase, then more fine-grain things like gforce in
the second phase. ALSO: We are moving from 100ms steps to 200ms steps which
will help exploration, learning efficiency, and create less unrealistic and
jerky actions."""

env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.02,
    speed_reward_coeff=0.50,
    gforce_threshold=None,
    end_on_lane_violation=False,
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
)

net_config = dict(
    hidden_units=(256, 256),
    activation=torch.nn.Tanh
)

eg = ExperimentGrid(name=experiment_name)
eg.add('env_name', env_config['env_name'], '', False)
# eg.add('seed', 0)
# eg.add('resume', '/home/c2/src/tmp/spinningup/data/intersection_2_agents_fine_tune_add_left_yield2/intersection_2_agents_fine_tune_add_left_yield2_s0_2020_03-23_22-40.11')
# eg.add('reinitialize_optimizer_on_resume', True)
# eg.add('num_inputs_to_add', 0)
# eg.add('pi_lr', 3e-6)
# eg.add('vf_lr', 1e-5)
# eg.add('boost_explore', 5)
eg.add('epochs', 8000)
eg.add('steps_per_epoch', 500)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)