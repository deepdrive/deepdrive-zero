import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """Attempting to speed up reduction in g-force, jerk, and lane 
violations"""

# jerk_penalty_coeff:
# acceptable 6g / 10s trampoline
# current - 600 m/s^3 to 1k,2k
#  It is reported[8] that most passengers rate a vertical jerk of 2.0 m/s3 as
#  acceptable and 6.0 m/s3 as intolerable.
#  For hospitals, 0.7 m/s3 is the recommended limit.
# https://en.wikipedia.org/wiki/Jerk_(physics)#In_motion_control

# Above appears to be way below what we measure and what is reported here as
# acceptable. https://iopscience.iop.org/article/10.1088/0143-0807/37/6/065008/pdf
# Importantly they depict the threshold
# for admissible acceleration onset or jerk as j = 15g/s or ~150m/s^3.

# So if in three seconds at 1aps
# t: 0       1      2
# v: 10  ->  20  -> 10
# a:    10   -> -10  = 1g
# j:         20

# Or if at 2aps
# t: 0       .5     1
# v: 10  ->  20  -> 10
# a:    20   -> -20  = 2g = 20m/s^2
# j:         40

# And 5aps - 12 physics frames per second
# t: 0       .2     .4
# v: 10  ->  20  -> 10
# a:    50   -> -50  = 5g
# j:         100


env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=6.7e-5,  # 2 * 0.20 / (60*100)
    gforce_penalty_coeff=0.12,  # 2 * 0.06
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.04,  # 2 * 0.04
    speed_reward_coeff=0.50,
    gforce_threshold=None,
    end_on_lane_violation=False,
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,  # 48 for 1 second
)

net_config = dict(
    hidden_units=(256, 256),
    activation=torch.nn.Tanh
)

eg = ExperimentGrid(name=experiment_name)
eg.add('env_name', env_config['env_name'], '', False)
# eg.add('seed', 0)
eg.add('resume', '/home/c2/dd0-data/snaphot11/intersection_2_agents_fine_tune_add_left_yield_from_scratch_resume/intersection_2_agents_fine_tune_add_left_yield_from_scratch_resume_s0_2020_03-29_00-28.47')
# eg.add('reinitialize_optimizer_on_resume', True)
# eg.add('num_inputs_to_add', 0)
eg.add('pi_lr', 3e-6)
eg.add('vf_lr', 1e-5)
# eg.add('boost_explore', 5)
eg.add('epochs', 8000)
eg.add('steps_per_epoch', 32000)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)