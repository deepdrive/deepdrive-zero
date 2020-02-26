import os

from spinup.utils.test_policy import load_policy, run_policy

# Register custom envs
import gym_match_input_continuous
import deepdrive_zero
import gym

TEST_STATIC_OBSTACLE = False



if TEST_STATIC_OBSTACLE:
    _, get_action = load_policy(
        '/home/c2/src/spinningup/data/dd2d-ppo-intersection/dd2d-ppo-intersection_s0',
        use_model_only=False)

    env = gym.make('deepdrive-2d-static-obstacle-no-g-pen-v0')
else:
    p = '/home/c2/src/spinningup/data/dd0-no-contraint-g-fine-tune-w-constraint-v0/dd0-no-contraint-g-fine-tune-w-constraint-v0_s0_2020_02-25_12-37.15/best_EpRet/2020_02-25_16-08.24'
    if 'no-end-g' in p or 'no-contraint-g' in p or 'no-g' in p:
        os.environ['END_ON_HARMFUL_GS'] = '0'
    if 'no-contraint' in p and 'w-constraint' not in p:
        os.environ['CONSTRAIN_CONTROLS'] = '0'
    _, get_action = load_policy(p, use_model_only=False, deterministic=True)
    # env = gym.make('deepdrive-2d-intersection-v0')
    env = gym.make('deepdrive-2d-intersection-w-gs-allow-decel-v0')

# env.unwrapped.physics_steps_per_observation = 1
run_policy(env, get_action)
