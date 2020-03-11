import os

from spinup.utils.test_policy import load_policy, run_policy, \
    load_policy_and_env

# Register custom envs
import gym_match_input_continuous
import deepdrive_zero
import gym

TEST_STATIC_OBSTACLE = False


# TODO: Move configs from env to python that train and test both use

if TEST_STATIC_OBSTACLE:
    _, get_action = load_policy(
        '/home/c2/src/spinningup/data/dd2d-ppo-intersection/dd2d-ppo-intersection_s0',
        use_model_only=False)

    env = gym.make('deepdrive-2d-static-obstacle-no-g-pen-v0')
else:
    p = '/home/c2/src/tmp/spinningup/data/deepdrive-2d-intersection-no-constrained-controls-example/deepdrive-2d-intersection-no-constrained-controls-example_s0_2020_03-10_13-14.50/best_HorizonReturn/2020_03-11_11-36.27'
    if 'no-end-g' in p or 'no-contraint-g' in p or 'no-g' in p or 'no-constrain' in p:
        os.environ['END_ON_HARMFUL_GS'] = '0'
        os.environ['GFORCE_PENALTY_COEFF'] = '0'
        os.environ['JERK_PENALTY_COEFF'] = '0'
    if 'no-constrain' in p:
        os.environ['CONSTRAIN_CONTROLS'] = '0'
    if 'delta-controls' in p or 'deepdrive-2d-intersection-no-g-or-jerk2' in p:
        os.environ['EXPECT_NORMALIZED_ACTION_DELTAS'] = '1'
    else:
        os.environ['EXPECT_NORMALIZED_ACTION_DELTAS'] = '0'

    if 'one-waypoint' in p:
        env_name = 'deepdrive-2d-one-waypoint-v0'
    else:
        env_name = 'deepdrive-2d-intersection-w-gs-allow-decel-v0'
    _, get_action = load_policy_and_env(p, deterministic=True)
    # env = gym.make('deepdrive-2d-intersection-v0')
    env = gym.make(env_name)
    env.configure_env()

# env.unwrapped.physics_steps_per_observation = 1
run_policy(env, get_action)
