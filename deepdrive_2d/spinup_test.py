from spinup.utils.test_policy import load_policy, run_policy

# Register custom envs
import gym_match_input_continuous
import deepdrive_2d
import gym

_, get_action = load_policy('/home/c2/src/spinningup/data/dd2d-ppo/dd2d-ppo_s0')
env = gym.make('deepdrive-2d-one-waypoint-v0')
env.unwrapped.enable_render()
run_policy(env, get_action)