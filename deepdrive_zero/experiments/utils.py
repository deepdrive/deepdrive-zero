import sys

import gym

from spinup.utils.test_policy import load_policy_and_env, run_policy


def run(train_fn, env_config, net_config=None):
    mode = '--train' if not sys.argv[1:] else sys.argv[1]
    if mode == '--train':
        train_fn()
    elif mode == '--play':
        player.start(env_config=env_config)
    elif mode == '--test':
        if not sys.argv[2:]:
            model_path = get_latest_model_path()  # TODO
        else:
            model_path = sys.argv[2]
        env = gym.make(env_config['env_name'])
        env.configure_env(env_config)
        _, get_action = load_policy_and_env(fpath=model_path, deterministic=True,
                                            env=env, net_config=net_config)
        run_policy(env, get_action)