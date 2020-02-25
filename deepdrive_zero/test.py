import os
import pkgutil
import sys

from deepdrive_zero.logs import log
import deepdrive_zero.physics.collision_detection
import deepdrive_zero.physics.bike_model
import deepdrive_zero.envs.env
import deepdrive_zero.utils

MODULES_TO_TEST = [
    deepdrive_zero.physics.collision_detection,
    deepdrive_zero.physics.bike_model,
    deepdrive_zero.envs.env,
    deepdrive_zero.utils,
]


def run_module(current_module):
    log.info('Running all tests')
    num = 0
    for attr in dir(current_module):
        if attr.startswith('test_'):
            num += 1
            log.info('Running ' + attr)
            getattr(current_module, attr)()
            log.success(f'Test: {attr} ran successfully')
    return num


def run_tests():
    current_module = sys.modules[__name__]
    if len(sys.argv) > 1:
        test_case = sys.argv[1]
        log.info('Running ' + test_case)
        getattr(current_module, test_case)()
        num = 1
        log.success(f'{test_case} ran successfully!')
    else:
        num = run_module(current_module)
        for module in MODULES_TO_TEST:
            num += run_module(module)

    log.success(f'{num} tests ran successfully!')


def main():
    run_tests()


# Why not use a test runner?
#  Because this allows easily debugging tests
#  and is simpler anyway.


if __name__ == '__main__':
    main()