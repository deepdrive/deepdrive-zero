from setuptools import setup

setup(name='deepdrive_zero',
      version='0.0.1',
      # And any other dependencies we need
      install_requires=['gym', 'numpy', 'scipy', 'arcade', 'loguru',
                        'python-box', 'numba', 'matplotlib',
                        'retry', 'dataclasses']
      )
