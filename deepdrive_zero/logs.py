from loguru import logger as log
from os.path import dirname, realpath, join
import os

ROOT_DIR = dirname(realpath(__file__))
LOG_DIR = join(ROOT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log.add(join(LOG_DIR, 'deepdrive-zero-{time}.log'))