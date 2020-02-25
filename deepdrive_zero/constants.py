import os
import sys
from os.path import join

from box import Box

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_MARGIN = 50
MAP_WIDTH_PX = SCREEN_WIDTH - SCREEN_MARGIN * 2
MAP_HEIGHT_PX = SCREEN_HEIGHT - SCREEN_MARGIN * 2
PLAYER_TURN_RADIANS_PER_KEYSTROKE = 1 / 128
SCREEN_TITLE = 'deepdrive-zero'
CHARACTER_SCALING = 1/4
USE_VOYAGE = True
PX_PER_M = 19.559472386854488

DIR = os.path.dirname(os.path.realpath(__file__))

if USE_VOYAGE:
    # https://www.convert-me.com/en/convert/acceleration    /ssixtymph_1.html?u=ssixtymph_1&v=7.4
    # Pacifica Hybrid Max accel m/s^2 = 3.625
    MAX_METERS_PER_SEC_SQ = 3.625  # 9.807 / 10 - 0.1
    VEHICLE_WIDTH = 2.300675555555556
    VEHICLE_HEIGHT = 5.17652
    VEHICLE_PNG = join(DIR, 'images/voyage-van-up.png')
else:
    MAX_METERS_PER_SEC_SQ = 4.79
    VEHICLE_PNG = join(DIR, 'images/tesla-up.png')
    # TODO: Run player to determine width and height

MAX_PIXELS_PER_SEC_SQ = MAX_METERS_PER_SEC_SQ * PX_PER_M
TESLA_LENGTH = 4.694
VOYAGE_VAN_LENGTH = 5.17652
MAP_IMAGE = join(DIR, 'images/map.png')

MAX_BRAKE_G = 1
G_ACCEL = 9.80665
CONTINUOUS_REWARD = True
GAME_OVER_PENALTY = -1
IS_DEBUG_MODE = getattr(sys, 'gettrace', None)
CACHE_NUMBA = True