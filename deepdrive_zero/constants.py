import math
import os
import sys
from os.path import join

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

# Agents and waypoints will be spawned within margin and origin is at margin,
# but agents can travel into the margin.
SCREEN_MARGIN = 50
MAP_WIDTH_PX = SCREEN_WIDTH - SCREEN_MARGIN * 2
MAP_HEIGHT_PX = SCREEN_HEIGHT - SCREEN_MARGIN * 2
PLAYER_TURN_RADIANS_PER_KEYSTROKE = 2 * math.pi / 256
SCREEN_TITLE = 'deepdrive-zero'
CHARACTER_SCALING = 1/4
USE_VOYAGE = True
TESLA_LENGTH = 4.694
TESLA_LENGTH_PX = 368
VOYAGE_VAN_LENGTH = 5.17652
VOYAGE_VAN_LENGTH_PX = 405
PX_PER_M = VOYAGE_VAN_LENGTH_PX / VOYAGE_VAN_LENGTH * CHARACTER_SCALING
SCREEN_WIDTH_METERS = SCREEN_WIDTH / PX_PER_M

DIR = os.path.dirname(os.path.realpath(__file__))

if USE_VOYAGE:
    MAX_METERS_PER_SEC_SQ = 3.625  # 0-60 in 7.4s
    VEHICLE_WIDTH = 2.300675555555556
    VEHICLE_LENGTH = VOYAGE_VAN_LENGTH
    VEHICLE_PNG = join(DIR, 'images/voyage-van-up.png')
else:
    MAX_METERS_PER_SEC_SQ = 4.79
    VEHICLE_WIDTH = 2.09  # Include side mirrors
    VEHICLE_LENGTH = TESLA_LENGTH  # Include side mirrors
    VEHICLE_PNG = join(DIR, 'images/tesla-up.png')


MAX_PIXELS_PER_SEC_SQ = MAX_METERS_PER_SEC_SQ * PX_PER_M


MAP_IMAGE = join(DIR, 'images/map.png')

MAX_BRAKE_G = 1
G_ACCEL = 9.80665
CONTINUOUS_REWARD = True
GAME_OVER_PENALTY = -1
IS_DEBUG_MODE = getattr(sys, 'gettrace', None)
CACHE_NUMBA = True

MAX_STEER_CHANGE_PER_SECOND = 0.03 * 60  # TODO: Make this vehicle make/model based, currently based on playing in sim
MAX_ACCEL_CHANGE_PER_SECOND = 0.04 * 60  # TODO: Make this vehicle make/model based, currently based on playing in sim
MAX_BRAKE_CHANGE_PER_SECOND = 0.06 * 60  # TODO: Make this vehicle make/model based, currently based on playing in sim
STEERING_RANGE = math.pi / 6  # Standard for sharp turning vehicles (about 33 deg)  # TODO: Make this vehicle make/model based
MIN_STEER = -STEERING_RANGE/2
MAX_STEER = STEERING_RANGE/2
RIGHT_HAND_TRAFFIC = True
FPS = 60
PARTIAL_PHYSICS_STEP = 'partial_physics_step'
COMPLETE_PHYSICS_STEP = 'complete_physics_step'

COMFORTABLE_STEERING_ACTIONS = {
    0: 'IDLE',  # Zero steer, zero accel

    # Based on simple heuristic solution to the single waypoint env where
    # steering set to 10% of the delta between current and desired heading
    # smoothly steers towards the waypoint. Here we expect the net to use
    # the large and small steering to comfortably set the required initial
    # angle that can then be decayed in the same way to reach the waypoint.
    1: 'DECAY_STEERING',

    # Small steering adjustments
    2: 'SMALL_STEER_LEFT',
    3: 'SMALL_STEER_RIGHT',

    # Largest comfortable left at current velocity
    4: 'LARGE_STEER_LEFT',
    5: 'LARGE_STEER_RIGHT',

    # TODO: e-steer (max steer without losing traction)
}

COMFORTABLE_ACTIONS = {
    0: 'IDLE',
    1: 'DECAY_STEERING_MAINTAIN_SPEED',
    2: 'DECAY_STEERING_DECREASE_SPEED',
    3: 'DECAY_STEERING_INCREASE_SPEED',
    4: 'SMALL_STEER_LEFT_MAINTAIN_SPEED',
    5: 'SMALL_STEER_LEFT_DECREASE_SPEED',
    6: 'SMALL_STEER_LEFT_INCREASE_SPEED',
    7: 'SMALL_STEER_RIGHT_MAINTAIN_SPEED',
    8: 'SMALL_STEER_RIGHT_DECREASE_SPEED',
    9: 'SMALL_STEER_RIGHT_INCREASE_SPEED',
    10: 'LARGE_STEER_LEFT_MAINTAIN_SPEED',
    11: 'LARGE_STEER_LEFT_DECREASE_SPEED',
    12: 'LARGE_STEER_LEFT_INCREASE_SPEED',
    13: 'LARGE_STEER_RIGHT_MAINTAIN_SPEED',
    14: 'LARGE_STEER_RIGHT_DECREASE_SPEED',
    15: 'LARGE_STEER_RIGHT_INCREASE_SPEED',
    # TODO: Maintain speed (PID) * steer(decay, LR small steer, LR large steer)
    # TODO: Decrease speed comfort (PID) * steer(decay, LR small steer, LR large steer) - brake + 0 accel
    # TODO: Increase speed comfort (PID) * steer(decay, LR small steer, LR large steer)
    # Total safe = 5 steer * 3 accel + idle = 16

    # TODO: Increase speed (max without losing traction)
    # TODO: Brake emergency stop
}

COMFORTABLE_ACTIONS_MAINTAIN_SPEED = {1, 4, 7, 10, 13}
COMFORTABLE_ACTIONS_DECREASE_SPEED = {2, 5, 8, 11, 14}
COMFORTABLE_ACTIONS_INCREASE_SPEED = {3, 6, 9, 12, 15}
COMFORTABLE_ACTIONS_DECAY_STEERING = {1, 2, 3}
COMFORTABLE_ACTIONS_SMALL_STEER_LEFT = {4, 5, 6}
COMFORTABLE_ACTIONS_SMALL_STEER_RIGHT = {7, 8, 9}
COMFORTABLE_ACTIONS_LARGE_STEER_LEFT = {10, 11, 12}
COMFORTABLE_ACTIONS_LARGE_STEER_RIGHT = {13, 14, 15}
COMFORTABLE_ACTIONS_LARGE_STEER = COMFORTABLE_ACTIONS_LARGE_STEER_LEFT | COMFORTABLE_ACTIONS_LARGE_STEER_RIGHT

ACTIONS = COMFORTABLE_STEERING_ACTIONS  # TODO: Add accel and emergency actions

