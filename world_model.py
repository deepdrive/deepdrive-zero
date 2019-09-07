import math
import time
from math import pi, cos, sin

import numpy as np
from loguru import logger as log
from numba import njit

from utils import angle_between

MAX_BRAKE_G = 1
G_ACCEL = 9.80665


class Dynamics:
    def __init__(self, x, y, width, height,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,
                 angle=None,
                 map=None):
        # x is right, y is straight

        if angle and map:
            raise RuntimeError('Cannot specify both angle and map')

        self.vehicle_model = self.get_vehicle_model(width)
        self.width = width
        self.height = height
        self.angle = angle  # Angle in radians
        self.angle_change = 0
        self.speed = 0
        self.x = x
        self.y = y
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.map = map

        interp_dist_pixels = map.width / len(map.x)
        angle_waypoint_meters = 1

        # angle_waypoint_index = round(
        #     (angle_waypoint_meters * ROUGH_PIXELS_PER_METER) /
        #     interp_dist_pixels) + 1
        angle_waypoint_index = 6
        x1 = map.x[0]
        y1 = map.y[0]
        x2 = map.x[angle_waypoint_index]
        y2 = map.y[angle_waypoint_index]
        self.heading_x = x2
        self.heading_y = y2
        angle = -angle_between(np.array([0, 1]), np.array([x2-x1, y2-y1]))
        if (map.height - y1) / map.height <= 0.5 and \
                abs(angle) < (math.pi * 0.001):
            # TODO: Reproduce problem where car is backwards along spline
            #  and make sure this fixes it.
            angle += math.pi  # On top so, face down
        self.angle = angle

    @staticmethod
    def get_vehicle_model(width):
        # Bias towards the front a bit
        # https://www.fcausfleet.com/content/dam/fca-fleet/na/fleet/en_us/chrysler/2017/pacifica/vlp/docs/Pacifica_Specifications.pdf
        bias_towards_front = .05 * width
        # Center of gravity
        center_of_gravity = (width / 2) + bias_towards_front
        # Approximate axles to be 1/8 (1/4 - 1/8) from ends of car
        rear_axle = width * 1 / 8
        front_axle = width - rear_axle
        L_b = center_of_gravity - rear_axle
        L_a = front_axle - center_of_gravity
        return L_a, L_b

    def step(self, steer, accel, brake, dt):
        # TODO: Car lists with collision detection
        start = time.time()
        self.x, self.y, self.angle, self.angle_change, self.speed = \
            bike_with_friction_step(
                steer=steer, accel=accel, brake=brake, dt=dt,
                x=self.x, y=self.y, angle=self.angle,
                angle_change=self.angle_change,
                speed=self.speed,
                add_rotational_friction=self.add_rotational_friction,
                add_longitudinal_friction=self.add_longitudinal_friction,
                vehicle_model=self.vehicle_model,
            )
        log.trace(f'time {time.time() - start}')
        return self.x, self.y, self.angle


@njit(cache=True, nogil=True)
def bike_with_friction_step(
        steer, accel, brake, dt,
        x, y, angle, angle_change, speed, add_rotational_friction,
        add_longitudinal_friction, vehicle_model):
    steer = min(pi, steer)
    steer = max(-pi, steer)
    state = [x, y, angle_change, speed]

    change_x, change_y, angle_change, speed = \
        f_KinBkMdl(state, steer, accel, vehicle_model, dt)

    tuned_fps = 1 / 60  # The FPS we tuned friction ratios at
    friction_exponent = (dt / tuned_fps)
    if add_rotational_friction:
        angle_change = 0.95 ** friction_exponent * angle_change
    if add_longitudinal_friction:
        speed = 0.999 ** friction_exponent * speed
    if brake:
        speed = 0.97 ** friction_exponent * speed
        # self.speed = 0.97 * self.speed

    theta1 = angle
    theta2 = theta1 + pi / 2
    world_change_x = change_x * cos(theta2) + change_y * cos(theta1)
    world_change_y = change_x * sin(theta2) + change_y * sin(theta1)

    x += world_change_x
    y += world_change_y

    # self.player_sprite.center_x += world_change_x
    # self.player_sprite.center_y += world_change_y
    # self.player_sprite.angle += degrees(self.yaw_rate)
    angle += angle_change

    return x, y, angle, angle_change, speed


@njit(cache=True, nogil=True)
def f_KinBkMdl(state, steer_angle, accel, vehicle_model, dt):
    """
    process model
    input: state at time k, z[k] := [x[k], y[k], psi[k], v[k]]
    output: state at next time step z[k+1]
    # From https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    """

    # get states / inputs
    x = state[0]             # straight
    y = state[1]             # right
    angle_change = state[2]  # angle change
    speed = state[3]         # speed

    # extract parameters
    # Distance from center of gravity to front and rear axles
    (L_a, L_b) = vehicle_model

    # compute slip angle
    slip_angle = np.arctan(L_a / (L_a + L_b) * np.tan(steer_angle))

    # compute next state
    change_x = dt * (speed * np.cos(angle_change + slip_angle))
    change_y = dt * (speed * np.sin(angle_change + slip_angle))
    angle_change = angle_change + dt * speed / L_b * np.sin(slip_angle)
    speed_next = speed + dt * accel

    return np.array([change_x, change_y, angle_change, speed_next])
