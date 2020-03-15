from math import pi, cos, sin

import numpy as np
from numba import njit

from deepdrive_zero.constants import USE_VOYAGE, VEHICLE_WIDTH, CACHE_NUMBA


@njit(cache=CACHE_NUMBA, nogil=True)
def bike_with_friction_step(
        steer, accel, brake, dt,
        x, y, angle, angle_change, speed, add_rotational_friction,
        add_longitudinal_friction, vehicle_model):
    """

    :param steer: (float) Steering angle in radians
    :param accel: (float) m/s**2
    :param brake: (float) whether to brake or not
    :param dt: (float) time step
    :param x: (float) meters x
    :param y: (float) meters y
    :param angle: (float) radian angle offset
    :param angle_change: (float) angular velocity
    :param speed: (float) m/s
    :param add_rotational_friction: (bool)
        Whether to slow angular velocity every time step
    :param add_longitudinal_friction: (bool)
        Whether to slow velocity every time step
    :param vehicle_model:
        Distance from center of gravity to front and rear axles
    :return:
    """
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

    # TODO: We should just output the desired accel (+-) and allow higher magnitude
    #   negative accel than positive due to brake force. Otherwise network will
    #   be riding brakes more than reasonable (esp without disincentivizing this
    #   in the reward)
    speed = 0.96 ** brake * speed

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


@njit(cache=CACHE_NUMBA, nogil=True)
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


def get_vehicle_model(width):
    """
    :param width: Width of vehicle
    :return: Distance from center of gravity to front and rear axles
    """

    L_a, L_b, _, _ = get_vehicle_dimensions(width)

    return L_a, L_b


@njit(cache=CACHE_NUMBA, nogil=True)
def get_vehicle_dimensions(width):
    """
    :param width: Width of vehicle
    :return: Distance from center of gravity to front and rear axles
    """

    # Bias towards the front a bit
    # https://www.fcausfleet.com/content/dam/fca-fleet/na/fleet/en_us/chrysler/2017/pacifica/vlp/docs/Pacifica_Specifications.pdf

    if USE_VOYAGE:
        bias_towards_front = .05 * width
    else:
        bias_towards_front = 0

    # Center of gravity
    center_of_gravity = (width / 2) + bias_towards_front

    # Approximate axles to be 1/8 (1/4 - 1/8) from ends of car
    rear_axle = width * 1 / 8
    front_axle = width - rear_axle
    L_b = center_of_gravity - rear_axle
    L_a = front_axle - center_of_gravity

    return L_a, L_b, rear_axle, front_axle


def test_bike_with_friction_step():
    vehicle_model = get_vehicle_model(VEHICLE_WIDTH)

    # Do nothing
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=pi/4, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=1, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert np.isclose(x, 0) and x == 6.123233995736766e-17
    assert y == 1
    assert angle == 0
    assert speed == 0.9417362622231682
    assert angle_change == 0
