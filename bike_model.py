from math import pi

import numpy as np

MAX_BRAKE_G = 1
G_ACCEL = 9.80665


class BikeModel:
    def __init__(self, x, y, width, add_rotational_friction):
        # Here x is right, y is straight
        self.vehicle_model = self.get_vehicle_model(width)
        self.width = width
        self.yaw_rate = 0
        self.velocity = 0
        self.add_rotational_friction = add_rotational_friction

    @staticmethod
    def get_vehicle_model(width):
        # Bias towards the front a bit
        # https://www.fcausfleet.com/content/dam/fca-fleet/na/fleet/en_us/chrysler/2017/pacifica/vlp/docs/Pacifica_Specifications.pdf
        bias_towards_front = .05 * width
        # Center of gravity
        cog = (width / 2) + bias_towards_front
        # Approximate axles to be 1/8 (1/4 - 1/8) from ends of car
        rear_axle = width * 1 / 8
        front_axle = width - rear_axle
        L_b = cog - rear_axle
        L_a = front_axle - cog
        return L_a, L_b

    # TODO: Numba @njit this
    def step(self, steer, accel, dt):
        steer = min(pi, steer)
        steer = max(-pi, steer)

        # Set x and y to 0 so bike model gives the change in x,y.
        x = 0
        y = 0
        state = [x, y, self.yaw_rate, self.velocity]

        change_x, change_y, self.yaw_rate, self.velocity = \
            f_KinBkMdl(state, steer, accel, self.vehicle_model, dt)

        if self.add_rotational_friction:
            self.yaw_rate = self.yaw_rate * 0.95

        return change_x, change_y, self.yaw_rate, self.velocity


# TODO: Numba @njit this
def f_KinBkMdl(state, steer_angle, accel, vehicle_model, dt):
    """
    process model
    input: state at time k, z[k] := [x[k], y[k], psi[k], v[k]]
    output: state at next time step z[k+1]
    # Adapted from https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    """

    # get states / inputs
    x = state[0]         # straight
    y = state[1]         # right
    yaw_rate = state[2]  # yaw rate
    speed = state[3]     # speed

    # extract parameters
    # Distance from center of gravity to front and rear axles
    (L_a, L_b) = vehicle_model

    # compute slip angle
    slip_angle = np.arctan(L_a / (L_a + L_b) * np.tan(steer_angle))

    # compute next state
    x_next = x + dt * (speed * np.cos(yaw_rate + slip_angle))
    y_next = y + dt * (speed * np.sin(yaw_rate + slip_angle))
    yaw_rate_next = yaw_rate + dt * speed / L_b * np.sin(slip_angle)
    speed_next = speed + dt * accel

    return np.array([x_next, y_next, yaw_rate_next, speed_next])
