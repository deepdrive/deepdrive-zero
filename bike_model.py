from math import pi, cos, sin

import numpy as np

MAX_BRAKE_G = 1
G_ACCEL = 9.80665


class VehicleDynamics:
    def __init__(self, x, y, width, height, angle, add_rotational_friction=True,
                 add_longitudinal_friction=True):
        # Here x is right, y is straight
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
    def step(self, steer, accel, brake, dt):
        steer = min(pi, steer)
        steer = max(-pi, steer)

        state = [self.x, self.y, self.angle_change, self.speed]

        change_x, change_y, self.angle_change, self.speed = \
            f_KinBkMdl(state, steer, accel, self.vehicle_model, dt)

        tuned_fps = 1 / 60  # The FPS we tuned friction ratios at
        friction_exponent = (dt / tuned_fps)
        if self.add_rotational_friction:
            self.angle_change = 0.95 ** friction_exponent * self.angle_change
        if self.add_longitudinal_friction:
            self.speed = 0.999 ** friction_exponent * self.speed
        if brake:
            self.speed = 0.97 ** friction_exponent * self.speed
            # self.speed = 0.97 * self.speed

        theta1 = self.angle
        theta2 = theta1 + pi / 2
        world_change_x = change_x * cos(theta2) + change_y * cos(theta1)
        world_change_y = change_x * sin(theta2) + change_y * sin(theta1)

        self.x += world_change_x
        self.y += world_change_y

        # self.player_sprite.center_x += world_change_x
        # self.player_sprite.center_y += world_change_y
        # self.player_sprite.angle += degrees(self.yaw_rate)
        self.angle += self.angle_change

        return self.x, self.y, self.angle


# TODO: Numba @njit this
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
