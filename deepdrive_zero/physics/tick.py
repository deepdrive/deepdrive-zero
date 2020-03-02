import numpy as np
from numba import njit

from deepdrive_zero.constants import CACHE_NUMBA
from deepdrive_zero.physics.bike_model import bike_with_friction_step
from deepdrive_zero.logs import log

@njit(cache=CACHE_NUMBA, nogil=True)
def physics_tick(accel, add_longitudinal_friction, add_rotational_friction,
                 brake, curr_acceleration, curr_angle, curr_angle_change,
                 curr_angular_velocity, curr_gforce, curr_max_gforce,
                 curr_speed, curr_velocity, curr_x, curr_y, dt, n, prev_accel,
                 prev_brake, prev_steer, steer, vehicle_model, ignore_brake,
                 constrain_controls, max_steer_change, max_accel_change):
    if ignore_brake:
        brake = 0
    if curr_speed > 100:
        accel = 0
    steer_change = steer - prev_steer
    accel_change = accel - prev_accel
    if constrain_controls:
        max_steer_change = 0.02  # TODO: Make this physically based
        max_accel_change = 0.02  # TODO: Make this physically based
        steer_change = min(max_steer_change, steer_change)
        steer_change = max(-max_steer_change, steer_change)
        accel_change = min(max_accel_change, accel_change)
        accel_change = max(-max_accel_change, accel_change)
        i_steer, i_accel, i_brake = 0, 0, 0
    for i in range(n):
        """
        Enforce real-world constraint that you can't teleport the gas pedal
        or steering wheel between positions, rather you must visit the
        intermediate positions between subsequent settings.
        """
        interp = (i + 1) / n
        i_steer = prev_steer + interp * steer_change
        i_accel = prev_accel + interp * accel_change
        if brake:
            i_brake = prev_brake + interp * (brake - prev_brake)
        else:
            i_brake = 0
        # log.info(f'steer {steer} accel {accel} brake {brake} vel {self.velocity}')
        prev_x, prev_y, prev_angle = curr_x, curr_y, curr_angle
        curr_x, curr_y, curr_angle, curr_angle_change, curr_speed = \
            bike_with_friction_step(
                steer=i_steer, accel=i_accel, brake=i_brake, dt=dt,
                x=curr_x, y=curr_y, angle=curr_angle,
                angle_change=curr_angle_change,
                speed=curr_speed,
                add_rotational_friction=add_rotational_friction,
                add_longitudinal_friction=add_longitudinal_friction,
                vehicle_model=vehicle_model, )

        gforce_outputs = get_gforce_levels(
            curr_x, curr_y, curr_angle, prev_x, prev_y, prev_angle, dt,
            curr_velocity, accel, curr_max_gforce)
        (curr_gforce, curr_max_gforce, new_jerk, new_acceleration,
         curr_angular_velocity, new_velocity) = gforce_outputs

    return (new_acceleration, curr_angle, curr_angle_change,
            curr_angular_velocity, curr_gforce, new_jerk,
            curr_max_gforce, curr_speed, curr_x, curr_y, i_accel, i_brake,
            i_steer, new_velocity)


@njit(cache=CACHE_NUMBA, nogil=True)
def get_gforce_levels(x, y, curr_angle, prev_x, prev_y, prev_angle, dt,
                      curr_velocity, curr_accel, curr_max_gforce):
    pos_change = np.array([prev_x - x, prev_y - y])
    prev_velocity = curr_velocity
    curr_velocity = pos_change / dt
    new_accel = (curr_velocity - prev_velocity) / dt
    angular_velocity = (curr_angle - prev_angle) / dt
    accel_magnitude = np.linalg.norm(new_accel)
    gforce = accel_magnitude / 9.807
    max_gforce = max(gforce, curr_max_gforce)
    # prev_gforce.append(self.gforce)
    jerk = new_accel - curr_accel
    return gforce, max_gforce, jerk, new_accel, angular_velocity, curr_velocity