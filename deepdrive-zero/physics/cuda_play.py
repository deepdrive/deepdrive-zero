import math
import time

from numba import cuda
import numpy as np


@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1


@cuda.jit
def kinematic_bike_model(vehicle_states):
    """
    output: state of center of gravity at next time step
    # Modified https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    c.f. https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE
    """
    index = cuda.grid(1)

    # Unpack params for our thread
    (steer_angle,               # radians
     throttle,                  # meters/second^2
     cg_to_front_axle,          # center of gravity to front axle in meters
     cg_to_rear_axle,           # center of gravity to rear axle in meters
     dt,                        # seconds to advance
     velocity,                  # signed 1D meters/second towards front of car, so negative is reverse
     angular_velocity,          # radians/second
     _change_x_placeholder,     # meters
     _change_y_placeholder      # meters
     ) = vehicle_states[index]

    # compute slip angle
    slip = cg_to_front_axle / (cg_to_front_axle + cg_to_rear_axle)
    slip_angle = math.atan(slip * math.tan(steer_angle))

    # compute next state
    change_x = dt * (velocity * math.cos(angular_velocity + slip_angle))
    change_y = dt * (velocity * math.sin(angular_velocity + slip_angle))
    angular_velocity += dt * velocity / cg_to_rear_axle * math.sin(slip_angle)
    velocity += dt * throttle

    vehicle_states[index] = (steer_angle,
                             throttle,
                             cg_to_front_axle,
                             cg_to_rear_axle,
                             dt,
                             velocity,
                             angular_velocity,
                             change_x,
                             change_y)


def main():
    velocity = 0.
    angular_velocity = 0.
    steer_angle = 0.  # math.pi / 6
    throttle = 1.
    cg_to_front_axle = 3.
    cg_to_rear_axle = 3.
    dt = 1.
    _change_x_placeholder = 0.
    _change_y_placeholder = 0.

    vehicle_states = np.array([[steer_angle,
                               throttle,
                               cg_to_front_axle,
                               cg_to_rear_axle,
                               dt,
                               velocity,
                               angular_velocity,
                               _change_x_placeholder,
                               _change_y_placeholder],], dtype=np.float32)

    threads_per_block = 1  # multiples of 32 up to machine limit of 1024 on most modern machines
    blocks_per_grid = 1
    kinematic_bike_model[blocks_per_grid, threads_per_block](vehicle_states)

    (steer_angle,
     throttle,
     cg_to_front_axle,
     cg_to_rear_axle,
     dt,
     velocity,
     angular_velocity,
     change_x,
     change_y) = vehicle_states[0]

    # Initial time step has velocity only at the end of the step
    assert not change_x
    assert not change_y
    assert velocity == 1

    kinematic_bike_model[blocks_per_grid, threads_per_block](vehicle_states)

    (steer_angle,
     throttle,
     cg_to_front_axle,
     cg_to_rear_axle,
     dt,
     velocity,
     angular_velocity,
     change_x,
     change_y) = vehicle_states[0]

    assert change_x == 1
    assert not change_y
    assert velocity == 2


def play():
    an_array = np.arange(10000000).reshape((100,100000))
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    start1 = time.time()
    x1 = an_array + 1
    print(f'CPU {round(time.time() - start1, 9)}')
    increment_a_2D_array[blockspergrid, threadsperblock](an_array)
    start2 = time.time()
    print(f'GPU {round(time.time() - start2, 9)}')
    # print(an_array)


if __name__ == '__main__':
    main()
