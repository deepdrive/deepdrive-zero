import numpy as np
from math import pi
from loguru import logger as log
from numba import njit

from deepdrive_zero.constants import CACHE_NUMBA


@njit(cache=CACHE_NUMBA, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def flatten_points(points):
    return [coord for point in points for coord in point]


def get_angles_ahead(ego_angle, closest_map_index, map_points,
                     seconds_ahead, speed, route_length, total_points,
                     heading, ego_front):
    # TODO: Profile / numba
    num_indices = len(seconds_ahead)
    points_per_meter = total_points / route_length
    points_per_second = speed * points_per_meter
    seconds_per_index = 0.5
    points_per_index = points_per_second * seconds_per_index
    points_per_index = max(1, int(round(points_per_index)))
    min_first_point_dist = 2
    first_index = closest_map_index + points_per_meter * min_first_point_dist
    first_index = int(round(first_index))
    first_index = min(first_index, len(map_points) - 1)
    last_index = closest_map_index + points_per_index * num_indices
    last_index = min(last_index, len(map_points))
    points = map_points[int(first_index):last_index:points_per_index]
    points = list(points)

    while len(points) < num_indices:
        # Append last point where we're at the end
        points.append(points[-1])

    angles = [get_angle(heading, p - ego_front) for p in points]
    return angles


@njit(cache=CACHE_NUMBA, nogil=True)
def get_angle(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


def quadratic_regression(x, y):
    return np.polyfit(x, y, 2, full=True)


def np_rand():
    return np.random.rand()


def test_angle():
    def npf(x):
        return np.array(x, dtype=float)
    assert np.isclose(get_angle(npf((1, 1)), npf((1, 0))), pi / 4)
    assert np.isclose(get_angle(npf((1, 0)), npf((1, 1))), -pi / 4)
    assert np.isclose(get_angle(npf((0, 1)), npf((1, 0))), pi / 2)
    assert np.isclose(get_angle(npf((1, 0)), npf((0, 1))), -pi / 2)
    assert np.isclose(get_angle(npf((1, 0)), npf((1, 0))), 0)
    assert np.isclose(get_angle(npf((1, 0)), npf((-1, 0))), pi)


def test_quadratic_regression():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 0.8, 3.2, 9.2, -0.8, -1.0])
    z = quadratic_regression(x, y)


def play_regression():

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # y = np.array([0.0, 0.8, 3.2, 9.2, -0.8, -1.0])
    y = x ** 0.5
    z = np.polyfit(x, y, 2, full=True)

    print(sum(z[1]))

    p = np.poly1d(z[0])
    p3 = np.poly1d(np.polyfit(x, y, 2))

    import matplotlib.pyplot as plt
    xp = np.linspace(-2, 6, 100)
    _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p3(xp), '--')
    plt.ylim(-20, 20)
    plt.show()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    test_angle()