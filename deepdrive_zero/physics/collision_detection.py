import math
import sys
import timeit
from random import randint
from typing import Type, Union, List, Tuple

import numpy as np
from numba import njit

from deepdrive_zero.physics.bike_model import get_vehicle_dimensions
from deepdrive_zero.constants import CACHE_NUMBA

pi = np.pi


# TODO: Implement broad phase sweep and prune when you have more than 2 objects


@njit(cache=CACHE_NUMBA, nogil=True)
def lines_intersect(a1, a2, b1, b2):
    p = get_intersect(a1, a2, b1, b2)
    if p is not None:
        if is_between(a1, p, a2) and is_between(b1, p, b2):
            return True
    return False


@njit(cache=CACHE_NUMBA, nogil=True)
def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


@njit(cache=CACHE_NUMBA, nogil=True)
def is_between(start, mid, end):
    diff = abs(distance(start, mid) + distance(mid, end) - distance(start, end))
    return diff < 1e-4


@njit(cache=CACHE_NUMBA, nogil=True)
def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: np.array [x, y] a point on the first line
    a2: np.array [x, y] another point on the first line
    b1: np.array [x, y] a point on the second line
    b2: np.array [x, y] another point on the second line

    Explanation: http://robotics.stanford.edu/~birch/projective/node4.html
    """
    s = np.vstack((a1, a2, b1, b2))            # s for stacked
    h = np.hstack((s, np.ones((4, 1))))        # h for homogeneous
    l1 = np.cross(h[0], h[1])                  # get first line
    l2 = np.cross(h[2], h[3])                  # get second line
    x, y, z = np.cross(l1, l2)                 # point of intersection
    if z == 0:                                 # lines are parallel
        return None
    return np.array([x / z, y / z])


@njit(cache=CACHE_NUMBA, nogil=True)
def get_lines_from_rect_points(rect_points: tuple):
    p = rect_points
    ret = ((tuple(p[0]), tuple(p[1])),
           (tuple(p[1]), tuple(p[2])),
           (tuple(p[2]), tuple(p[3])),
           (tuple(p[3]), tuple(p[0])))
    return ret


def check_collision_agents(agents: list):
    """
    :param agents: List of agents with ego_lines property containing 4 points
        representing corners of hit box
    """
    pair_indexes = get_pair_indexes(len(agents))
    collisions = []
    for i, j in pair_indexes:
        if check_collision(agents[i].ego_lines, agents[j].ego_lines):
            agents[i].collided_with.append(agents[j])
            agents[j].collided_with.append(agents[i])
            collisions.append((i,j))
    return collisions


@njit(cache=CACHE_NUMBA, nogil=True)
def check_collision_ego_obj(ego_rect: tuple, obj2: tuple):
    """

    :param ego_rect: Points representing 4 corners of ego
    :param obj2: n x 2 x 2 tuple of start & end points for n lines
    :return: (bool) True if collision
    """
    return check_collision(get_lines_from_rect_points(ego_rect), obj2)


@njit(cache=CACHE_NUMBA, nogil=True)
def check_collision(obj1: tuple, ob2: tuple):
    """

    :param obj1: n x 2 x 2 tuple of start & end points for n lines
    :param ob2: n x 2 x 2 tuple of start & end points for n lines
    :return:
    """

    for line1_start, line1_end in obj1:
        for line2_start, line2_end in ob2:
            if lines_intersect(np.array(line1_start), np.array(line1_end),
                               np.array(line2_start), np.array(line2_end)):
                return True
    return False


def get_rect(center_x, center_y, angle, width, length):
    """
    :param center_x: x-coordinate of the center of vehicle in meters
    :param center_y: y-coordinate of the center of vehicle in meters
    :param angle: angle in radians
    :param width: width of vehicle in meters
    :param length: length of vehicle in meters
    :return: 4 points of the rectangle:
            Starts at front left and goes clockwise:
            front left, front right, back right, back left
    """
    ego_rect = _get_rect(center_x, center_y, angle, width, length)

    # Numba likes tuples
    ego_rect_tuple = tuple(map(tuple, ego_rect.tolist()))

    return ego_rect, ego_rect_tuple


@njit(cache=CACHE_NUMBA, nogil=True)
def _get_rect(center_x, center_y, angle, width, length):
    """
    :param center_x: x-coordinate of the center of vehicle in meters
    :param center_y: y-coordinate of the center of vehicle in meters
    :param angle: angle in radians
    :param width: width of vehicle in meters
    :param length: length of vehicle in meters
    :return: 4 points of the rectangle:
            Starts at top left and goes clockwise
            top left, top right, bottom right, bottom left
    """
    front_axle, rear_axle, overhang_front, overhang_rear = get_vehicle_dimensions(length)

    # Find vector of rear axle center wrt inertial frame
    # Vector of rear axle center wrt center of car
    c_p = np.array([0, -rear_axle, 1.0])
    # Pose of center of car wrt inertial frame
    t1 = _pose(center_x, center_y, angle)
    # Vector of rear axle center wrt inertial frame
    i_p = _transform(t1, c_p.T)

    # Find rectangle vertices wrt inertial frame
    # Vector of rectangle vertices wrt rear axle
    p = np.array([[-width / 2.0, length - overhang_rear, 1.0],
                  [width / 2.0, length - overhang_rear, 1.0],
                  [width / 2.0, -overhang_rear, 1.0],
                  [-width / 2.0, -overhang_rear, 1.0]])
    # Pose of rear axle wrt inertial frame
    t2 = _pose(i_p[0][0], i_p[1][0], angle)
    # Vector of rectangle vertices wrt inertial frame
    ret = _transform(t2, p.T).T

    return ret


@njit(cache=CACHE_NUMBA, nogil=True)
def _transform(pose, point):
    """
    :param pose: 3x3 pose matrix which defines the pose of a coordinate frame wrt to some second coordinate frame
    :param point: 2x1 vector defined wrt the first coordinate frame
    :return: 2x1 vector wrt the second coordinate frame
    """

    return (pose @ point).reshape((3, -1))[0:2]


@njit(cache=CACHE_NUMBA, nogil=True)
def _pose(x, y, angle):
    """
    :param x: x-coordinate of the origin of a coordinate frame in meters
    :param y: y-coordinate of the origin of a coordinate frame in meters
    :param angle: orientation of the coordinate frame in radians
    :return: 3x3 homogeneous transformation matrix
    """

    return np.array([[np.cos(angle), -np.sin(angle), x],
                     [np.sin(angle),  np.cos(angle), y],
                     [0, 0, 1.0]])


@njit(cache=CACHE_NUMBA, nogil=True)
def get_pair_indexes(length: int) -> List:
    indexes = list(range(length))
    ret = []
    for i1 in indexes:
        for i2 in indexes[i1+1:]:
            ret.append((i1, i2))
    return ret


def test_check_collision():
    ego_rect, ego_rect_tuple = get_rect(1, 1, pi / 4, 2, 1)
    ert = tuple(map(tuple, ego_rect.tolist()))  # ego rect tuple
    max_x = max(ego_rect.T[0])
    min_x = min(ego_rect.T[0])
    max_y = max(ego_rect.T[1])
    min_y = min(ego_rect.T[1])
    y_at_max_x = ego_rect[np.where(ego_rect.T[0] == max_x)][0][1]
    mid_x = (max_x - min_x) / 2 + min_x
    mid_y = (max_y - min_y) / 2 + min_y

    # Top boundary should collide
    assert check_collision_ego_obj(ert, (((1e-10, max_y), (1e10, max_y)),))

    # Bottom boundary should collide
    assert check_collision_ego_obj(ert, (((1e-10, min_y), (1e10, min_y)),))

    # Right boundary should collide
    assert check_collision_ego_obj(ert, (((max_x, -1e10), (max_x, 1e10)),))

    # Left boundary should collide
    assert check_collision_ego_obj(ert, (((min_x, -1e10), (min_x, 1e10)),))

    # Small line should collide
    assert check_collision_ego_obj(ert, (((max_x, y_at_max_x),
                                          (max_x, y_at_max_x + 1e-4)),))

    # Zero length line should not collide
    assert not check_collision_ego_obj(ert, (((max_x, y_at_max_x),
                                              (max_x, y_at_max_x)),))

    # Mid x should collide
    assert check_collision_ego_obj(ert, (((mid_x, max_y), (mid_x, min_y)),))

    # Mid x outside y should not collide
    assert not check_collision_ego_obj(ert, (((mid_x, 1e10), (mid_x, 1e10 + 1)),))

    # Mid y should collide
    assert check_collision_ego_obj(ert, (((min_x, mid_y), (max_x, mid_y)),))

    # Mid x outside y should not collide
    assert not check_collision_ego_obj(ert, (((1e20, mid_y), (mid_x, 1e20 - 1)),))


def test_transform():
    point = np.array([1.0, 2.0, 1.0])
    pose = _pose(2.0, 3.0, 0.0)
    transformed_point = _transform(pose, point)

    assert (all(np.isclose(transformed_point.T,
                           np.array([3.0, 5.0])).flatten()))

    point = np.array([1.0, 2.0, 1.0])
    pose = _pose(2.0, 3.0, pi/2)
    transformed_point = _transform(pose, point)

    assert (all(np.isclose(transformed_point.T,
                           np.array([0.0, 4.0])).flatten()))


def test_get_rect():
    r, _ = get_rect(0, 0, pi / 2, 2, 1)
    assert all(np.isclose(r[0], [-0.5, -1]))

    r, _ = get_rect(1, 1, pi / 2, 2, 1)
    assert all(np.isclose(r, [[0.5, 0],
                              [0.5, 2],
                              [1.5, 2],
                              [1.5, 0]]).flatten())


def test_lines_intersect():
    assert not lines_intersect(np.array([0, 1], dtype=np.float),
                               np.array([0, 2], dtype=np.float),
                               np.array([1, 10], dtype=np.float),
                               np.array([2, 10], dtype=np.float))
    assert lines_intersect(np.array([0, 1], dtype=np.float),
                           np.array([0, 2], dtype=np.float),
                           np.array([-1, 1], dtype=np.float),
                           np.array([1, 2], dtype=np.float))
    assert lines_intersect(np.array([0, 1], dtype=np.float),
                           np.array([0, 2], dtype=np.float),
                           np.array([0, 2], dtype=np.float),
                           np.array([2, 0], dtype=np.float))
    assert not lines_intersect(np.array([0, 0], dtype=np.float),
                               np.array([0, 1], dtype=np.float),
                               np.array([0, 2], dtype=np.float),
                               np.array([2, 0], dtype=np.float))

    lines_intersect(np.array([randint(0, 10), randint(0, 10)], dtype=np.float),
                    np.array([randint(0, 10), randint(0, 10)], dtype=np.float),
                    np.array([randint(0, 10), randint(0, 10)], dtype=np.float),
                    np.array([randint(0, 10), randint(0, 10)], dtype=np.float))


def test_lines_intersect_x2():
    test_lines_intersect()
    test_lines_intersect()


def test_get_pairs_indexes():
    assert get_pair_indexes(3) == [(0, 1), (0, 2), (1, 2)]


def main():
    if '--test_check_collision' in sys.argv:
        # get_lines_from_rect_points(
        #     ((1, 0), (1, 0), (1, 0), (1, 0))
        # )
        test_check_collision()
    elif '--test_get_pair_indexes' in sys.argv:
        test_get_pairs_indexes()
    elif '--test_transform' in sys.argv:
        test_transform()
    elif '--test_get_rect' in sys.argv:
        test_get_rect()
    else:
        print(timeit.timeit(test_lines_intersect_x2, number=1000))




if __name__ == "__main__":
    main()
