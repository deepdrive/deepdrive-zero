import math
import timeit
from random import randint

import numpy as np
from numba import njit


# TODO: Implement broad phase sweep and prune when you have more than 2 objects


@njit(cache=True, nogil=True)
def lines_intersect(a1, a2, b1, b2):
    p = get_intersect(a1, a2, b1, b2)
    if p is not None:
        if is_between(a1, p, a2) and is_between(b1, p, b2):
            return True
    return False


@njit(cache=True, nogil=True)
def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


@njit(cache=True, nogil=True)
def is_between(a, c, b):
    diff = abs(distance(a, c) + distance(c, b) - distance(a, b))
    return diff < 1e-4


@njit(cache=True, nogil=True)
def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack((a1, a2, b1, b2))            # s for stacked
    h = np.hstack((s, np.ones((4, 1))))        # h for homogeneous
    l1 = np.cross(h[0], h[1])                  # get first line
    l2 = np.cross(h[2], h[3])                  # get second line
    x, y, z = np.cross(l1, l2)                 # point of intersection
    if z == 0:                                 # lines are parallel
        return None
    return np.array([x / z, y / z])


def test():
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


def test2():
    test()
    test()



if __name__ == "__main__":
    # simple(np.array([1, 1], dtype=np.float))
    # assert not lines_intersect(np.array([0, 1], dtype=np.float),
    #                            np.array([0, 2], dtype=np.float),
    #                            np.array([1, 10], dtype=np.float),
    #                            np.array([2, 10], dtype=np.float))
    # lines_intersect(np.array([0, 1], dtype=np.float), np.array([0, 2], dtype=np.float), np.array([1, 10], dtype=np.float), np.array([2, 10], dtype=np.float))
    # get_intersect(np.array([0, 1], dtype=np.float), np.array([0, 2], dtype=np.float), np.array([1, 10], dtype=np.float), np.array([2, 10], dtype=np.float))
    print(timeit.timeit(test2, number=1000))
    # assert lines_intersect((0, 1), (0, 2), (-1, 1), (1, 2))
    # assert lines_intersect((0, 1), (0, 2), (0, 2), (2, 0))
    # assert not lines_intersect((0, 0), (0, 1), (0, 2), (2, 0))
    # print('Tests passed!')


    # print(get_intersect((0, 1), (0, 2), (1, 10), (1, 9)) ) # parallel  lines
    # print(get_intersect((0, 1), (0, 2), (1, 10), (2, 10))) # vertical and horizontal lines
    # print(get_intersect((0, 1), (1, 2), (0, 10), (1, 9)) ) # another line for fun
    pass
