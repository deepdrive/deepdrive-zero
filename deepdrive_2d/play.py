import math

import numpy as np


def lines_intersect(a1, a2, b1, b2):
    a1 = np.array(a1, dtype=np.float)
    a2 = np.array(a2, dtype=np.float)
    b1 = np.array(b1, dtype=np.float)
    b2 = np.array(b2, dtype=np.float)
    p = get_intersect(a1, a2, b1, b2)
    if p is not None:
        if is_between(a1, p, a2) and is_between(b1, p, b2):
            return True
    return False


def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def is_between(a, c, b):
    return math.isclose(distance(a, c) + distance(c, b), distance(a, b))


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])      # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])            # get first line
    l2 = np.cross(h[2], h[3])            # get second line
    x, y, z = np.cross(l1, l2)           # point of intersection
    if z == 0:                           # lines are parallel
        return None
    return x / z, y / z


def overlap_sucks(a1, a2, b1, b2):
    a_x = [a1[0], a2[0]]
    a_y = [a1[1], a2[1]]
    b_x = [b1[0], b2[0]]
    b_y = [b1[1], b2[1]]

    ret = overlap_one_dim(a_x, b_x) and overlap_one_dim(a_y, b_y)

    return ret


def overlap_one_dim(a_dim, b_dim):
    left, right = (a_dim, b_dim) if min(a_dim) <= min(b_dim) else (b_dim, a_dim)
    ret = max(left) >= min(right)
    return ret


if __name__ == "__main__":
    assert not lines_intersect((0, 1), (0, 2), (1, 10), (2, 10))
    assert lines_intersect((0, 1), (0, 2), (-1, 1), (1, 2))
    assert lines_intersect((0, 1), (0, 2), (0, 2), (2, 0))
    assert not lines_intersect((0, 0), (0, 1), (0, 2), (2, 0))

    # print(get_intersect((0, 1), (0, 2), (1, 10), (1, 9)) ) # parallel  lines
    # print(get_intersect((0, 1), (0, 2), (1, 10), (2, 10))) # vertical and horizontal lines
    # print(get_intersect((0, 1), (1, 2), (0, 10), (1, 9)) ) # another line for fun
