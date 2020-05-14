import time

import numpy as np
from numba import njit, prange, numba
from numba.typed import List

from deepdrive_zero.constants import CACHE_NUMBA


@njit(cache=CACHE_NUMBA, nogil=True)
def get_lane_distance(p0, p1, ego_rect_pts: np.array, is_left_lane_line=True):
    """
    Get the ego's distance from the lane segment p1-p0
    by rotating p1 about p0 such that the segment is vertical.
    Then rotate the ego points by the same angle and find the minimum x
    coordinate of the ego, finally comparing it to the x coordinate of the
    rotated line.

    Note that direction of travel should be from p0 to p1.

    :param p0: First lane line point
    :param p1: Second lane line point
    :param ego_rect_pts: Four points of ego rectangle
    :param is_left_lane_line: Whether lane line should be to the left of ego, else right
    :return: Max distance from lane if crossed, min distance if not. Basically
        how bad is your position.
    """
    # TODO:
    #  Handle this case (bad map):
    #  https://user-images.githubusercontent.com/181225/81328684-d74a1880-908c-11ea-920e-caf4c94d3d5c.jpg
    #  Perhaps check that the closest point to the infinite line
    #  made up by p0 and p1 is NOT closer than the closest point to the
    #  bounded line segment, as in this case you could have a problem with the
    #  map, where for example on a dog leg left turn, the ego point is before
    #  the turn and therefore can be legally be left of the infinite line
    #  but right of the lane segment before the turn (i.e. the lane segment
    #  that _should_ have been passed.

    lane_adjacent = p1[0] - p0[0]
    lane_opposite = p1[1] - p0[1]

    if lane_adjacent == 0:
        # Already parallel
        angle_to_vert = 0
    else:
        # Not parallel
        lane_angle = np.arctan2(lane_opposite, lane_adjacent)
        angle_to_vert = np.pi/2 - lane_angle

    # Rotate about p0
    rot_ego_pts = rotate_points(ego_rect_pts, angle_to_vert, p0)

    if is_left_lane_line:
        return np.min(rot_ego_pts[:, 0]) - p0[0]
    else:
        return p0[0] - np.max(rot_ego_pts[:, 0])


@njit(cache=CACHE_NUMBA, nogil=True)
def rotate_points(points, angle, about_pt):
    a, b = about_pt
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Planar rotation about point
    # http://web.cs.iastate.edu/~cs577/handouts/homogeneous-transform.pdf
    rotate_about = np.array([[cos_a, -sin_a, -a * cos_a + b * sin_a + a],
                             [sin_a, cos_a, -a * sin_a - b * cos_a + b]],
                            dtype=np.float32)

    rot_ego_pts = []
    for i in prange(len(points)):
        pt = points[i] #Invalid use of Function(<built-in function getitem>) with argument(s) of type(s): (float64, Literal[int](0))
        homogeneous_coord = np.array((pt[0], pt[1], 1), dtype=np.float32)
        rot_ego_pts.append(list((rotate_about @ homogeneous_coord).T))

    return np.array(rot_ego_pts, dtype=np.float32)


def test_rotate_points():
    rot = rotate_points(points=np.array([(4, 4), (6, 2)]),
                        angle=-np.pi/4,
                        about_pt=np.array((5, 0)))
    assert np.isclose(rot[0][0], rot[1][0])

    start = time.time()
    rot = rotate_points(points=np.array([np.random.random(2) for _ in range(10000)]),
                        angle=-np.pi/4,
                        about_pt=np.array((5, 0)))
    print(f'took {time.time() - start} seconds')

    # prange
    # took 0.01629781723022461 seconds
    # with tbb: 0.01625680923461914

    # range
    # took 0.0012233257293701172 seconds
    # took 0.012378215789794922 seconds
    # took 0.01807570457458496 seconds

def test_lane_distance():
    lane_dist = get_lane_distance(p0=(5, 0), p1=(0, 5),
                                  ego_rect_pts=np.array([(4, 4), (6, 2)]),
                                  is_left_lane_line=True)
    p0_to_6_2 = np.array([6,2]) - np.array([5,0])
    p0_to_6_2_mag = np.linalg.norm(p0_to_6_2)
    assert np.isclose(
        lane_dist,
        p0_to_6_2_mag * np.cos(np.arctan(p0_to_6_2[1]/p0_to_6_2[0]) - np.pi/4))


if __name__ == '__main__':
    test_lane_distance()
    test_rotate_points()

@njit(parallel=True)
def prange_test(A):
    s = 0
    # Without "parallel=True" in the jit-decorator
    # the prange statement is equivalent to range
    for i in prange(A.shape[0]):
        s += A[i]
    return s


def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False,
                                clampA1=False, clampB0=False, clampB1=False):
    """
    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance

    Clamp is basically getting the distance between finite line segments.
    Otherwise lines are considered to be infinitely long.

    See stack overflow for more about clamp behavior.

    https://stackoverflow.com/a/18994296/134077
    """

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)
