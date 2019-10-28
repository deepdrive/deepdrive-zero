import numpy as np
from loguru import logger as log


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # noinspection PyTypeChecker
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def flatten_points(points):
    return [coord for point in points for coord in point]


def get_angles_ahead(angle, closest_map_index, map_points,
                     seconds_ahead, speed, total_length, total_points):
    # TODO: Profile / numba
    num_indices = len(seconds_ahead)
    points_per_meter = total_points / total_length
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

    # TODO: Fix this! Doesn't make sense to get angles along trajectory if not
    #   on trajectory, and doesn't make sense to correct with current angle
    #   by subtracting it when not on trajectory.
    cp = map_points[closest_map_index]
    angles = [get_heading(cp, p) for p in points]
    angles = angle - np.array(angles)
    return angles


def get_heading(p1, p2):
    return -angle_between(np.array([0, 1]), np.array([p2[0] - p1[0],
                                                      p2[1] - p1[1]]))
