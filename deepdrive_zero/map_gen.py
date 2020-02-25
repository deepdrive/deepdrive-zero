import os
import time

import numpy as np
from loguru import logger as log
from scipy.interpolate import interp1d

from deepdrive_zero.constants import PX_PER_M, MAP_WIDTH_PX, MAP_HEIGHT_PX, \
    SCREEN_MARGIN, MAP_IMAGE

GAP_M = 1
DIR = os.path.dirname(os.path.realpath(__file__))


# TODO: @njit
def gen_random_map(should_plot=False, num_course_points=3, resolution=10,
                   should_save=True) -> np.array:
    # TODO: Linear interp with 2 points for straight roads
    # TODO: Randomize spacing
    # TODO: Different interpolations
    # TODO: Randomize num course points
    min_x = 0
    max_x = 1
    min_y = 0.1
    max_y = 0.9
    x = np.linspace(min_x, max_x, num=num_course_points + 1,
                    endpoint=True)
    y = np.array(list(np.random.rand(num_course_points + 1)))
    return gen_map(x, y, resolution, should_plot, should_save)


def gen_map(x, y, resolution=10, should_plot=True, should_save=False):
    linear_interp = interp1d(x, y)
    min_x = x.min()
    max_x = x.max()
    num_course_points = len(x) - 1

    # TODO: Remove first interpolation if making equidistant
    cubic_interp = interp1d(x, y, kind='cubic')

    xnew = np.linspace(min_x, max_x,
                       num=num_course_points * resolution, endpoint=True)

    ynew = cubic_interp(xnew)

    # Normalize 0->1
    ynew = (ynew - ynew.min()) / (ynew.max() - ynew.min())

    equidistant = interpolate_equidistant(np.column_stack((xnew, ynew)),
                                          distance=1/len(xnew))

    xequi = equidistant[:, 0]
    yequi = equidistant[:, 1]

    if should_plot or should_save:
        import matplotlib.pyplot as plt

        if should_plot:
            plt.plot(x, y, 'o', xequi, linear_interp(xequi), '-', xequi,
                     yequi,
                     '--')
            plt.legend(['data', 'linear', 'cubic'], loc='best')
            plt.show()
        elif should_save:
            plt.plot(xnew, ynew, '--', color='xkcd:orange', linewidth=1)
            plt.axis('off')
            start_save = time.time()
            # TODO: Specify height and width
            plt.savefig(MAP_IMAGE, bbox_inches='tight',
                        pad_inches=0,
                        facecolor='xkcd:cornflower blue',
                        edgecolor='xkcd:cornflower blue',
                        dpi=200)
            log.debug(f'Save image time {time.time() - start_save}')

    retx = xequi
    rety = yequi

    return retx, rety


def interpolate_equidistant(points: np.array,
                            distance: float = None) -> np.array:
    """

    :param points: Spline points
    :param distance: Desired distance between output spline points
    :return: Equidistant spline points with same shape as input
    """
    distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    distances = np.hstack([[0], distances])
    total_distance = distances[-1]
    if distance is None:
        distance = total_distance / len(points)
    total_points = total_distance // distance + 1
    chords = list(np.arange(0, total_points) * distance)
    if distances[-1] % distance > (distance / 3):
        # Add an extra interp point so there won't be big gap at the end
        chords.append(distances[-1])
    cubic_interpolator = interp1d(distances, points, 'cubic', axis=0)
    ret = cubic_interpolator(chords)
    return ret


def get_intersection():
    """
    Returns lane width and 6 lines that make up the intersection,
    3 vertical, 3 horizontal

    Each line consists of 2 points and has shape 2x2

    """
    ppm = PX_PER_M
    _lane_width_feet = 10  # https://www.citylab.com/design/2014/10/why-12-foot-traffic-lanes-are-disastrous-for-safety-and-must-be-replaced-now/381117/
    lane_width = 0.3048 * _lane_width_feet
    map_width = MAP_WIDTH_PX / ppm
    map_height = MAP_HEIGHT_PX / ppm
    margin = SCREEN_MARGIN / ppm
    # Everything in meters from here on out
    left_x = map_width / 2 - lane_width + margin
    top_y = map_height / 2 + lane_width + margin
    left_vert = np.array(((left_x, margin),
                          (left_x, map_height + margin)))
    mid_vert_x = left_x + lane_width
    mid_vert = np.array(((mid_vert_x, margin),
                         (mid_vert_x, map_height + margin)))
    right_vert_x = left_x + lane_width * 2
    right_vert = np.array(((right_vert_x, margin),
                           (right_vert_x, map_height + margin)))
    top_horiz = np.array(((margin, top_y),
                          (map_width + margin, top_y)))
    mid_horiz_y = top_y - lane_width
    mid_horiz = np.array(((margin, mid_horiz_y),
                          (map_width + margin, mid_horiz_y)))
    bottom_horiz_y = top_y - lane_width * 2
    bottom_horiz = np.array(((margin, bottom_horiz_y),
                             (map_width + margin, bottom_horiz_y)))
    lines = left_vert, mid_vert, right_vert, top_horiz, mid_horiz, bottom_horiz
    return lines, lane_width


def main():
    coords = ((0, 1), (0.5, 1), (1, 1), (1.02, 0))
    zipped = list(zip(*coords))
    gen_map(x=np.array(zipped[0]),
            y=np.array(zipped[1]))


if __name__ == '__main__':
    main()
    # gen_random_map(should_plot=True)