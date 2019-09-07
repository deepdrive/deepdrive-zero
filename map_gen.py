import numpy as np
from scipy.interpolate import interp1d

GAP_M = 1


# TODO: @njit
def gen_map(should_plot=False, num_course_points=3, resolution=25,
            make_equidistant=True, should_save=True,
            map_width=None,
            map_height=None,
            screen_margin=None) -> np.array:
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
    linear_interp = interp1d(x, y)

    # TODO: Remove first interpolation if making equidistant
    cubic_interp = interp1d(x, y, kind='cubic')

    xnew = np.linspace(min_x, max_x,
                       num=num_course_points * resolution, endpoint=True)

    ynew = cubic_interp(xnew)

    # Normalize 0->1
    ynew = (ynew - ynew.min()) / (ynew.max() - ynew.min())

    if not make_equidistant:
        if should_plot or should_save:
            import matplotlib.pyplot as plt
            if should_plot:
                plt.plot(x, y, 'o', xnew, linear_interp(xnew), '-',
                         xnew, ynew, '--')
                plt.legend(['data', 'linear', 'cubic'], loc='best')
                plt.show()
            elif should_save:
                plt.plot(xnew, ynew, '--')
                plt.savefig('images/map.png')
        retx = xnew
        rety = ynew
    else:
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
                plt.plot(xnew, ynew, '--')
                plt.axis('off')

                plt.savefig('images/map.png', bbox_inches='tight', pad_inches=0)

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


if __name__ == '__main__':
    gen_map(should_plot=True, make_equidistant=False)
