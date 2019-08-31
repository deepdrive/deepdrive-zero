import numpy as np
from scipy.interpolate import interp1d


# TODO: @njit
def gen_map(should_plot=False, num_course_points=4,
            resolution=25) -> np.array:
    # TODO: Linear interp with 2 points for straight roads
    # TODO: Randomize spacing
    # TODO: Different interpolations
    # TODO: Randomize num course points
    x = np.linspace(0, num_course_points, num=num_course_points + 1,
                    endpoint=True)
    y = np.random.rand(num_course_points + 1)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')

    xnew = np.linspace(0, num_course_points,
                       num=num_course_points * resolution, endpoint=True)
    if should_plot:
        import matplotlib.pyplot as plt
        plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
        plt.legend(['data', 'linear', 'cubic'], loc='best')
        plt.show()
    return xnew


if __name__ == '__main__':
    gen_map(should_plot=True)
