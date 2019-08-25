import numpy as np
from scipy.interpolate import interp1d

total_points = 3
x = np.linspace(0, total_points, num=total_points+1, endpoint=True)
y = np.random.rand(total_points+1)
# y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, total_points, num=77, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
