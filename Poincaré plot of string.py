import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from pylab import *
# define a function to calculate our slopes at a given position #
# eta = (theta, omega, r, v)
def f(eta):
    # angle
    theta = eta[0]
    # angular speed
    omega = eta[1]
    # length of spring
    r = eta[2]
    # speed of the length of spring changing?
    v = eta[3]
    # slope of angle
    f_theta = omega
    # slope of r
    f_r = v
    # slope of angular speed
    f_omega = -2 * v * omega / r - (g / r) * sin(theta)
    # slope of v
    f_v = g * cos(theta) + r * omega ** 2 + (k / m) * (l - r)
    # return an array of our slopes
    return array([f_theta, f_omega, f_r, f_v], float)

m = 1.00
k = 40
l = 1.00
g = 9.81
omega0 = sqrt(g / l)
theta_initial_deg = 5.0
omega_initial_deg = 0.0
r_initial = l * 2.50
v_initial = 0.0
theta_initial = theta_initial_deg * pi / 180
omega_initial = omega_initial_deg * pi / 180

# set up a time interval domain
a = 0.0  # interval start
b = 100.0  # interval end
dt = 0.005  # timestep
t_points = arange(a, b, dt)  # array of times

# initial conditions eta = (theta, omega, r, v)
eta = array([theta_initial, omega_initial, r_initial, v_initial], float)

# create empty sets to update with values of interest, then invoke Runge-Kutta
theta_points = []
omega_points = []
r_points = []
v_points = []
x_points = []  # 新增 x 座標列表
y_points = []  # 新增 y 座標列表

for t in t_points:
    # add current conditions to lists
    theta_points.append(eta[0])
    omega_points.append(eta[1])
    r_points.append(eta[2])
    v_points.append(eta[3])

    # 計算 x 座標並儲存
    x_points.append(eta[2] * sin(eta[0]))
    # 計算 y 座標並儲存
    y_points.append(-eta[2] * cos(eta[0]))

    # calculate where we think we are going from slopes of current position
    # Runge-Kutta methods
    k1 = dt * f(eta)
    k2 = dt * f(eta + 0.5 * k1)
    k3 = dt * f(eta + 0.5 * k2)
    k4 = dt * f(eta + k3)
    eta += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# convert values to degrees to plot
theta_points_deg = []
omega_points_deg = []
for i in range(len(t_points)):
    theta_points_deg.append(theta_points[i] * 180 / pi)
    omega_points_deg.append(omega_points[i] * 180 / pi)

# Initialize lists to store Poincaré section points
poincare_theta = []
poincare_omega = []

# Define a condition for collecting Poincaré section data (for instance, every n time steps)
interval = 20  # Choose an appropriate interval
for i in range(len(theta_points)):
    if i % interval == 0:
        poincare_theta.append(theta_points[i])
        poincare_omega.append(omega_points[i])

# Plot the Poincaré section
plt.figure()
plt.scatter(poincare_theta, poincare_omega, s=5, c='red')
plt.xlabel('Theta (Angle)')
plt.ylabel('Omega (Angular Speed)')
plt.title('Poincaré Section of Elastic Pendulum')
plt.show()