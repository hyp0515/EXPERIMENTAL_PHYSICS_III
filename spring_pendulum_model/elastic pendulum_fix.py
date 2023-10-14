import matplotlib.animation as animation
import pandas as pd
from numpy import sqrt, sin, cos, arange, pi, append, array
from pylab import plot, xlabel, ylabel, title, show, axhline, savefig, subplots_adjust,figure, xlim, rcParams, rc, rc_context, subplot, tight_layout, axvline, legend


# define a function to calculate our slopes at a given position
# eta = (theta, omega, r, v) 
def f(eta):
    #angle
    theta = eta[0]
    #angular speed
    omega = eta[1]
    #length of spring
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
k = 24.0
l = 1.00
g = 9.81
omega0 = sqrt(g / l)
theta_initial_deg = 50.0
omega_initial_deg = 10.0
r_initial = l * 1.20
v_initial = 0.0
theta_initial = theta_initial_deg * pi / 180
omega_initial = omega_initial_deg * pi / 180

# set up a domain (time interval of interest)
a = 0.0                                 # interval start
b = 1.0                                # interval end
dt = 0.010                              # timestep
t_points = arange(a,b,dt)               # array of times

# initial conditions eta = (theta, omega, r, v)
eta = array([theta_initial, omega_initial, r_initial, v_initial], float)

# create empty sets to update with values of interest, then invoke Runge-Kutta
theta_points = []
omega_points = []
r_points = []
v_points = []
for t in t_points:
    # add current conditions to lists
    theta_points.append(eta[0])
    omega_points.append(eta[1])
    r_points.append(eta[2])
    v_points.append(eta[3])

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

# animate our results
# start with styling options
rcParams.update({'font.size': 18})
rc('axes', linewidth=2)
with rc_context({'axes.edgecolor': 'white', 'xtick.color': 'white','ytick.color': 'white', 'figure.facecolor': 'darkslategrey','axes.facecolor': 'darkslategrey', 'axes.labelcolor': 'white','axes.titlecolor': 'white'}):
    fig = figure(figsize=(10, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    ax_pend = subplot(2, 2, 1, aspect='equal', adjustable='datalim')
    ax_pend.tick_params(axis='both', colors="darkslategrey")

    ax_phase = subplot(2, 2, 2)
    axvline(color='k', lw=1)
    axhline(color='k', lw=1)
    title("Phase Diagram")
    xlabel("Angle (deg)")
    ylabel("Angular Speed (deg/s)")

    ax_ang = subplot(2, 2, (3, 4))
    axhline(color='k', lw=1)
    title("Angle Vs. Time")
    xlabel("Time (s)")
    ylabel("Angle (deg)")

    ims = []
    index = 0
    while index <= len(t_points) - 1:
        ln, = ax_pend.plot([0, (r_points[index]) * sin(theta_points[index])],[0, -(r_points[index]) * cos(theta_points[index])],color='white', lw=3)
        bob, = ax_pend.plot((r_points[index]) * sin(theta_points[index]),-(r_points[index]) * cos(theta_points[index]), 'o',markersize=22, color="darkturquoise", zorder=100)
        phase_curve, = ax_phase.plot(theta_points_deg[:(index + 1)],omega_points_deg[:(index + 1)],color="coral", lw=1.6)
        phase_dot, = ax_phase.plot(theta_points_deg[index:(index + 1)],omega_points_deg[index:(index + 1)],color="darkturquoise", marker="o", markersize=14)

        ang, = ax_ang.plot(t_points[:(index + 1)], theta_points_deg[:(index + 1)],color="darkturquoise", lw=2.8)
        ims.append([ln, bob, phase_curve, phase_dot, ang])
        index += 8

# save as a animation
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save('C:/Users/wayne/桌面/elastic_pendulum_num.mp4', writer=writervideo)

# save as a excel
data = {
    'Time (s)': t_points,                # time
    'Angle (deg)': theta_points_deg,     # angel
    'Angular Speed (deg/s)': omega_points_deg  # angular velocity
    }
df = pd.DataFrame(data)
df.to_excel('C:/Users/wayne/桌面/elastic_pendulum_num.xlsx', index=False)
