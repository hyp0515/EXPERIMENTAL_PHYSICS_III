import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit, minimize
###############################################################################
# Data 
file_path = "nylon_24_simple_1.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')  
'''
Make sure to install "openpyxl" by typing "pip install pandas openpyxl"
'''
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta', 'p', 'k']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
'''
Data can be easily extracted by keys e.g. data['t'] for time array etc.
'''
r = np.mean(data['r'])
x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])
###############################################################################
# Convenient function for getting index of time data
def second_to_index(time):
    return np.searchsorted(data['t'], time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega, c, d):   
    return A*np.exp(-decay*t)*np.sin(omega*t+c)+d

params, covariance = curve_fit(
    sin_function, 
    data['t'], 
    data['x'], 
    p0=[x_amp, 0.1, 0.2*np.pi, 0, 0],
)  
'''
The order of the params is according to the input of sin_function, ignoring 't'
A     : Amplitude
decay : decay rate of damping
omega : 2*pi/period
c, d  : parameters for a better fitting result
'''
plt.scatter(data['t'], data['x'], label = 'experiment', color = 'red')
plt.plot(data['t'], sin_function(data['t'], *params), label = 'fitted result')
plt.legend()
plt.title("Original data and fitted result")
plt.xlabel(r"t")
plt.ylabel(r"x displacement")
plt.show()
###############################################################################
# Phase Diagram (and fit data with ellipse)

# Using the method from ChatGPT and it said the method is 
# 'Ellipse fitting using algebraic distance minimization'
# The ellipse parameters are extracted from 
# the algebraic parameters using eigenvalue decomposition.

theta = data['theta'].copy()
dtheta = theta[1:]-theta[:-1]
theta = theta[1:]
t = data['t'].copy()
dt = t[1:]-t[:-1]
omega = dtheta/dt  # Angular velocity

def fitEllipse(x, y):
    x, y = x[:,np.newaxis], y[:,np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b - a*c
    x0 = (c*d - b*f) / num
    y0 = (a*f - b*d) / num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b / (a - c)) / 2
        else:
            return np.pi/2 + np.arctan(2*b / (a - c)) / 2

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b - a*c) * ((c - a) * np.sqrt(1 + 4*b*b / ((a - c) * (a - c))) - (c + a))
    down2 = (b*b - a*c) * ((a - c) * np.sqrt(1 + 4*b*b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


# Fit ellipse
a = fitEllipse(theta, omega)
center = ellipse_center(a)
phi = ellipse_angle_of_rotation(a)
axes = ellipse_axis_length(a)

fig, ax = plt.subplots()
ax.scatter(theta, omega, color='blue', s=5, label='Data')
ellipse = Ellipse(xy=center, width=2*axes[0], height=2*axes[1], angle=np.degrees(phi), edgecolor='r', facecolor='none')
ax.add_patch(ellipse)
plt.title("Phase Diagram")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\omega$")
plt.legend()
plt.show()
