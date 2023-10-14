import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit, minimize
###############################################################################
# Data 
file_path = "couple_5_l.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')  
'''
Make sure to install "openpyxl" by typing "pip install pandas openpyxl"
'''
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
'''
Data can be easily extracted by keys e.g. data['t'] for time array etc.
'''
r = np.mean(data['r'])
x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])
new_t = np.linspace(0, 71, len(data['t']))
# plt.plot(new_t, data['x'])
# plt.show()
###############################################################################
# Convenient function for getting index of time data
def second_to_index(time):
    return np.searchsorted(new_t, time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f):   
    return A*np.exp(-decay*t)*np.cos(omega_1*t+f)*np.sin(omega_2*t+d)+e

params, covariance = curve_fit(
    sin_function, 
    new_t, 
    data['x'],
    p0=[0.128, 0.00461, np.pi/25.666, 2*np.pi/1.52, 0, 0.009, 0]
)  
# '''
# The order of the params is according to the input of sin_function, ignoring 't'
# A     : Amplitude
# decay : decay rate of damping
# omega : 2*pi/period
# c, d  : parameters for a better fitting result
# '''
print(params)
plt.scatter(new_t, data['x'], label = 'experiment', color = 'red')
plt.plot(new_t, sin_function(new_t, *params), label = 'fitted result')
plt.legend()
plt.title("Original data and fitted result")
plt.xlabel(r"t")
plt.ylabel(r"x")
plt.show()
###############################################################################
