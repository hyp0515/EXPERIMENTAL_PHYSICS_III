import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit, minimize
###############################################################################
# Data 
def extract_data_from_path(path):
    df = pd.read_excel(path, engine='openpyxl')
    raw_data = np.array(df)
    raw_data[:, 5] = np.deg2rad(raw_data[:, 5])
    columns = ['t', 'x', 'y', 'r', 'v', 'theta']
    data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
    return data
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_2_55cm/couple_6_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_2_55cm/couple_6_R.xlsx")
# sync time
new_t_L = np.linspace(0, 140, len(data_L['t']))  
new_t_R = np.linspace(0, 142, len(data_R['t']))
 
# plt.plot(new_t_L, data_L['x'], color = 'red')
# plt.plot(new_t_R, data_R['x'], color = 'blue')
# plt.show()
###############################################################################
# Convenient function for getting index of time data and fitting
def second_to_index_L(time):
    return np.searchsorted(new_t_L, time)
def second_to_index_R(time):
    return np.searchsorted(new_t_R, time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f, g):   
    return A*np.exp(-decay*t)*(np.cos(omega_1*t+d)+e)*np.sin(omega_2*t+f)+g

params_R, covariance_R = curve_fit(
    sin_function, 
    new_t_R, 
    data_R['x'],
    p0=[0.05, 0.006, np.pi/50, 2*np.pi/1.3, 0.121, 2, 0, 0.023],
)  
params_L, covariance_L = curve_fit(
    sin_function, 
    new_t_L, 
    data_L['x'],
    p0=[0.05, 0.006, np.pi/50, 2*np.pi/1.3, 0.121, 2, 0, 0.023],
)  

print(params_R)
print(params_L)
plt.scatter(new_t_R, data_R['x'], label = 'experiment')
plt.plot(new_t_R, sin_function(new_t_R, *params_R), label = 'fitted result')
plt.scatter(new_t_L, data_L['x'], label = 'experiment')
plt.plot(new_t_L, sin_function(new_t_L, *params_L), label = 'fitted result')
plt.legend()
plt.title("Original data and fitted result")
plt.xlabel(r"t")
plt.ylabel(r"x")
plt.show()
###############################################################################