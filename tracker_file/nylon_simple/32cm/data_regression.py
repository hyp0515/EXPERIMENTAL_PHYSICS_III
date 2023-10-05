import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "nylon_32_simple_3.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta', 'p', 'k']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}

r = np.mean(data['r'])
x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])

# plt.plot(data['t'], data['x'])
# plt.show()
###############################################################################
def second_to_index(time):
    return np.searchsorted(data['t'], time)
###############################################################################

def sin_function(t, A, decay, omega, c, d):   
    return A*np.exp(-decay*t)*np.sin(omega*t+c)+d

params, covariance = curve_fit(
    sin_function, 
    data['t'], 
    data['x'], 
    p0=[x_amp, 0.01, 0.2*np.pi, 0, 0],
)
# print(x_amp)
# print(params)

plt.scatter(data['t'], data['x'], label = 'experiment', color = 'red')
plt.plot(data['t'], sin_function(data['t'], *params), label = 'fitted result')
plt.legend()
plt.show()