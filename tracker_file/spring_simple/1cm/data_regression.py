import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "./tracker_file/spring_simple/1cm/spring_1_simple_2.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta', 'p', 'k']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}

r = np.mean(data['r'])
x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])

# plt.plot(data['t'], data['r']-0.4175)
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
    p0=[x_amp, 0.1, 0.2*np.pi, 0, 0],
)
# print(x_amp)
# print(params)

# plt.scatter(data['t'], data['x'], label = 'experiment', color = 'red')
# plt.plot(data['t'], sin_function(data['t'], *params), label = 'fitted result')
# plt.legend()
# plt.show()
###############################################################################

print(len(data['t']))

t_remake = np.linspace(0, (142/3420)*(len(data['t'])), len(data['t']))

x = data['x']

dx = x[1:]-x[:-1]
dt = t_remake[1:]-t_remake[:-1]
v_x = dx/dt

k = 0.5*0.336*v_x**2
print(np.max(k))