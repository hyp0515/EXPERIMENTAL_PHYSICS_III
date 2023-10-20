import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

g = 9.81
k = 0.1

file_path = "./tracker_file/spring_simple/3cm/spring_3_simple_2.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta', 'p', 'k']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}

r = np.mean(data['r'])
x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])
y0 = np.min(data['y'])
dy = data['y']-y0
dr = data['r']-r
mass = 0.3361
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
    p0=[x_amp, 0.1, 0.2*np.pi, 0, 0],
)
# print(x_amp)
# print(params)

# plt.scatter(data['t'], data['x'], label = 'experiment', color = 'red')
# plt.plot(data['t'], sin_function(data['t'], *params), label = 'fitted result')
# plt.legend()
# plt.show()

###############################################################################
theta = data['theta'].copy()
dtheta = theta[1:]-theta[:-1]
theta = theta[1:]
t = data['t'].copy()
dt = t[1:]-t[:-1]
omega = dtheta/dt  # Angular velocity

# plt.scatter(theta[200:300], omega[200:300])
# plt.scatter(theta[-100:], omega[-100:])
# plt.scatter(theta[:100], omega[:100])
# plt.scatter(data['x'][:100], data['y'][:100])
# plt.scatter(data['x'][-100:], data['y'][-100:])
# r_array = r * np.ones((len(data['r'])))
# plt.plot(data['t'], data['r'])
# plt.plot(data['t'], r_array)
# plt.show()


# plt.plot(data['t'], data['k'])
# plt.show()

# K  = data['k']
# Ug = mass * g * dy
# Uk = 0.5*k*dr**2

t_remake = np.linspace(0, (142/3420)*(len(data['t'])), len(data['t']))

x = data['x']

dx = x[1:]-x[:-1]
dt = t_remake[1:]-t_remake[:-1]
v_x = dx/dt

k = 0.5*0.336*v_x**2
print(np.max(k))
