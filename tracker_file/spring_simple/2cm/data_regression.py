import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "./tracker_file/spring_simple/2cm/spring_2_simple_1.xlsx"

df = pd.read_excel(file_path, engine='openpyxl')
raw_data = np.array(df)
raw_data[:, 5] = np.deg2rad(raw_data[:, 5])

columns = ['t', 'x', 'y', 'r', 'v', 'theta', 'p', 'k']
data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}


x_mean = np.mean(data['x'])
x_amp = np.max(data['x'])-x_mean
theta_mean = np.mean(data['theta'])

# plt.plot(data['t'], data['r'])
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

# plt.scatter(data['t'], data['x'], label = 'experiment', color = 'red')
# plt.plot(data['t'], sin_function(data['t'], *params), label = 'fitted result')
# plt.legend()
# plt.show()

###############################################################################
# theta = data['theta'].copy()
# dtheta = theta[1:]-theta[:-1]
# theta = theta[1:]
# t = data['t'].copy()
# dt = t[1:]-t[:-1]
# omega = dtheta/dt  # Angular velocity

# plt.scatter(theta, omega)
# plt.show()

t_remake = np.linspace(0, (5/240)*(len(data['t'])), len(data['t']))

x = data['x']

dx = x[1:]-x[:-1]
dt = t_remake[1:]-t_remake[:-1]
v_x = dx/dt
print(np.max(v_x))
k = 0.5*0.336*v_x**2
# print(np.max(k))
# print((5/240)*(len(data['t'])))

# print(0.3361*9.8/54.978)

r = data['r'].copy()
dr = r-(0.3361*9.8/54.978)-0.34

u_e = 0.5*54.978*dr[1:]**2
e_k = k

e_total = u_e+e_k
# plt.plot(t_remake[1:], u_e)
# plt.show()

yf = np.fft.fft(dr)
xf = np.fft.fftfreq(len(dr), 5/240)

plt.subplot(2,1,1)
plt.plot(t_remake, dr)
plt.title('Noisy Signal')

plt.subplot(2,1,2)
plt.plot(xf, 2/len(dr) * np.abs(yf))
plt.title('Magnitude Spectrum')
# plt.xlim([0, 1])  # Display only positive frequencies up to Nyquist frequency
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()

