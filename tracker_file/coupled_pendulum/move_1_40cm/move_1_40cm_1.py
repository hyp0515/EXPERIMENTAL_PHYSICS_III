import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
###############################################################################
# Data 
def extract_data_from_path(path):
    df = pd.read_excel(path, engine='openpyxl')
    raw_data = np.array(df)
    raw_data[:, 5] = np.deg2rad(raw_data[:, 5])
    columns = ['t', 'x', 'y', 'r', 'v', 'theta']
    data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
    return data
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/move_1_40cm_1_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/move_1_40cm_1_R.xlsx")
# sync time
new_t_L = np.linspace(0, 69, len(data_L['t']))  
new_t_R = np.linspace(0, 69, len(data_R['t']))
 
# plt.plot(new_t_L, data_L['x'], color = 'red')
# plt.plot(new_t_R, data_R['x'], color = 'blue')
# plt.show()
###############################################################################
# Convenient function for getting index of time data
def second_to_index_L(time):
    return np.searchsorted(new_t_L, time)
def second_to_index_R(time):
    return np.searchsorted(new_t_R, time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f):   
    return A*np.exp(-decay*t)*np.cos(omega_1*t+d)*np.sin(omega_2*t+e)+f

params_R, covariance_R = curve_fit(
    sin_function, 
    new_t_R, 
    data_R['x'],
    p0=[1, 0.01, np.pi/23.3, 2*np.pi/1.45, 0, 0, 0.009],
)  
params_L, covariance_L = curve_fit(
    sin_function, 
    new_t_L, 
    data_L['x'],
    p0=[1, 0.01, np.pi/23.3, 2*np.pi/1.45, 0, 0, 0.009],
)  

# print(params_R)
# print(params_L)
# plt.scatter(new_t_R, data_R['x']-params_R[-1], label = 'experiment')
# plt.plot(new_t_R, sin_function(new_t_R, *params_R)-params_R[-1], label = 'fitted result')
# plt.scatter(new_t_L, data_L['x']-params_L[-1], label = 'experiment')
# plt.plot(new_t_L, sin_function(new_t_L, *params_L)-params_L[-1], label = 'fitted result')
# plt.legend()
# plt.title("Original data and fitted result")
# plt.xlabel(r"t")
# plt.ylabel(r"x")
# plt.show()
###############################################################################
T = 69  # Total duration in seconds
N = len(data_L['t'])  # Total number of data points
fs = N / T  # Sampling rate in Hz

# Generate a time vector
t = np.linspace(0, T, N, endpoint=False)  # Time vector from 0 to T seconds
y_noisy = data_L['x']-params_L[-1]

# Apply FFT
yf = np.fft.fft(y_noisy)
xf = np.fft.fftfreq(N, 1/fs)

# Plotting
# plt.subplot(2,1,1)
# plt.plot(t, y_noisy)
# plt.title('Noisy Signal')

# plt.subplot(2,1,2)
# plt.plot(xf, 2/N * np.abs(yf))
# plt.title('Magnitude Spectrum')
# plt.xlim([0, 1])  # Display only positive frequencies up to Nyquist frequency
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.show()


yf = yf[:t.size//2]
xf = xf[:t.size//2]
peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
phase_angles = np.angle(yf[peaks])

# print("Identified Peak Frequencies (Hz):", xf[peaks]*2*np.pi)
# print((xf[peaks[0]]+xf[peaks[1]])*np.pi)
# print((-xf[peaks[0]]+xf[peaks[1]])*np.pi)

def sin_function_refit(t, decay, A, B, e):
    omega_s = xf[peaks[1]]
    omega_a = xf[peaks[0]]
    symmetry = A*np.cos(2*np.pi*omega_s*t+phase_angles[1])
    antisymmetry = B*np.cos(2*np.pi*omega_a*t+phase_angles[0])
    return np.exp(-decay*t)*(symmetry+antisymmetry)+e

param, cov = curve_fit(
    sin_function_refit,
    t,
    data_L['x']-params_L[-1],
    p0=[4e-3, 0, 0, 0]   
)

# print(param[1:-1])
# plt.plot(t, data_L['x']-params_L[-1], label = 'raw data')
# plt.plot(t, sin_function_refit(t, *param), label = 'fitted')
# plt.show()
###############################################################################
import sys
sys.path.append('./coupled_pendulum_model')
from coupled_pendulum_model import move_1_coupled_pendulum


D_array = np.linspace(0.21, 0.56, 100)
md = move_1_coupled_pendulum(D=D_array,s=0.18,d=0.21,L=0.41)  
plt.plot(D_array, md.omega_s, label='s')
plt.plot(D_array, md.omega_a, label='a')
plt.legend()
# plt.plot(D_array, md.N)
plt.scatter(0.40, xf[peaks[1]]*2*np.pi, label='s')
plt.scatter(0.40, xf[peaks[0]]*2*np.pi, label='a')
plt.legend()
plt.show()
# print(md.omega_a, md.omega_s)
