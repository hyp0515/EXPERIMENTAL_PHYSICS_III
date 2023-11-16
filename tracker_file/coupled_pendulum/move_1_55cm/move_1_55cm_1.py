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
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_55cm/move_1_55cm_1_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_55cm/move_1_55cm_1_R.xlsx")
# sync time
data_L['t'] = np.linspace(0, 131, len(data_L['t']))  
data_R['t'] = np.linspace(0, 131, len(data_R['t']))
 
# plt.plot(new_t_L, data_L['x'], color = 'red')
# plt.plot(new_t_R, data_R['x'], color = 'blue')
# plt.show()
###############################################################################
# Convenient function for getting index of time data
def second_to_index_L(time):
    return np.searchsorted(data_L['t'], time)
def second_to_index_R(time):
    return np.searchsorted(data_R['t'], time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f):   
    return A*np.exp(-decay*t)*np.cos(omega_1*t+d)*np.sin(omega_2*t+e)+f

params_R, covariance_R = curve_fit(
    sin_function, 
    data_R['t'], 
    data_R['x'],
    p0=[0.19, 0.0045, np.pi/95, 2*np.pi/1.35, 0, 0, 0.009],
)  
params_L, covariance_L = curve_fit(
    sin_function, 
    data_L['t'], 
    data_L['x'],
    p0=[0.1404, 0.003885, np.pi/91.15, 2*np.pi/1.3, 0, 0, 0.009],
)  
data_L['x'] = data_L['x'] - params_L[-1]
data_R['x'] = data_R['x'] - params_R[-1]
# print(params_R)
# print(params_L)
plt.scatter(data_R['t'], data_R['x'], label = 'experiment')
plt.plot(data_R['t'], sin_function(data_R['t'], *params_R)- params_R[-1], label = 'fitted result')
plt.scatter(data_L['t'], data_L['x'], label = 'experiment')
plt.plot(data_L['t'], sin_function(data_L['t'], *params_L)- params_L[-1], label = 'fitted result')
plt.legend()
plt.title("Original data and fitted result")
plt.xlabel(r"t[s]")
plt.ylabel(r"x[m]")
plt.savefig('./Figures/move_1_55cm_1')
plt.show()
###############################################################################
T = 131  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = data_L['t']  # Time vector from 0 to T seconds
# y_noisy = data_L['x']

# # Apply FFT
# yf = np.fft.fft(y_noisy)
# xf = np.fft.fftfreq(N, 1/fs)

# # Plotting
# plt.subplot(2,1,1)
# plt.plot(t, y_noisy)
# plt.title('Original data')

# plt.subplot(2,1,2)
# plt.plot(xf, 2/N * np.abs(yf))
# plt.title('Magnitude Spectrum')
# plt.xlim([0.6, 0.9])  # Display only positive frequencies up to Nyquist frequency
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.savefig('./Figures/fft_move_1_55cm')
# plt.show()
# from scipy.signal import find_peaks
# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
# peaks = np.append(peaks,peaks+2)
# print("Identified Peak Frequencies (Hz):", xf[peaks])
# print(2/N*np.abs(yf[peaks]), 2/N*np.abs(yf[peaks+1]), 2/N*np.abs(yf[peaks+2]), 2/N*np.abs(yf[peaks+3]))
# print(len(data_R['t']))