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
    raw_data[:, 4] = np.deg2rad(raw_data[:, 4])
    columns = ['t', 'x', 'y', 'r', 'theta']
    data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
    return data
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_2_46cm/move_2_46cm_1_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_2_46cm/move_2_46cm_1_R.xlsx")
# sync time
data_L['t'] = np.linspace(0, 96, len(data_L['t']))  
new_t_R = np.linspace(0, 96, len(data_R['t']))
 
# plt.plot(new_t_L, data_L['x'], color = 'red')
# plt.plot(new_t_R, data_R['x'], color = 'blue')
# plt.show()
###############################################################################
# Convenient function for getting index of time data and fitting
def second_to_index_L(time):
    return np.searchsorted(data_L['t'], time)
def second_to_index_R(time):
    return np.searchsorted(new_t_R, time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f, g):   
    return A*np.exp(-decay*t)*(np.cos(omega_1*t+d)+e)*np.sin(omega_2*t+f)+g

# params_R, covariance_R = curve_fit(
#     sin_function, 
#     new_t_R, 
#     data_R['x'],
    
# )  
# params_L, covariance_L = curve_fit(
#     sin_function, 
#     new_t_L, 
#     data_L['x'],
    
# )  
data_L['x'] = data_L['x'] 
data_R['x'] = data_R['x'] 
# print(params_R)
# print(params_L)
# plt.scatter(new_t_R, data_R['x'], label = 'experiment')
# # plt.plot(new_t_R, sin_function(new_t_R, *params_R), label = 'fitted result')
# plt.scatter(new_t_L, data_L['x'], label = 'experiment')
# # plt.plot(new_t_L, sin_function(new_t_L, *params_L), label = 'fitted result')
# plt.legend()
# plt.title("Original data and fitted result")
# plt.xlabel(r"t")
# plt.ylabel(r"x")
# plt.show()
###############################################################################
T = 96  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = np.linspace(0, T, N, endpoint=False)  # Time vector from 0 to T seconds
# y_noisy = data_L['x']-params_L[-1]

# # Apply FFT
# yf = np.fft.fft(y_noisy)
# xf = np.fft.fftfreq(N, 1/fs)

# # Plotting
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

# from scipy.signal import find_peaks
# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
# phase_angles = np.angle(yf[peaks])

# print("Identified Peak Frequencies (Hz):", xf[peaks])
# print((xf[peaks[0]]+xf[peaks[1]])*np.pi)
# print((-xf[peaks[0]]+xf[peaks[1]])*np.pi)

# def sin_function_refit(t, decay, A, B, e):
#     omega_s = xf[peaks[1]]
#     omega_a = xf[peaks[0]]
#     symmetry = A*np.cos(2*np.pi*omega_s*t+phase_angles[1])
#     antisymmetry = B*np.cos(2*np.pi*omega_a*t+phase_angles[0])
#     return np.exp(-decay*t)*(symmetry+antisymmetry)+e

# param, cov = curve_fit(
#     sin_function_refit,
#     t,
#     data_L['x']-params_L[-1],
#     p0=[4e-3, 0, 0, 0]   
# )

# print(param[1:-1])
# plt.plot(t, data_L['x']-params_L[-1], label = 'raw data')
# plt.plot(t, sin_function_refit(t, *param), label = 'fitted')
# plt.legend()
# plt.show()