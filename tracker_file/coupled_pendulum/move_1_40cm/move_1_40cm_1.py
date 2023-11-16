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
data_L['t'] = np.linspace(0, 69, len(data_L['t']), endpoint=False)  
data_R['t'] = np.linspace(0, 69, len(data_R['t']), endpoint=False)
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

params_L, covariance_L = curve_fit(
    sin_function, 
    data_L['t'], 
    data_L['x'],
    p0=[1, 0.01, np.pi/23.3, 2*np.pi/1.45, 0, 0, 0.009],
)  

params_R, covariance_R = curve_fit(
    sin_function, 
    data_R['t'], 
    data_R['x'],
    p0=[1, 0.01, np.pi/23.3, 2*np.pi/1.45, 0, 0, 0.009],
)
data_L['x'] = data_L['x'] - params_L[-1]
data_R['x'] = data_R['x'] - params_R[-1]
# plt.scatter(data_R['t'], data_R['x'], label = 'experiment')
# plt.plot(data_R['t'], sin_function(data_R['t'], *params_R)- params_R[-1], label = 'fitted result')
# plt.scatter(data_L['t'], data_L['x'], label = 'experiment')
# plt.plot(data_L['t'], sin_function(data_L['t'], *params_L)- params_L[-1], label = 'fitted result') 
# plt.legend()
# plt.title("Original data and fitted result")
# # plt.xlim((10,60))
# plt.xlabel(r"t[s]")
# plt.ylabel(r"x[m]")
# plt.savefig('./Figures/move_1_40cm_1')
# plt.show()
###############################################################################
T = 69  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = data_L['t'].copy()  # Time vector from 0 to T seconds
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
# plt.xlim([0.4, 1])  # Display only positive frequencies up to Nyquist frequency
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.savefig('./Figures/fft_move_1')
# plt.show()


# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
# phase_angles = np.angle(yf[peaks])

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
# plt.show()
###############################################################################
# import sys
# sys.path.append('./coupled_pendulum_model')
# from coupled_pendulum_model import move_1_coupled_pendulum
# L = 0.41
# s = 0.18
# d = 0.21
# md = move_1_coupled_pendulum(D=0.40, s=s, d=0.21, L=0.41)
# print(md.omega_s/(2*np.pi), md.omega_a/(2*np.pi))

# def sin_function_refit(t, decay, A, B, e,):
#     omega_s = md.omega_s/(2*np.pi)
#     omega_a = md.omega_a/(2*np.pi)
#     symmetry = A*np.cos(2*np.pi*omega_s*t+phase_angles[1])
#     antisymmetry = B*np.cos(2*np.pi*omega_a*t+phase_angles[0])
#     return np.exp(-decay*t)*(symmetry+antisymmetry)+e

# def sin_function_refit(t, A, decay, d, e):
#     omega_s = md.omega_s
#     omega_a = md.omega_a
#     return A*np.exp(-decay*t)*np.cos(0.5*(omega_a-omega_s)*t+d)*np.cos(0.5*(omega_a+omega_s)*t+e)

# param, cov = curve_fit(
#     sin_function_refit,
#     t,
#     data_L['x']-params_L[-1],
#     p0=[0.5, 4e-3, 0, 0]   
# )


# print(param)
# plt.plot(t, data_L['x']-params_L[-1], label = 'raw data')
# plt.plot(t, sin_function_refit(t, *param), label = 'fitted')
# plt.show()