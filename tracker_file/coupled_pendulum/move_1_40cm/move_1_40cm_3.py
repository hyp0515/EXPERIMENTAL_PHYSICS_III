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
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/move_1_40cm_3_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/move_1_40cm_3_R.xlsx")
# sync time
data_L['t'] = np.linspace(0, 73, len(data_L['t']))  
data_R['t'] = np.linspace(0, 73, len(data_R['t']))
###############################################################################
# Convenient function for getting index of time data
def second_to_index_L(time):
    return np.searchsorted(data_L['t'], time)
def second_to_index_R(time):
    return np.searchsorted(data_R['t'], time)
###############################################################################
# Fitting x displacement to get the period of the pendulum
def sin_function(t, A, decay, omega_1, omega_2, d, e, f):   
    return A*np.exp(-decay*t)*np.sin(omega_1*t+d)*np.sin(omega_2*t+e)+f

params_R, covariance_R = curve_fit(
    sin_function, 
    data_R['t'], 
    data_R['x'],
    p0=[-1, 0.01, np.pi/24.28, 2*np.pi/1.42, 0, 0.009, 0]
)  
params_L, covariance_L = curve_fit(
    sin_function, 
    data_L['t'], 
    data_L['x'],
    p0=[1, 0.01, np.pi/24.28, 2*np.pi/1.42, 0, 0.009, 0]
)
 
data_L['x'] = data_L['x'] - params_L[-1]
data_R['x'] = data_R['x'] - params_R[-1]
# print(params_R)
# print(params_L)
# plt.scatter(data_R['t'], data_R['x'], label = 'experiment')
# plt.plot(data_R['t'], sin_function(data_R['t'], *params_R)- params_R[-1], label = 'fitted result')
# plt.scatter(data_L['t'], data_L['x'], label = 'experiment')
# plt.plot(data_L['t'], sin_function(data_L['t'], *params_L)- params_L[-1], label = 'fitted result') 
# plt.legend()
# plt.title("Original data and fitted result")
# plt.xlim((10,60))
# plt.xlabel(r"t[s]")
# plt.ylabel(r"x[m]")
# plt.savefig('./Figures/fitted_result')
# plt.show()
# ###############################################################################
# import sys
# sys.path.append('./coupled_pendulum_model')
# import coupled_pendulum_model as model
#################################################################################
# Parameters
T = 73  # Total duration in seconds
# N = len(data_R['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = np.linspace(0, T, N, endpoint=False)  # Time vector from 0 to T seconds
# y_noisy = data_R['x']

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
# plt.xlim([0, fs/2])  # Display only positive frequencies up to Nyquist frequency
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.show()

##

# T = 73  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = np.linspace(0, T, N, endpoint=False)  # Time vector from 0 to T seconds
# y_noisy = data_L['x']

# # Apply FFT
# yf = np.fft.fft(y_noisy)
# xf = np.fft.fftfreq(N, 1/fs)

# Plotting
# plt.subplot(2,1,1)
# plt.plot(t, y_noisy)
# plt.title('Noisy Signal')

# plt.subplot(2,1,2)
# plt.plot(xf, 2/N * np.abs(yf))
# plt.title('Magnitude Spectrum')
# plt.xlim([0, fs/2])  # Display only positive frequencies up to Nyquist frequency
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.show()
# from scipy.signal import find_peaks
# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
# phase_angles = np.angle(yf[peaks])

# # print("Identified Peak Frequencies (Hz):", xf[peaks]*2*np.pi)
# # print(0.5*2*np.pi*(+xf[peaks][0]+xf[peaks][1]))

# import sys
# sys.path.append('./coupled_pendulum_model')
# from coupled_pendulum_model import move_1_coupled_pendulum


# D_array = np.linspace(0.21, 0.56, 100)
# md = move_1_coupled_pendulum(D=D_array,s=0.18,d=0.21,L=0.41)  
# plt.plot(D_array, md.omega_s, label='s')
# plt.plot(D_array, md.omega_a, label='a')
# # plt.legend()
# # plt.plot(D_array, md.N)
# plt.scatter(0.40, xf[peaks[1]]*2*np.pi, label='s')
# plt.scatter(0.40, xf[peaks[0]]*2*np.pi, label='a')
# plt.legend()
# plt.show()