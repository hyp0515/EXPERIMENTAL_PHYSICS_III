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
data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_55cm/couple_5_L.xlsx")
data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_55cm/couple_5_R.xlsx")
# sync time
new_t_L = np.linspace(0, 131, len(data_L['t']))  
new_t_R = np.linspace(0, 131, len(data_R['t']))
 
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
    p0=[0.19, 0.0045, np.pi/95, 2*np.pi/1.35, 0, 0, 0.009],
)  
params_L, covariance_L = curve_fit(
    sin_function, 
    new_t_L, 
    data_L['x'],
    p0=[0.1404, 0.003885, np.pi/91.15, 2*np.pi/1.3, 0, 0, 0.009],
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
T = 131  # Total duration in seconds
N = len(data_L['t'])  # Total number of data points
fs = N / T  # Sampling rate in Hz

# Generate a time vector
t = np.linspace(0, T, N, endpoint=False)  # Time vector from 0 to T seconds
y_noisy = data_L['x']

# Apply FFT
yf = np.fft.fft(y_noisy)
xf = np.fft.fftfreq(N, 1/fs)

# Plotting
plt.subplot(2,1,1)
plt.plot(t, y_noisy)
plt.title('Noisy Signal')

plt.subplot(2,1,2)
plt.plot(xf, 2/N * np.abs(yf))
plt.title('Magnitude Spectrum')
plt.xlim([0, fs/2])  # Display only positive frequencies up to Nyquist frequency
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
from scipy.signal import find_peaks
yf = yf[:t.size//2]
xf = xf[:t.size//2]
peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)

print("Identified Peak Frequencies (Hz):", xf[peaks])
print(len(data_R['t']))