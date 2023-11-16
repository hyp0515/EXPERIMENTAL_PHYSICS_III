import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def extract_data_from_path(path):
    df = pd.read_excel(path, engine='openpyxl')
    raw_data = np.array(df)
    raw_data[:, 4] = np.deg2rad(raw_data[:, 4])
    columns = ['t', 'x', 'y', 'r', 'theta']
    data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
    return data

data_L = extract_data_from_path("./tracker_file/coupled_pendulum/different_length/different_length_1_L.xlsx")

data_L['t'] = np.linspace(0, 93, len(data_L['t']), endpoint=False)

plt.plot(data_L['t'], data_L['x'])
plt.xlim((20,60))
plt.xlabel(r"t[s]")
plt.ylabel(r"x[m]")
plt.title('Different Length')
plt.savefig('./Figures/different_length')
plt.show()



# T = 93  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = data_L['t'].copy()  # Time vector from 0 to T seconds
# y_noisy = data_L['x']-np.mean(data_L['x'])

# # Apply FFT
# yf = np.fft.fft(y_noisy)
# xf = np.fft.fftfreq(N, 1/fs)

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

# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.007)
# phase_angles = np.angle(yf[peaks])

# print("Identified Peak Frequencies (Hz):", xf[peaks])
