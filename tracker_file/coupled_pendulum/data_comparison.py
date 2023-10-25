import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import importlib
import sys
sys.path.append('./coupled_pendulum_model')
from coupled_pendulum_model import move_1_coupled_pendulum

L = 0.41
s = 0.18
d = 0.21

def model_with_different_D(D=False):  # D in m
    if D is False:        
        D_array = np.linspace(0.21, 0.56, 100)
    else:
        D_array = D
    return D_array, move_1_coupled_pendulum(D=D_array,s=s,d=d,L=L) 


def extract_data_from_module(move, D, num=1):  # D in cm    
    path = f"./tracker_file/coupled_pendulum/move_{move}_{D}cm/"
    sys.path.append(path)
    data_module = f"move_{move}_{D}cm_{num}"
    df = importlib.import_module(data_module)  
    return df



D_array, md = model_with_different_D()
plt.plot(D_array, md.omega_s, label='s', color = 'r')
plt.plot(D_array, md.omega_a, label='a', color = 'y')
for D in [40, 55]:
    if D == 40:
        for num in range(1,3):
            df = extract_data_from_module(1, D, num=num)
            yf_L = np.fft.fft(df.data_L['x']-df.params_L[-1])
            xf_L = np.fft.fftfreq(len(df.data_L['t']), df.T/len(df.data_L['t']))
            yf_L = yf_L[:df.data_L['t'].size//2]
            xf_L = xf_L[:df.data_L['t'].size//2]
            peaks, _ = find_peaks(2/len(df.data_L['t'])*np.abs(yf_L), height=0.01)
            plt.scatter(D*0.01, xf_L[peaks[1]]*2*np.pi, label='s', color = 'r')
            plt.scatter(D*0.01, xf_L[peaks[0]]*2*np.pi, label='a', color = 'y')
            yf_R = np.fft.fft(df.data_R['x']-df.params_R[-1])
            xf_R = np.fft.fftfreq(len(df.data_R['t']), df.T/len(df.data_R['t']))
            yf_R = yf_R[:df.data_R['t'].size//2]
            xf_R = xf_R[:df.data_R['t'].size//2]
            peaks, _ = find_peaks(2/len(df.data_R['t'])*np.abs(yf_R), height=0.01)
            plt.scatter(D*0.01, xf_R[peaks[1]]*2*np.pi, label='s', color = 'r')
            plt.scatter(D*0.01, xf_R[peaks[0]]*2*np.pi, label='a', color = 'y')
    else:
        df = extract_data_from_module(1, D)
        yf_L = np.fft.fft(df.data_L['x']-df.params_L[-1])
        xf_L = np.fft.fftfreq(len(df.data_L['t']), df.T/len(df.data_L['t']))
        yf_L = yf_L[:df.data_L['t'].size//2]
        xf_L = xf_L[:df.data_L['t'].size//2]
        peaks, _ = find_peaks(2/len(df.data_L['t'])*np.abs(yf_L), height=0.01)
        plt.scatter(D*0.01, xf_L[peaks[1]]*2*np.pi, label='s', color = 'r')
        plt.scatter(D*0.01, xf_L[peaks[0]]*2*np.pi, label='a', color = 'y')
        yf_R = np.fft.fft(df.data_R['x']-df.params_R[-1])
        xf_R = np.fft.fftfreq(len(df.data_R['t']), df.T/len(df.data_R['t']))
        yf_R = yf_R[:df.data_R['t'].size//2]
        xf_R = xf_R[:df.data_R['t'].size//2]
        peaks, _ = find_peaks(2/len(df.data_R['t'])*np.abs(yf_R), height=0.005)
        plt.scatter(D*0.01, xf_R[peaks+2]*2*np.pi, label='s', color = 'r')
        plt.scatter(D*0.01, xf_R[peaks]*2*np.pi, label='a', color = 'y')
plt.show()

# T = 69  # Total duration in seconds
# N = len(data_L['t'])  # Total number of data points
# fs = N / T  # Sampling rate in Hz

# # Generate a time vector
# t = data_L['t'].copy()  # Time vector from 0 to T seconds
# y_noisy = data_L['x']-params_L[-1]

# # Apply FFT
# yf = np.fft.fft(y_noisy)
# xf = np.fft.fftfreq(N, 1/fs)

# yf = yf[:t.size//2]
# xf = xf[:t.size//2]
# peaks, _ = find_peaks(2/N*np.abs(yf), height=0.01)
# phase_angles = np.angle(yf[peaks])


# D_array = np.linspace(0.21, 0.56, 100)
# md = move_1_coupled_pendulum(D=D_array,s=0.18,d=0.21,L=0.41)  
# plt.plot(D_array, md.omega_s, label='s')
# plt.plot(D_array, md.omega_a, label='a')
# plt.legend()
# # plt.plot(D_array, md.N)
# plt.scatter(0.40, xf[peaks[1]]*2*np.pi, label='s')
# plt.scatter(0.40, xf[peaks[0]]*2*np.pi, label='a')
# plt.legend()
# plt.show()