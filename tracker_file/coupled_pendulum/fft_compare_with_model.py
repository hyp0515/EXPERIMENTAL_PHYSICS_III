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
def find_omega(df):
    yf_L = np.fft.fft(df.data_L['x'])
    xf_L = np.fft.fftfreq(len(df.data_L['t']), df.T/len(df.data_L['t']))
    yf_L = yf_L[:df.data_L['t'].size//2]
    xf_L = xf_L[:df.data_L['t'].size//2]
    peaks_L, _ = find_peaks(2/len(df.data_L['t'])*np.abs(yf_L), height=0.005)
    yf_R = np.fft.fft(df.data_R['x'])
    xf_R = np.fft.fftfreq(len(df.data_R['t']), df.T/len(df.data_R['t']))
    yf_R = yf_R[:df.data_R['t'].size//2]
    xf_R = xf_R[:df.data_R['t'].size//2]
    peaks_R, _ = find_peaks(2/len(df.data_R['t'])*np.abs(yf_R), height=0.005)
    pL = xf_L[peaks_L]
    pR = xf_R[peaks_R]
    if len(pR)==1:
        pR = np.append(pR, xf_R[peaks_R+2])
        # pR = pR[::-1]
    return pL, pR
def plot(D, omega):
    plt.scatter(D*0.01, omega[1]*(2*np.pi), color = 'y')
    plt.scatter(D*0.01, omega[0]*(2*np.pi), color = 'r')
    return



D_array, md = model_with_different_D()
plt.plot(D_array, md.omega_s, label='Symmetry', color = 'r')
plt.plot(D_array, md.omega_a, label='Anti-Symmetry', color = 'y')

for D in [40, 55, 30, 50, 46, 54]:
    if D == 40:
        for num in range(1,3):
            df = extract_data_from_module(1, D, num=num)
            pL, pR = find_omega(df)
            plot(D, pL); plot(D, pR)
    if D == 55:
        df = extract_data_from_module(1, D)
        pL, pR = find_omega(df)
        plot(D, pL); plot(D, pR)
        df = extract_data_from_module(2, D)
        pL, pR = find_omega(df)
        plot(D, pL); plot(D, pR)
    if D ==30:
        df = extract_data_from_module(2, D)
        pL, pR = find_omega(df)
        plot(D, pL); plot(D, pR)
    if D ==50:
        for num in range(1,3):
            df = extract_data_from_module(2, D, num=num)
            pL, pR = find_omega(df)
            plot(D, pL); plot(D, pR)
    if D==46:
        df = extract_data_from_module(2, D, num=2)
        pL, pR = find_omega(df)
        plot(D, pL); plot(D, pR)
    if D==54:
        df = extract_data_from_module(2, D)
        pL, pR = find_omega(df)
        plot(D, pL); plot(D, pR)
plt.title(r'Compare experiment data with model ($\omega$)')
plt.ylabel(r'$\omega [1/s]$')
plt.xlabel(r'$D [m]$')
plt.legend()
plt.savefig('fft_compare_with_model')
# plt.show()
