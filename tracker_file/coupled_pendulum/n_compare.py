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
def find_n(df, start, end):
    peaks, _ = find_peaks(df.data_L['x'])
    start_index = np.searchsorted(df.data_L['t'], start)  
    end_index = np.searchsorted(df.data_L['t'], end)
    n_p_L=len(peaks[(peaks >= start_index) & (peaks <= end_index)]) 
    return n_p_L*2-8
def plot(D, n):
    plt.scatter(D*0.01, n, color = 'black', marker='x')
    return

df = extract_data_from_module(1, 40)
n_p = find_n(df, 10, 34)
plot(40, n_p)
df = extract_data_from_module(1, 40, num=2)
n_p = find_n(df, 10, 34)
plot(40, n_p)
df = extract_data_from_module(1, 40, num=3)
n_p = find_n(df, 11, 35)
plot(40, n_p)
df = extract_data_from_module(1, 55)
n_p = find_n(df, 0.42, 88)
plot(55, n_p)
df = extract_data_from_module(2, 30)
n_p = find_n(df, 11, 35)
plot(30, n_p)
df = extract_data_from_module(2, 46)
n_p = find_n(df, 15.5, 45)
plot(46, n_p)
df = extract_data_from_module(2, 46, 2)
n_p = find_n(df, 14, 43)
plot(46, n_p)
df = extract_data_from_module(2, 50)
n_p = find_n(df, 14, 46)
plot(50, n_p)
df = extract_data_from_module(2, 50, 2)
n_p = find_n(df, 20, 60)
plot(50, n_p)
df = extract_data_from_module(2, 50, 3)
n_p = find_n(df, 18, 57)
plot(50, n_p)
df = extract_data_from_module(2, 54)
n_p = find_n(df, 20, 59)
plot(54, n_p)
df = extract_data_from_module(2, 55)
n_p = find_n(df, 0.1, 108)
plot(55, n_p)
D_array, md = model_with_different_D()
plt.plot(D_array, md.N, label='Model', color = 'r')
plt.title('Compare experiment data with model (N)')
plt.ylabel(r'Number of oscillation')
plt.xlabel(r'$D [m]$')
plt.legend()
plt.savefig('./Figures/n_compare')
# plt.show()
