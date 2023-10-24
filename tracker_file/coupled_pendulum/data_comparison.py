import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    return move_1_coupled_pendulum(D=D_array,s=s,d=d,L=L) 


def extract_data(move, D, LR):  # D in cm
    data_dict ={
        
    }
    path = f"./move_{move}_{D}cm/"
    df = pd.read_excel()
    
    return


# def extract_data_from_path(path):
#     df = pd.read_excel(path, engine='openpyxl')
#     raw_data = np.array(df)
#     raw_data[:, 5] = np.deg2rad(raw_data[:, 5])
#     columns = ['t', 'x', 'y', 'r', 'v', 'theta']
#     data = {col: raw_data[:, idx] for idx, col in enumerate(columns)}
#     return data
# data_L = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/couple_3_L.xlsx")
# data_R = extract_data_from_path("./tracker_file/coupled_pendulum/move_1_40cm/couple_3_R.xlsx")

md = model_with_different_D()