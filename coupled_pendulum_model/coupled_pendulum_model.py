import numpy as np
import matplotlib.pyplot as plt

"""
Reference:
https://www.youtube.com/watch?v=s3i0QRs2gBI&ab_channel=K%21W%21BiRD%E5%A5%87%E7%95%B0%E9%B3%A5

______________________D________________________
         \ th:                    /
          \  :                   /
           s :                  s
            \:                 /
             \________d_______/
             |                |
             |                |
             L                L
             |                |  
             |                |
             1                2
"""
g = 9.8
# L = 0.41

class move_1_coupled_pendulum:
    
    def __init__(self, D, s, d, L):
        self.D = D
        self.s = s
        self.d = d
        self.L = L
        self.th = np.arcsin(0.5*(D-d)/s)
        self.omega()
        self.compute_N()
        return   

    def omega(self):
        omega_s = np.sqrt(g/(self.L+self.s*np.cos(self.th)))
        omega_a = np.sqrt(g/(self.L+(0.5*self.d*self.s*np.cos(self.th))/\
            (0.5*self.d+self.s*np.cos(self.th))))
        self.omega_s = omega_a
        self.omega_a = omega_s
        return 
    
    def compute_N(self):
        N = (self.omega_s+self.omega_a)/(self.omega_s-self.omega_a)
        self.N = N
        return
    
    