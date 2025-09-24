#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:43:03 2025

@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

# %% Defino Variables Globales
N = 1000 # Cantidad de muestras

#%%########
## Ruido ##
###########
def ruido(tipo, long = N, v_min = -1, v_max = 1, desv_med = 0, desv_est = 0.1):
    if tipo == 'random':
        ruido = np.random.randn(long)
    elif tipo == 'uniforme':
        ruido = np.random.uniform(v_min, v_max, long)
    elif tipo == 'normal':
        ruido = np.random.normal(desv_med, desv_est, N)
    return ruido

ruido_random = ruido(tipo = 'random')
ruido_uniforme = ruido(tipo = 'uniforme')
ruido_normal = ruido(tipo = 'normal')

plt.figure(1)
plt.clf()
plt.title('Figura 1: Ruido Random')
plt.grid(True)
plt.plot(ruido_random, '.-', label = 'Ruido Random', color = 'cornflowerblue')
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.xlim(0, N)
plt.ylim(min(ruido_random)-0.1, max(ruido_random)+0.1)
plt.tight_layout()

plt.figure(2)
plt.clf()
plt.title('Figura 2: Ruido Uniforme')
plt.grid(True)
plt.plot(ruido_uniforme, '.-', label = 'Ruido Uniforme', color = 'cornflowerblue')
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.xlim(0, N)
plt.ylim(min(ruido_uniforme)-0.1, max(ruido_uniforme)+0.1)
plt.tight_layout()

plt.figure(3)
plt.clf()
plt.title('Figura 3: Ruido Normal')
plt.grid(True)
plt.plot(ruido_normal, '.-', label = 'Ruido Normal', color = 'cornflowerblue')
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.xlim(0, N)
plt.ylim(min(ruido_normal)-0.01, max(ruido_normal)+0.01)
plt.tight_layout()