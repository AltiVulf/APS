#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:35:03 2025

@author: Tomás Altimare Bercovich
"""

#%% Llamo a las bibliotecas que voy a usar
import numpy as np
import matplotlib.pyplot as plt
from spicy import signal

#%% Defino las variables que voy a usar
N = 8
X = np.zeros(N, dtype = np.complex128)

fs = 1000
ts = 1/fs

n = np.arange(N) * ts
x = (3 * np.sin(n * np.pi/2) + 4)

#%% Armo la serie
for k in range(N): 
    for n in range(N): # range(N) va de 0 a N-1
        X[k] += x[n] + np.exp((-1j * k * 2 * np.pi / N) * n)

print(X)

plt.figure(1)
plt.title('Espectro (DFT)')
plt.xlabel("Indice k")
plt.ylabel("|X[k]|")
plt.plot(np.arange(N), np.abs(X), 'o--')
markerline, stemlines, baseline = plt.stem(np.arange(N), np.abs(X))
plt.setp(markerline, color='blue')
plt.setp(stemlines, color='blue')
plt.setp(baseline, color='blue')
plt.xlim(0, N-1)
plt.grid(True)

