
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows

# Parámetros
N = 1000      #muestras
R = 200       #realizaciones
fs = N        #frecuencia de muestreo 
f0 = fs / 4   #frecuencia de la senoidal
resolucionEspectral=fs/N
frecRandom= np.random.uniform(low=-2, high=2, size=R)*resolucionEspectral #ojo que es un vector
frecRandom = frecRandom.reshape(1, R)

#tiempo en vector y matriz
Ts = 1/fs
t = np.linspace(0, (N-1)*Ts, N)            #vector de tiempo 
t_mat = np.tile(t.reshape(N, 1), (1, R))   #matriz 1000x200 (columnas = realizaciones)

#señal senoidal en matriz
a0 = np.sqrt(2)
señalMatriz = a0 * np.sin(2*np.pi*(f0+frecRandom)*t_mat)   # matriz 1000x200

def SNRvarianza(señal, SNRdb):
    potSeñal = np.mean(señal**2)
    return potSeñal / (10**(SNRdb / 10))

Rvar = SNRvarianza(señalMatriz, 10)

#rudio pero matriz
ruidoMatriz = np.random.normal(0, np.sqrt(Rvar), (N, R))

#señal+ruido
señalConRuidoMatriz = señalMatriz + ruidoMatriz