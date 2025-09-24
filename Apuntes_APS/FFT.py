"""
Created on Wed Aug 27 2025

@author: Tomás Altimare Bercovich

Descripción:
------------


"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from numpy.fft import fft

#%% Defino las variables que voy a usar
N = 1000 # Cantidad de muestras
fs = N # Frecuencia de sampleo
df = fs/N # Resolución espectal
ts = 1/fs # Tiempo/Periodo de sampleo

Vmax = 1
dc = 0
frec = 1

#%% Defino el seno
def funcion_senoidal(ff, nn, vmax = 1, dc = 0, ph = 0, fs = 2): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje Y), funcion (eje X)
    """
    ts = 1/fs # Tiempo/Periodo de sampleo
    
    # grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = vmax * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    return tt, xx

#%% Empiezo a armar las FFT

t1, x1 = funcion_senoidal(ff = (N/4)*df, nn = N, fs = fs)
t2, x2 = funcion_senoidal(ff = ((N/4)+1)*df, nn = N, fs = fs)
t3, x3 = funcion_senoidal(ff = ((N/4)+0.5)*df, nn = N, fs = fs)


# Armo las FFT
X1_fft = np.fft.fft(x1)
X1_abs = np.abs(X1_fft)
X1_ang = np.angle(X1_fft)

X2_fft = np.fft.fft(x2)
X2_abs = np.abs(X2_fft)
X2_ang = np.angle(X2_fft)

X3_fft = np.fft.fft(x3)
X3_abs = np.abs(X3_fft)
X3_ang = np.angle(X3_fft)

n = np.arange(N)
eje_frec = np.arange(N)*df # Grilla de sampleo en frecuencia

plt.figure(1)
plt.clf()
plt.plot(eje_frec, X1_abs, 'x', label = 'X1 abs (N/4)', color='firebrick')
plt.plot(eje_frec, np.log10(X1_abs)*20, 'x', label = 'X1 abs en dB', color = 'firebrick')
plt.plot(eje_frec, X2_abs, '.', label = 'X2 abs (N/4+1)', color='midnightblue')
plt.plot(eje_frec, np.log10(X2_abs)*20, '.', label = 'X2 abs en dB', color = 'midnightblue')
plt.plot(eje_frec, X3_abs, '+', label = 'X3 abs (N/4+0,5)', color='forestgreen')
plt.plot(eje_frec, np.log10(X3_abs)*20, '+', label = 'X3 abs en dB', color = 'forestgreen')
plt.title('Espectro (FFT)')
plt.xlabel("Frecuencia Normaliada (xπ rad/sample)")
plt.ylabel("Amplitud [dB]")
plt.xlim(0, fs/2)
plt.grid(True)
plt.legend() # Muestra las etiquetas que le ponemos a los plot
plt.show() # Sirve más que nada en Jupyter Notebook