"""
Created on Wed Aug 20 2025

@author: Tomás Altimare Bercovich

Descripción:
------------

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

#%%########################
## Definiciones Globales ##
###########################

fs = 1000 # Frecuencia de sampleo
N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
T_simulacion = N/fs # Tiempo total de simulación
n = np.arange(N) # Las N muestras equiespaciadas

#%%#########################
## Funciones del Programa ##
############################

# Función Seno
def funcion_senoidal(ff, nn, amp = 1, dc = 0, ph = 0, fs = 1000): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje Y), funcion (eje X)
    """    
    # Grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = amp * np.sin( 2 * np.pi * ff * tt + ph ) + dc
    return tt, xx

# Función para imprimir en dB
def dB(X):
    X_shift = fftshift(X) # Armo la FFT centrada
    X_abs = np.abs(X_shift)**2 # Armo el módulo al cuadrado de la FFT (PSD)(la densidad espectral)
    X_max = np.max(X_abs) if np.max(X_abs)>0 else 1 # Calculo el máximo de la funcion para poder trasladar las funciones al mismo eje
    return 20 * np.log10(X_abs / X_max)

# def zero_padding(sig, N = 1000, fs = 1000): 
#     """
#     padding(sig: señal original, N: # muestras, fs: frecuencia de muestreo)
#     """
#     zeroPadding = np.zeros(N*9) # Lleno el vector de ceros
#     zeroPadding[:len(sig)] = sig # Meto la señal dentro del vector
    
#     sig_padding = np.abs(fft(zeroPadding)) # 
    
#     # zero = np.zeros(len(sig)*9)
#     # sig_padding = np.concatenate((sig, zero))
#     # sig_padding = fft(sig_padding)
    
#     return sig_padding

def zero_padding(sig, n_zeros = 9):
    Nsig = len(sig)
    padded = np.concatenate([sig, np.zeros(n_zeros * Nsig)]) # Relleno padded con n cantidad de ceros
    return fft(padded) # Armo la densidad espectral con la fft

#%%############
## Punto (1) ##
###############
#%% Defino las variables que voy a usar
a0 = np.sqrt(2)
k0 = N/4
df = fs/N 
f1 = k0 * df
f2 = (k0 + 0.25) * df
f3 = (k0 + 0.5) * df
eje_ff = np.arange(N) * df

#%% Armo los senos
tt, x1 = funcion_senoidal(ff = f1, nn = N, amp = a0, fs = fs)
_, x2 = funcion_senoidal(ff = f2, nn = N, amp = a0, fs = fs)
_, x3 = funcion_senoidal(ff = f3, nn = N, amp = a0, fs = fs)

#%% Calculo las potenciias de cada uno
potencia_x1 = np.sum(x1 ** 2) / N
potencia_x2 = np.sum(x2 ** 2) / N
potencia_x3 = np.sum(x3 ** 2) / N

# %% Ploteo
def plotear_fig1_a_fig4():
    plt.figure(1)
    plt.title('Figura 1: Densidades espectrales (comparativa)')
    plt.grid(True)
    plt.plot(eje_ff, dB(fft(x1)), '.', color = 'red', label = 'x1 (N/4)')
    plt.plot(eje_ff, dB(fft(x2)), '.', color = 'green', label = 'x2 (N/4 + 0.25)')
    plt.plot(eje_ff, dB(fft(x3)), '.', color = 'blue', label = 'x3 (N/4 + 0.5)')
    plt.xlim(0, N/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x1: {potencia_x1:3.1f}\n"
        f"Potencia x2: {potencia_x2:3.1f}\n"
        f"Potencia x3: {potencia_x3:3.1f}\n"
    )
    plt.figtext(0.5, -0.1, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(2)
    plt.title('Figura 2: Densidad espectral de f = N/4')
    plt.grid(True)
    plt.plot(eje_ff, dB(fft(x1)), '.', color = 'red', label = 'x1 (N/4)')
    plt.xlim(0, N/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x1: {potencia_x1:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(3)
    plt.title('Figura 3: Densidad espectral de f = N/4+0.25')
    plt.grid(True)
    plt.plot(eje_ff, dB(fft(x2)), '.', color = 'green', label = 'x2 (N/4 + 0.25)')
    plt.xlim(0, N/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x2: {potencia_x2:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(4)
    plt.title('Figura 4: Densidad espectral de f = N/4+0.5')
    plt.grid(True)
    plt.plot(eje_ff, dB(fft(x3)), '.', color = 'blue', label = 'x3 (N/4 + 0.5)')
    plt.xlim(0, N/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x3: {potencia_x3:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
plotear_fig1_a_fig4()

#%%############
## Punto (2) ##
###############
#%% Verifico Parseval para los 3 senos
X1_fft = fft(x1)
X2_fft = fft(x2)
X3_fft = fft(x3)

A1 = np.sum(np.abs(x1)**2)
B1 = np.sum(np.abs(X1_fft)**2)/N
A2 = np.sum(np.abs(x2)**2)
B2 = np.sum(np.abs(X2_fft)**2)/N
A3 = np.sum(np.abs(x3)**2)
B3 = np.sum(np.abs(X3_fft)**2)/N

parseval_x1 = np.abs(A1-B1)
parseval_x2 = np.abs(A2-B2)
parseval_x3 = np.abs(A3-B3)

def verificacion_parseval():
    if (parseval_x1 < 10e-9): # Comparo con un numero pequeño y no cero debido a que Python no suele arrojar ceros (por los numeros de máquina y el redondeo)
        print("\nSe cumple parseval para x1\n")
        
    if (parseval_x2 < 10e-9):
        print("Se cumple parseval para x2\n")
    
    if (parseval_x3 < 10e-9):
        print("Se cumple parseval para x3\n")

#%%############
## Punto (3) ##
###############
N_padding = N + N*9
# eje_ff_padding = np.linspace(0, (N-1)*(df), N_padding)
eje_ff_padding = np.linspace(0, fs, N_padding, endpoint=False)

# Grafico las Densidades de Potencia con Padding
def plotear_fig5_a_fig8():
    plt.figure(5)
    plt.clf()
    plt.title('Figura 5: Densidades espectrales (comparativa) - Zero Padding')
    plt.grid(True)
    plt.plot(eje_ff_padding, dB(zero_padding(x1)), '.', color = 'red', label = 'x1 (N/4)')
    plt.plot(eje_ff_padding, dB(zero_padding(x2)), '.', color = 'green', label = 'x2 (N/4 + 0.25)')
    plt.plot(eje_ff_padding, dB(zero_padding(x3)), '.', color = 'blue', label = 'x3 (N/4 + 0.5)')
    plt.xlim(0, N/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x1: {potencia_x1:3.1f}\n"
        f"Potencia x2: {potencia_x2:3.1f}\n"
        f"Potencia x3: {potencia_x3:3.1f}\n"
    )
    plt.figtext(0.5, -0.1, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(6)
    plt.clf()
    plt.title('Figura 6: Densidad espectral de f = N/4 - Zero Padding')
    plt.grid(True)
    plt.plot(eje_ff_padding, dB(zero_padding(x1)), '.', color = 'red', label = 'x1 (N/4)')
    plt.xlim(240, 260)
    plt.ylim(-300, 10)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x1: {potencia_x1:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(7)
    plt.clf()
    plt.title('Figura 7: Densidad espectral de f = N/4+0.25 - Zero Padding')
    plt.grid(True)
    plt.plot(eje_ff_padding, dB(zero_padding(x2)), '.', color = 'green', label = 'x2 (N/4 + 0.25)')
    plt.xlim(240, 260)
    plt.ylim(-50, 2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x2: {potencia_x2:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
    plt.figure(8)
    plt.clf()
    plt.title('Figura 8: Densidad espectral de f = N/4+0.5 - Zero Padding')
    plt.grid(True)
    plt.plot(eje_ff_padding, dB(zero_padding(x3)), '.', color = 'blue', label = 'x3 (N/4 + 0.5)')
    plt.xlim(230, 270)
    plt.ylim(-320, 5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X|\u00b2 [dB]")
    plt.legend()
    # Añado una leyenda con los datos sobre la señal:
    info_fig = (
        f"Potencia x3: {potencia_x3:3.1f}\n"
    )
    plt.figtext(0.5, -0.05, info_fig, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"))
    plt.tight_layout()
    
plotear_fig5_a_fig8()