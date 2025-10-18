"""
Created on Wed Aug 20 2025

@author: Tomás Altimare Bercovich

Descripción:
------------

"""

#%% Importo los módulos y bibliotecas que voy a utilizar
import numpy as np
import matplotlib.pyplot as plt
import TS1_Sintesis_de_Señales as TS1 # Importo el TS1 para poder usar las señales
from numpy.fft import fft, fftshift
import scipy.signal.windows as window

#%%########################
## Definiciones Globales ##
###########################

fs = 1000 # Frecuencia de sampleo
N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
T_simulacion = N/fs # Tiempo total de simulación
n = np.arange(N) # Las N muestras equiespaciadas
tt = n/fs # Grilla temporal de sampleo
df = fs/N

#%%#########################
## Funciones del Programa ##
############################

# def funcion_senoidal(ff, nn, amp = 1, dc = 0, ph = 0, fs = 2, ruido = 0): 
#     """
#     funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
#     - Output: tiempo (eje Y), funcion (eje X)
#     """    
#     # Grilla de sampleo temporal
#     n = np.arange(nn)
#     tt = n / fs
    
#     xx = amp * np.sin(ff * tt + ph) + dc + ruido
#     return tt, xx

def funcion_senoidal(ff, nn, vmax = 1, dc = 0, ph = 0, fs = 2, ruido = 0): 
    """
    funcion_senoidal(frec, #_muestras, amplitud, offset, fase, frec_f_wav)
    - Output: tiempo (eje Y), funcion (eje X)
    """    
    # Grilla de sampleo temporal
    n = np.arange(nn)
    tt = n / fs
    
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph ) + dc + ruido
    return tt, xx

# Función para imprimir en dB
def dB(X):
    X_shift = fftshift(X) # Armo la FFT centrada
    X_abs = np.abs(X_shift) # Armo el módulo de la FFT (PSD)(la densidad espectral)
    X_max = np.max(X_abs) if np.max(X_abs)>0 else 1 # Calculo el máximo de la funcion para poder trasladar las funciones al mismo eje
    return 20 * np.log10(X_abs / X_max)

# Función para generar distintos tipos de ruido
def ruido(tipo, long = N, v_min = -1, v_max = 1, desv_med = 0, desv_est = 0.1):
    if tipo == 'random':
        ruido = np.random.randn(long)
    elif tipo == 'uniforme':
        ruido = np.random.uniform(v_min, v_max, long)
    elif tipo == 'normal':
        ruido = np.random.normal(desv_med, desv_est, long)
    return ruido

plt.close("all")

#%%############
## Punto (1) ##
###############
# Defino las variables que voy a usar
a0 = np.sqrt(2)
SNR = 10
omega0 = N/4 #np.pi/2
var = 1
v_med = 0
fr = np.random.uniform(low = -2, high = 2)
omega1 = omega0 + fr * 2 * np.pi/N
#omega1 = omega0 + fr * 

na = np.random.normal(v_med, var)

pot_ruido = a0**2 / (2*10**(SNR/10))
print(f"Potencia ruido: {pot_ruido:3.1f}")
# var_ruido = np.var(ruido)
# print(f"Ruido: {var_ruido:3.3f}")

# tt_ruido, xx_ruido = funcion_senoidal(ff = omega1, nn = N, amp = a0, fs = fs, ruido = na)
# tt, xx = funcion_senoidal(ff = omega1, nn = N, amp = a0, fs = fs)

t_sen, x_sen = funcion_senoidal(ff = fs/4, nn = N)
x_sen = fft(np.abs(x_sen))

# x_sen_ruido = x_sen + ruido
# print(f"Varianza x_sen: {np.var(x_sen):3.1f}")

# N_ruido = ruido('normal', desv_est = np.sqrt(pot_ruido))

#%%##################
## Matriz de Senos ##
#####################
R = 200
#%% VENTANAS
flattop = window.flattop(N).reshape((-1,1))
blackmanharris = window.blackmanharris(N).reshape((-1,1))
hamming = window.hamming(N).reshape((-1,1))
taylor = window.taylor(N).reshape((-1,1))
bohman = window.bohman(N).reshape((-1,1))

# %% Vectores columna y fila de tiempo y frecuencia
tt_vector = np.arange(N)/fs
ff_vector = np.random.uniform(-2, 2, R) * df # Frecuencias random
tt_columnas = tt_vector.reshape((-1,1)) # Tamaño N (vector COLUMNA)
ff_filas = ff_vector.reshape((1,-1)) # Tamaño R (vector FILA)

# %% Matrices de tiempo, frecuencia y ruido
TT_sen = np.tile(tt_columnas, (1, R)) # Tamaño NxR (matriz)
FF_sen = np.tile(ff_filas, (N, 1)) # Tamaño RxN (matriz)
matriz_ruido = np.random.normal(loc = 0, scale = np.sqrt(pot_ruido), size = (N,R))

# %% Armo el seno con y sin ruido
xx_sen = a0 * np.sin (2 * np.pi * (N/4+fr) * df * TT_sen)
xx_sen_ruido = a0 * np.sin (2 * np.pi * (N/4+fr+FF_sen) * df * TT_sen) + matriz_ruido 

# %% FFT's frecuencia FIJA
XX_sen = fft(xx_sen, n = 10*N, axis = 0)/N
XX_sen_ruido = fft(xx_sen_ruido, n = 10*N, axis = 0)/N

XX_flattop = xx_sen_ruido * flattop # Matriz de senos RxN con ruido (ventaneado)
XX_blackmanharris = xx_sen_ruido * blackmanharris
XX_hamming = xx_sen_ruido * hamming

# %% FFT's frecuencia VARIABLE
xx_sen_ruido_frec = a0 * np.sin (2 * np.pi * (N/4+fr+FF_sen) * df * TT_sen) + matriz_ruido
XX_vent_frec = fft(xx_sen_ruido_frec, n = 10*N, axis = 0)/N

XX_flattop_frec = xx_sen_ruido_frec * flattop # Matriz de senos RxN con ruido (ventaneado)
XX_blackmanharris_frec = xx_sen_ruido_frec * blackmanharris
XX_hamming_frec = xx_sen_ruido_frec * hamming

# %% Armo las fetas frecuencia FIJA
a_sin_ventana = XX_sen_ruido[N//4,:]
a_flattop = XX_flattop[N//4,:] # El // Fuerza una división entera
a_blackmanharris = XX_blackmanharris[N//4,:]
a_hamming = XX_hamming[N//4,:]

# %% Armo las fetas recuencia VARIABLE
a_sin_ventana_var = XX_vent_frec[N//4,:]
a_flattop_var = XX_flattop_frec[N//4,:] # El // Fuerza una división entera
a_blackmanharris_var = XX_blackmanharris_frec[N//4,:]
a_hamming_var = XX_hamming_frec[N//4,:]

bins = 10
transparencia = 0.5
"""
plt.figure(1)
plt.title('Figura 1: Señales (sin dB)')
plt.plot(a_sin_ventana, alpha = transparencia, color = 'red', label = 'Rectangular')
plt.plot(a_flattop, alpha = transparencia, color = 'blue', label = 'Flattop')
plt.plot(a_blackmanharris, alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.plot(a_hamming, alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()

plt.figure(2)
plt.title('Figura 2: Histogramas (sin dB)')
plt.hist(a_sin_ventana, bins = bins, alpha = transparencia, color = 'red', label = 'Rectangular')
plt.hist(a_flattop, bins = bins, alpha = transparencia, color = 'blue', label = 'Flattop')
plt.hist(a_blackmanharris, bins = bins, alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.hist(a_hamming, bins = bins, alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()

plt.figure(3)
plt.title('Figura 3: Señales (CON dB)')
plt.plot(dB(a_sin_ventana), alpha = transparencia, color = 'red', label = 'Rectangular')
plt.plot(dB(a_flattop), alpha = transparencia, color = 'blue', label = 'Flattop')
plt.plot(dB(a_blackmanharris), alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.plot(dB(a_hamming), alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()
"""
plt.figure(1)
plt.title('Figura 1: Señales (sin dB)')
plt.plot(a_sin_ventana_var, alpha = transparencia, color = 'red', label = 'Rectangular')
plt.plot(a_flattop_var, alpha = transparencia, color = 'blue', label = 'Flattop')
plt.plot(a_blackmanharris_var, alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.plot(a_hamming_var, alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()

plt.figure(2)
plt.title('Figura 2: Histogramas (sin dB)')
plt.hist(a_sin_ventana_var, bins = bins, alpha = transparencia, color = 'red', label = 'Rectangular')
plt.hist(a_flattop_var, bins = bins, alpha = transparencia, color = 'blue', label = 'Flattop')
plt.hist(a_blackmanharris_var, bins = bins, alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.hist(a_hamming_var, bins = bins, alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()

plt.figure(3)
plt.title('Figura 3: Señales (CON dB)')
plt.plot(dB(a_sin_ventana_var), alpha = transparencia, color = 'red', label = 'Rectangular')
plt.plot(dB(a_flattop_var), alpha = transparencia, color = 'blue', label = 'Flattop')
plt.plot(dB(a_blackmanharris_var), alpha = transparencia, color = 'green', label = 'BlackmanHarris')
plt.plot(dB(a_hamming_var), alpha = transparencia, color = 'yellow', label = 'Hamming')
plt.legend()

# # %% Ploteos
# plt.figure(1)
# plt.title('Figura 1: Seno con Ruido')
# plt.grid(True)
# plt.plot(t_sen, dB(x_sen), color = 'blue')
# # plt.plot(t_sen, dB(x_sen_ruido), color = 'red')
# #plt.plot(tt, xx, color = 'red')
# #plt.xlim(0, T_simulacion)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("F(t)")
# plt.tight_layout()

# plt.figure(2)
# plt.title('Figura 2: Seno en dB')
# plt.grid(True)
# plt.plot(t_sen, 20*np.log10(np.abs(x_sen)), color = 'teal')
# #plt.plot(tt, xx, color = 'red')
# plt.xlim(0, T_simulacion)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("F(t)")
# plt.tight_layout()

# # 1: Ponerle SNR a X
# plt.figure(2)
# plt.title('Figura 1: Seno con Ruido')
# plt.grid(True)
# plt.plot(tt_ruido, xx_ruido, color = 'teal')
# #plt.plot(tt, xx, color = 'red')
# plt.xlim(0, T_simulacion)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("F(t)")
# plt.tight_layout()


