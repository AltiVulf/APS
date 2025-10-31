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
from scipy import signal as sig 
import scipy.io as sio
import scipy.signal.windows as window
from scipy.signal import periodogram
from scipy.io.wavfile import write

#%%########################
## Definiciones Globales ##
###########################

fs = 1000 # Frecuencia de Sampleo
N = 1000 # Cantidad de muestras
ts = 1/fs # Tiempo entre muestras
T_simulacion = N/fs # Tiempo total de simulación
n = np.arange(N) # Las N muestras equiespaciadas

#%%#########################
## Funciones del Programa ##
############################


#%%###################
## Filtro Analógico ##
######################
# %% -- Parámetros de Plantilla de Diseño del filtro --
wp = 1 # frecuencia de corte o de paso (rad/s) o pulsación de paso
ws = 5 # frecuencia o pulsación de stop (rad/s)

alpha_p = 1 # atenuación máxima a la wp, alpha_max, perdidas de banda de paso
alpha_s = 40 # atenuación minima a la ws, alpha_min, mínima atenuación requerida en banda de paso

# Aproximadores de módulo:
f_aprox = ['butter', 'cheby1', 'cheby2', 'ellip', 'cauer']

# %% -- Diseño del filtro (en sampleos) --
for i in range(4, 5):
    b, a = sig.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, 
                         analog = True, ftype = f_aprox[i], output = 'ba')
    # Respuesta en frecuencia del filtro (calculada) 
    w, h = sig.freqs(b, a) # worN = np.logspace(2,-2,1000)
    phase = np.unwrap(np.angle(h)) # fase del grupo
    gd = -np.diff(phase) / np.diff(w) # retardo

    # %%  -- Calculo polos y ceros --
    z, p, k = sig.tf2zpk(b, a) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]
    
    # %% Grafico
    # Magnitud
    plt.figure(1)
    plt.plot(w, 20*np.log10(abs(h)), label = f"{f_aprox[i]}")
    plt.title('Figura 1: Respuesta en Magnitud')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Fase
    plt.figure(2)
    plt.plot(w, np.degrees(phase), label = f"{f_aprox[i]}")
    plt.title('Figura 2: Fase')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('Fase [°]')
    plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Retardo de grupo
    plt.figure(3)
    plt.plot(w[:-1], gd, label = f"{f_aprox[i]}")
    plt.title('Figura 3: Retardo de Grupo')
    plt.xlabel('Pulsación angular [r/s]')
    plt.ylabel('τg [s]')
    plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Diagrama de polos y ceros
    plt.figure(4)
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label= f'Polos de {f_aprox[i]}')
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'Ceros de {f_aprox[i]}')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.title('Figura 4: Diagrama de Polos y Ceros (plano s)')
    plt.xlabel('σ [rad/s]')
    plt.ylabel('jω [rad/s]')
    plt.legend()
    plt.grid(True)

sos = sig.tf2sos(b, a, analog = True)

#%%#################
## Filtro Digital ##
####################
# %% -- Parámetros de Plantilla de Diseño del filtro --
i=0
fpaso = [100, 135] # frecuencia de corte [Hz]
fstop = [99, 140] # frecuencia de stop [Hz]

alpha_pd = 1 # alpha maximo
alpha_sd = 40 # alpha minimo

# NOTA: En filtros digitales, no olvidar de poner la frecuencia de muestreo
for i in range(4, 5):
    dig_sos = sig.iirdesign(wp = fpaso, ws = fstop, gpass = alpha_pd, gstop = alpha_sd, 
                         analog = False, ftype = f_aprox[i], output = 'sos', fs = fs)
    # Respuesta en frecuencia del filtro (calculada) 
    w_dig, h_dig = sig.freqz_sos(dig_sos, fs = fs) # omega en Hz
    phase = np.unwrap(np.angle(h_dig)) # fase del grupo
    
    w_dig_rad = w_dig / ((fs/2) * np.pi)
    gd = -np.diff(phase) / np.diff(w_dig_rad) # retardo

    # %%  -- Calculo polos y ceros --
    z_dig, p_dig, k_dig = sig.sos2zpk(dig_sos) # Zpk = [ [z0,z1,...,zn], [p0,p1,...,pn], k]
    
    # %% Grafico
    # Magnitud
    plt.figure(5)
    plt.plot(w_dig, 20*np.log10(abs(h_dig)), label = f"{f_aprox[i]}")
    # plt.plot(w_dig, 20*np.log10(abs(h_dig)), label = f"{f_aprox[i]}")
    plt.title('Figura 1: Respuesta en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Fase
    plt.figure(6)
    plt.plot(w_dig, np.degrees(phase), label = f"{f_aprox[i]}")
    # plt.plot(w_dig, np.degrees(phase), label = f"{f_aprox[i]}")
    plt.title('Figura 2: Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [°]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Retardo de grupo
    plt.figure(7)
    plt.plot(w_dig[:-1], gd, label = f"{f_aprox[i]}")
    plt.title('Figura 3: Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [s]')
    # plt.xlim(0.1, 10)
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
    # Diagrama de polos y ceros
    plt.figure(8)
    plt.plot(np.real(p_dig), np.imag(p_dig), 'x', markersize=10, label= f'Polos de {f_aprox[i]}')
    if len(z_dig) > 0:
        plt.plot(np.real(z_dig), np.imag(z_dig), 'o', markersize=10, fillstyle='none', label=f'Ceros de {f_aprox[i]}')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.title('Figura 4: Diagrama de Polos y Ceros (plano Z)')
    plt.xlabel('σ [rad/s]')
    plt.ylabel('jω [rad/s]')
    plt.legend()
    plt.grid(True)
    
